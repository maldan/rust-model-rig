//! Configured IK: two-bone analytic / CCD (N) + pole plane + tip follows target rotation.

use glam::{Quat, Vec3};
use mega_render::{LineOpts, Scene};

use crate::ik::{ccd_rotate_joint, quat_from_matrix};
use crate::rig::{AppMode, BoneId, IkChain, RigDocument};

const CCD_ITERS: u32 = 28;

/// Run all enabled chains (Pose).
pub fn evaluate_ik_chains(scene: &mut Scene, rig: &RigDocument) {
    if rig.mode != AppMode::Pose {
        return;
    }
    let chains: Vec<IkChain> = rig.ik_chains.iter().filter(|c| c.enabled).cloned().collect();
    for chain in &chains {
        solve_chain(scene, rig, chain);
    }
}

fn solve_chain(scene: &mut Scene, rig: &RigDocument, chain: &IkChain) {
    if chain.bones.is_empty() {
        return;
    }
    let tip = chain.tip;
    let target_pos = scene
        .world_matrix(chain.target.node)
        .transform_point3(Vec3::ZERO);
    let pole = scene
        .world_matrix(chain.pole.node)
        .transform_point3(Vec3::ZERO);

    // Deterministic base pose, then solve positions, then tip rotation from target.
    restore_chain_bind(scene, rig, chain);

    if chain.bones.len() == 2 && chain.lengths.len() >= 2 {
        solve_two_bone(scene, chain, tip, target_pos, pole, chain.pole_angle);
    } else {
        solve_ccd(scene, chain, tip, target_pos, pole, chain.pole_angle);
    }

    apply_tip_rotation(scene, chain);
}

fn restore_chain_bind(scene: &mut Scene, rig: &RigDocument, chain: &IkChain) {
    for &id in &chain.bones {
        let Some(bind) = rig.bind_locals.get(&id.node.key()).copied() else {
            continue;
        };
        if let Some(n) = scene.nodes.get_mut(id.node) {
            n.local.rotation = bind.rotation;
        }
    }
    // Tip starts from bind; overwritten by target rotation after solve.
    if let Some(bind) = rig.bind_locals.get(&chain.tip.node.key()).copied() {
        if let Some(n) = scene.nodes.get_mut(chain.tip.node) {
            n.local.rotation = bind.rotation;
        }
    }
}

fn solve_two_bone(
    scene: &mut Scene,
    chain: &IkChain,
    tip: BoneId,
    target: Vec3,
    pole: Vec3,
    pole_angle: f32,
) {
    let root = chain.bones[0];
    let mid = chain.bones[1];
    let root_pos = scene.world_matrix(root.node).transform_point3(Vec3::ZERO);
    let len1 = chain.lengths[0].max(1e-4);
    let len2 = chain.lengths[1].max(1e-4);
    let (mid_pos, tip_pos) = two_bone_positions(root_pos, len1, len2, target, pole, pole_angle);
    swing_bone_to(scene, root, mid, mid_pos);
    // Roll around the thigh even when the leg is straight (knee offset ~ 0).
    apply_pole_roll(scene, root, mid, chain.pole_ref_local, pole, pole_angle);
    swing_bone_to(scene, mid, tip, tip_pos);
}

fn solve_ccd(
    scene: &mut Scene,
    chain: &IkChain,
    tip: BoneId,
    target: Vec3,
    pole: Vec3,
    pole_angle: f32,
) {
    for _ in 0..CCD_ITERS {
        for i in (0..chain.bones.len()).rev() {
            ccd_rotate_joint(scene, chain.bones[i], tip, target, 1.0);
        }
    }
    // Single stable pole pass: twist around reach axis (no mid-CCD fight).
    let mid = chain.bones[chain.bones.len() / 2];
    align_chain_to_pole(scene, chain.bones[0], mid, tip, target, pole, pole_angle, chain.pole_ref_local);
    for _ in 0..(CCD_ITERS / 2) {
        for i in (0..chain.bones.len()).rev() {
            ccd_rotate_joint(scene, chain.bones[i], tip, target, 1.0);
        }
    }
    align_chain_to_pole(scene, chain.bones[0], mid, tip, target, pole, pole_angle, chain.pole_ref_local);
}

/// Tip world rotation = target world rotation * offset (offset captured at Create IK).
fn apply_tip_rotation(scene: &mut Scene, chain: &IkChain) {
    let target_world = quat_from_matrix(scene.world_matrix(chain.target.node));
    let desired = (target_world * chain.tip_rot_offset).normalize();
    let parent_world = scene
        .nodes
        .get(chain.tip.node)
        .and_then(|n| n.parent)
        .map(|p| scene.world_matrix(p))
        .unwrap_or(glam::Mat4::IDENTITY);
    let parent_rot = quat_from_matrix(parent_world);
    let Some(node) = scene.nodes.get_mut(chain.tip.node) else {
        return;
    };
    node.local.rotation = (parent_rot.inverse() * desired).normalize();
}

/// Swing `bone` so its child joint moves toward `desired_child` (preserves twist).
fn swing_bone_to(scene: &mut Scene, bone: BoneId, child: BoneId, desired_child: Vec3) {
    let origin = scene.world_matrix(bone.node).transform_point3(Vec3::ZERO);
    let cur_child = scene.world_matrix(child.node).transform_point3(Vec3::ZERO);
    let from = (cur_child - origin).normalize_or_zero();
    let to = (desired_child - origin).normalize_or_zero();
    if from.length_squared() < 1e-10 || to.length_squared() < 1e-10 {
        return;
    }
    let dot = from.dot(to).clamp(-1.0, 1.0);
    if dot > 0.999999 {
        return;
    }
    let q = Quat::from_rotation_arc(from, to);
    apply_world_delta_rot(scene, bone, q);
}

fn apply_world_delta_rot(scene: &mut Scene, bone: BoneId, world_delta: Quat) {
    let parent_world = scene
        .nodes
        .get(bone.node)
        .and_then(|n| n.parent)
        .map(|p| scene.world_matrix(p))
        .unwrap_or(glam::Mat4::IDENTITY);
    let parent_rot = quat_from_matrix(parent_world);
    let Some(node) = scene.nodes.get_mut(bone.node) else {
        return;
    };
    let world_r = parent_rot * node.local.rotation;
    node.local.rotation = (parent_rot.inverse() * (world_delta * world_r).normalize()).normalize();
}

fn apply_pole_roll(
    scene: &mut Scene,
    bone: BoneId,
    child: BoneId,
    ref_local: Vec3,
    pole: Vec3,
    pole_angle: f32,
) {
    if ref_local.length_squared() < 1e-12 {
        return;
    }
    let origin = scene.world_matrix(bone.node).transform_point3(Vec3::ZERO);
    let child_pos = scene.world_matrix(child.node).transform_point3(Vec3::ZERO);
    let aim = (child_pos - origin).normalize_or_zero();
    if aim.length_squared() < 1e-8 {
        return;
    }
    let rot = quat_from_matrix(scene.world_matrix(bone.node));
    let from = reject(rot * ref_local, aim);
    let mut to = reject(pole - origin, aim);
    if pole_angle.abs() > 1e-6 {
        to = Quat::from_axis_angle(aim, pole_angle.to_radians()) * to;
    }
    twist_dir(scene, bone, aim, from, to);
}

fn twist_dir(scene: &mut Scene, bone: BoneId, axis: Vec3, from: Vec3, to: Vec3) {
    if from.length_squared() < 1e-10 || to.length_squared() < 1e-10 {
        return;
    }
    let from = from.normalize();
    let to = to.normalize();
    let mut angle = from.dot(to).clamp(-1.0, 1.0).acos();
    if from.cross(to).dot(axis) < 0.0 {
        angle = -angle;
    }
    if angle.abs() < 1e-6 {
        return;
    }
    apply_world_delta_rot(scene, bone, Quat::from_axis_angle(axis, angle));
}

fn align_chain_to_pole(
    scene: &mut Scene,
    root: BoneId,
    mid: BoneId,
    tip: BoneId,
    target: Vec3,
    pole: Vec3,
    pole_angle: f32,
    ref_local: Vec3,
) {
    let root_pos = scene.world_matrix(root.node).transform_point3(Vec3::ZERO);
    let mid_pos = scene.world_matrix(mid.node).transform_point3(Vec3::ZERO);
    let tip_pos = scene.world_matrix(tip.node).transform_point3(Vec3::ZERO);
    let mut axis = (target - root_pos).normalize_or_zero();
    if axis.length_squared() < 1e-8 {
        axis = (tip_pos - root_pos).normalize_or_zero();
    }
    if axis.length_squared() < 1e-8 {
        return;
    }

    let mut from = reject(mid_pos - root_pos, axis);
    if from.length_squared() < 1e-8 && ref_local.length_squared() > 1e-12 {
        let rot = quat_from_matrix(scene.world_matrix(root.node));
        from = reject(rot * ref_local, axis);
    }
    let to = pole_in_plane(root_pos, target, pole, pole_angle);
    twist_dir(scene, root, axis, from, to);
}

fn pole_in_plane(root: Vec3, target: Vec3, pole: Vec3, pole_angle: f32) -> Vec3 {
    let dir = (target - root).normalize_or_zero();
    if dir.length_squared() < 1e-8 {
        return Vec3::ZERO;
    }
    let mut bend = reject(pole - root, dir);
    if bend.length_squared() < 1e-10 {
        return Vec3::ZERO;
    }
    if pole_angle.abs() > 1e-6 {
        bend = Quat::from_axis_angle(dir, pole_angle.to_radians()) * bend;
    }
    bend
}

fn reject(v: Vec3, axis: Vec3) -> Vec3 {
    v - axis * v.dot(axis)
}

fn two_bone_positions(
    root: Vec3,
    len1: f32,
    len2: f32,
    target: Vec3,
    pole: Vec3,
    pole_angle: f32,
) -> (Vec3, Vec3) {
    let mut to_t = target - root;
    let mut dist = to_t.length();
    if dist < 1e-6 {
        to_t = Vec3::Y * (len1 + len2);
        dist = to_t.length();
    }
    let max_r = len1 + len2;
    let min_r = (len1 - len2).abs() + 1e-4;
    let dist_c = dist.clamp(min_r, max_r - 1e-4);
    let dir = to_t / dist;

    let mut bend = pole_in_plane(root, target, pole, pole_angle);
    if bend.length_squared() < 1e-10 {
        bend = orphan_perp(dir);
    }
    let bend = bend.normalize();

    let cos_a = ((len1 * len1 + dist_c * dist_c - len2 * len2) / (2.0 * len1 * dist_c))
        .clamp(-1.0, 1.0);
    let sin_a = (1.0 - cos_a * cos_a).max(0.0).sqrt();

    let mid = root + (dir * cos_a + bend * sin_a) * len1;
    let tip = root + dir * dist_c;
    (mid, tip)
}

fn orphan_perp(dir: Vec3) -> Vec3 {
    let mut n = dir.cross(Vec3::Y);
    if n.length_squared() < 1e-10 {
        n = dir.cross(Vec3::X);
    }
    n.normalize_or_zero()
}

pub fn draw_ik_helpers(scene: &mut Scene, rig: &RigDocument) {
    if !rig.show_skeleton {
        return;
    }
    for chain in &rig.ik_chains {
        if !chain.enabled || chain.bones.is_empty() {
            continue;
        }
        let tip = scene
            .world_matrix(chain.tip.node)
            .transform_point3(Vec3::ZERO);
        let target = scene
            .world_matrix(chain.target.node)
            .transform_point3(Vec3::ZERO);
        let mid_id = chain.bones[chain.bones.len() / 2];
        let mid = scene.world_matrix(mid_id.node).transform_point3(Vec3::ZERO);
        let pole = scene
            .world_matrix(chain.pole.node)
            .transform_point3(Vec3::ZERO);

        scene.debug.line(
            tip,
            target,
            LineOpts::color([1.0, 0.55, 0.15, 0.85])
                .width(2.0)
                .overlay(),
        );
        scene.debug.line(
            mid,
            pole,
            LineOpts::color([0.85, 0.35, 0.95, 0.85])
                .width(2.0)
                .overlay(),
        );
    }
}
