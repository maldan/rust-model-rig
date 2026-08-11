//! Soft bone chains: world-gravity Verlet + spring to animated pose + support plane.

use glam::{Quat, Vec3};
use mega_render::{LineOpts, Scene, Transform};
use std::collections::HashMap;

use crate::ik::quat_from_matrix;
use crate::rig::{AppMode, BoneId, RigDocument, SoftChain};

const CONSTRAINT_ITERS: u32 = 6;
const SUBSTEPS: u32 = 4;

/// Run all enabled soft chains (Pose only). Call after IK.
pub fn evaluate_soft_chains(scene: &mut Scene, rig: &mut RigDocument, dt: f32) {
    if rig.mode != AppMode::Pose {
        return;
    }
    let dt = dt.clamp(1.0 / 240.0, 1.0 / 20.0);
    let mut chains = std::mem::take(&mut rig.soft_chains);
    for chain in &mut chains {
        if chain.enabled {
            restore_soft_bind(scene, &rig.bind_locals, chain);
            solve_chain(scene, chain, dt);
        }
    }
    rig.soft_chains = chains;
}

fn restore_soft_bind(
    scene: &mut Scene,
    binds: &HashMap<(u32, u32), Transform>,
    chain: &SoftChain,
) {
    for &id in &chain.bones {
        let Some(bind) = binds.get(&id.node.key()).copied() else {
            continue;
        };
        if let Some(n) = scene.nodes.get_mut(id.node) {
            n.local.rotation = bind.rotation;
        }
    }
}

fn solve_chain(scene: &mut Scene, chain: &mut SoftChain, dt: f32) {
    let n_bones = chain.bones.len();
    if n_bones < 2 {
        return;
    }
    // Particles: one per bone + virtual tip past the last bone.
    let n = n_bones + 1;

    let mut rest = Vec::with_capacity(n);
    for &b in &chain.bones {
        rest.push(scene.world_matrix(b.node).transform_point3(Vec3::ZERO));
    }
    {
        let tip = chain.bones[n_bones - 1];
        let tip_m = scene.world_matrix(tip.node);
        let axis = tip_m.transform_vector3(Vec3::Y).normalize_or_zero();
        let tip_dir = if axis.length_squared() > 1e-8 {
            axis
        } else {
            let mut d = (rest[n_bones - 1] - rest[n_bones.saturating_sub(2)]).normalize_or_zero();
            if d.length_squared() < 1e-8 {
                d = Vec3::Y;
            }
            d
        };
        rest.push(rest[n_bones - 1] + tip_dir * chain.tip_length.max(1e-4));
    }

    if !chain.initialized || chain.curr_pos.len() != n {
        chain.prev_pos = rest.clone();
        chain.curr_pos = rest.clone();
        chain.initialized = true;
    }

    let gravity = Vec3::new(0.0, -chain.gravity, 0.0);
    let h = dt / SUBSTEPS as f32;
    let h2 = h * h;
    // Frame-rate independent: keep ≈ e^(-damping * h)
    let vel_keep = (-chain.damping.max(0.0) * h).exp();
    // Critical-ish spring blend toward rest each substep.
    let stiff_blend = (1.0 - (-chain.stiffness.max(0.0) * h).exp()).clamp(0.0, 1.0);

    let anchor_rot = quat_from_matrix(scene.world_matrix(chain.anchor.node));
    let mut support_n = (anchor_rot * chain.support_normal_local).normalize_or_zero();
    if support_n.length_squared() < 1e-8 {
        support_n = Vec3::Z;
    }
    let max_angle = chain.max_angle.max(0.05);

    // Segment lengths between particles (bone joints + virtual tip).
    let mut seg_len = Vec::with_capacity(n);
    seg_len.push(0.0);
    for i in 1..n_bones {
        seg_len.push(chain.lengths[i].max(1e-4));
    }
    seg_len.push(chain.tip_length.max(1e-4));

    for _ in 0..SUBSTEPS {
        // Pin root to animated pose.
        chain.curr_pos[0] = rest[0];
        chain.prev_pos[0] = rest[0];

        for i in 1..n {
            let vel = chain.curr_pos[i] - chain.prev_pos[i];
            let mut next = chain.curr_pos[i] + vel * vel_keep + gravity * h2;
            next += (rest[i] - next) * stiff_blend;
            chain.prev_pos[i] = chain.curr_pos[i];
            chain.curr_pos[i] = next;
        }

        let plane_p = chain.curr_pos[0];
        for _ in 0..CONSTRAINT_ITERS {
            chain.curr_pos[0] = rest[0];
            for i in 1..n {
                let parent = chain.curr_pos[i - 1];
                let len = seg_len[i];

                let mut d = chain.curr_pos[i] - parent;
                if d.length_squared() < 1e-12 {
                    d = (rest[i] - parent).normalize_or_zero() * len;
                    if d.length_squared() < 1e-12 {
                        d = support_n * len;
                    }
                } else {
                    d = d.normalize() * len;
                }
                let mut p = parent + d;

                // Stay in front of support plane (chest).
                let side = (p - plane_p).dot(support_n);
                if side < 0.0 {
                    p += support_n * (-side);
                    let d2 = p - parent;
                    if d2.length_squared() > 1e-12 {
                        p = parent + d2.normalize() * len;
                    }
                }

                // Angle cone vs animated rest.
                let rest_dir = (rest[i] - parent).normalize_or_zero();
                let mut dir = (p - parent).normalize_or_zero();
                if rest_dir.length_squared() > 1e-8 && dir.length_squared() > 1e-8 {
                    let dot = rest_dir.dot(dir).clamp(-1.0, 1.0);
                    let ang = dot.acos();
                    if ang > max_angle {
                        let axis = rest_dir.cross(dir);
                        if axis.length_squared() > 1e-10 {
                            let q = Quat::from_axis_angle(axis.normalize(), max_angle);
                            dir = q * rest_dir;
                            p = parent + dir * len;
                        } else {
                            p = parent + rest_dir * len;
                        }
                    }
                }

                chain.curr_pos[i] = p;
            }
        }
    }

    // Aim each bone at the next particle (last bone → virtual tip).
    for i in 0..n_bones {
        let target = chain.curr_pos[i + 1];
        if i + 1 < n_bones {
            swing_bone_to(scene, chain.bones[i], chain.bones[i + 1], target);
        } else {
            swing_bone_aim(scene, chain.bones[i], target);
        }
    }
}

fn swing_bone_to(scene: &mut Scene, bone: BoneId, child: BoneId, desired_child: Vec3) {
    let origin = scene.world_matrix(bone.node).transform_point3(Vec3::ZERO);
    let cur_child = scene.world_matrix(child.node).transform_point3(Vec3::ZERO);
    let from = (cur_child - origin).normalize_or_zero();
    let to = (desired_child - origin).normalize_or_zero();
    apply_swing(scene, bone, from, to);
}

/// Rotate bone so its local +Y aims at a world point (for tip / virtual end).
fn swing_bone_aim(scene: &mut Scene, bone: BoneId, desired_tip: Vec3) {
    let m = scene.world_matrix(bone.node);
    let origin = m.transform_point3(Vec3::ZERO);
    let from = m.transform_vector3(Vec3::Y).normalize_or_zero();
    let to = (desired_tip - origin).normalize_or_zero();
    apply_swing(scene, bone, from, to);
}

fn apply_swing(scene: &mut Scene, bone: BoneId, from: Vec3, to: Vec3) {
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

/// Debug overlay for soft chains.
pub fn draw_soft_helpers(scene: &mut Scene, rig: &RigDocument) {
    if rig.mode != AppMode::Pose {
        return;
    }
    for chain in &rig.soft_chains {
        if !chain.enabled || chain.curr_pos.len() < 2 {
            continue;
        }
        let col = [0.35, 0.85, 1.0, 0.85];
        for w in chain.curr_pos.windows(2) {
            scene.debug.line(
                w[0],
                w[1],
                LineOpts::color(col).width(2.5).overlay(),
            );
        }
        let anchor_rot = quat_from_matrix(scene.world_matrix(chain.anchor.node));
        let n = (anchor_rot * chain.support_normal_local).normalize_or_zero();
        if n.length_squared() > 1e-8 && !chain.curr_pos.is_empty() {
            let p = chain.curr_pos[0];
            scene.debug.line(
                p,
                p + n * 0.08,
                LineOpts::color([0.2, 1.0, 0.45, 0.7])
                    .width(2.0)
                    .overlay(),
            );
        }
    }
}
