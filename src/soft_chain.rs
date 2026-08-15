//! Adapter between the rig's bones and the generic particle-chain simulation
//! in `mega-physics` (`mega_physics::chain`): world-gravity Verlet + spring to
//! animated pose + support plane + capsule↔capsule soft collision. The
//! physics lib knows nothing about bones/scenes — this file is entirely
//! about translating bone poses into [`mega_physics::chain::ChainFrame`]
//! inputs and turning simulated particle positions back into bone rotations.
//!
//! Soft-grab (interactive drag) is app-side tooling on top of
//! [`mega_physics::chain::PullTarget`].

use glam::{Quat, Vec2, Vec3};
use mega_physics::chain::{self, Chain, ChainCapsule, ChainFrame, PreparedChain, PullTarget};
use mega_physics::Isometry;
use mega_render::{quat_from_matrix, LineOpts, Scene, Transform};
use mega_ui::Rect;
use std::collections::HashMap;
use std::f32::consts::TAU;

use crate::pick::{self, Ray};
use crate::rig::{AppMode, BoneCollider, BoneId, RigDocument, SoftChain};

const CONSTRAINT_ITERS: u32 = 6;
const SUBSTEPS: u32 = 4;
const COLLISION_PASSES: u32 = 3;
/// How hard Soft Grab pulls toward the cursor per constraint pass (0–1).
const GRAB_BLEND: f32 = 0.55;
/// Falloff along root→grab (`< 1` = mid-chain feels more of the pull).
const GRAB_CHAIN_POWER: f32 = 0.5;

/// Active Soft Grab: soft-pin one particle to a camera-plane cursor target.
#[derive(Clone, Debug)]
pub struct SoftGrabDrag {
    pub chain_id: u32,
    /// Index into the chain's simulated particles (≥ 1; root is pinned to body).
    pub particle: usize,
    /// Bone to select while grabbing (tip particle → last bone).
    pub bone: BoneId,
    pub target: Vec3,
    pub plane_point: Vec3,
    pub plane_normal: Vec3,
}

/// Extra per-chain data needed to place bone colliders that ride along a
/// soft chain's particles (see [`build_live_capsules`]).
struct ChainAxis {
    tip_local_axis: Vec3,
}

/// Run all enabled soft chains (Pose only). Call after IK.
pub fn evaluate_soft_chains(
    scene: &mut Scene,
    rig: &mut RigDocument,
    dt: f32,
    grab: Option<&SoftGrabDrag>,
) {
    if rig.mode != AppMode::Pose {
        return;
    }
    let dt = dt.clamp(1.0 / 240.0, 1.0 / 20.0);
    let colliders = rig.colliders.clone();
    let mut chains = std::mem::take(&mut rig.soft_chains);

    for chain in &mut chains {
        if chain.enabled {
            restore_soft_bind(scene, &rig.bind_locals, chain);
        }
    }

    let mut scratches: Vec<Option<PreparedChain>> = Vec::with_capacity(chains.len());
    let mut axes: Vec<Option<ChainAxis>> = Vec::with_capacity(chains.len());
    for c in &mut chains {
        if c.enabled {
            let (s, a) = prepare_chain(scene, c, dt);
            scratches.push(s);
            axes.push(a);
        } else {
            scratches.push(None);
            axes.push(None);
        }
    }

    // While Soft Grab holds a chain: no rest-spring fight (avoids stored snap energy).
    // (Handled per-substep below by passing `pull` only for the grabbed chain.)

    for _ in 0..SUBSTEPS {
        for i in 0..chains.len() {
            if let Some(scratch) = scratches[i].as_ref() {
                chains[i].sim.integrate(scratch);
            }
        }
        for i in 0..chains.len() {
            if let Some(scratch) = scratches[i].as_ref() {
                let pull = grab
                    .filter(|g| g.chain_id == chains[i].id)
                    .map(|g| PullTarget {
                        particle: g.particle,
                        target: g.target,
                        blend: GRAB_BLEND,
                        chain_power: GRAB_CHAIN_POWER,
                    });
                chains[i].sim.constrain(scratch, CONSTRAINT_ITERS, pull);
            }
        }
        // Capsule↔capsule from *current* particle poses (lockstep) — stable, no particle hits.
        resolve_capsule_capsule(scene, &mut chains, &scratches, &colliders);
        // Held grab is kinematic: kill Verlet velocity so release can't shoot.
        if let Some(g) = grab {
            if let Some(chain) = chains.iter_mut().find(|c| c.id == g.chain_id) {
                chain.sim.zero_velocity();
            }
        }
    }

    for i in 0..chains.len() {
        if let (Some(scratch), Some(axis)) = (scratches[i].as_ref(), axes[i].as_ref()) {
            write_chain_bones(scene, &chains[i], scratch, axis);
        }
    }

    rig.soft_chains = chains;
}

/// Pick nearest soft particle under the cursor and start a Soft Grab.
pub fn begin_soft_grab(
    scene: &Scene,
    rig: &RigDocument,
    viewport: Rect,
    cursor: Vec2,
) -> Option<SoftGrabDrag> {
    let ray = pick::ray_from_viewport(scene, viewport, cursor)?;
    let (chain_id, particle, bone, hit) = pick_soft_particle(scene, rig, &ray)?;

    let plane_normal = (scene.camera.target - scene.camera.eye).normalize_or_zero();
    let plane_normal = if plane_normal.length_squared() < 1e-8 {
        -ray.dir
    } else {
        plane_normal
    };

    Some(SoftGrabDrag {
        chain_id,
        particle,
        bone,
        target: hit,
        plane_point: hit,
        plane_normal,
    })
}

/// Update grab target from cursor (camera-facing plane).
pub fn update_soft_grab(scene: &Scene, drag: &mut SoftGrabDrag, viewport: Rect, cursor: Vec2) {
    let Some(ray) = pick::ray_from_viewport(scene, viewport, cursor) else {
        return;
    };
    if let Some(hit) = ray_plane(&ray, drag.plane_point, drag.plane_normal) {
        drag.target = hit;
    }
}

fn pick_soft_particle(
    scene: &Scene,
    rig: &RigDocument,
    ray: &Ray,
) -> Option<(u32, usize, BoneId, Vec3)> {
    let mut best: Option<(f32, u32, usize, BoneId, Vec3)> = None;
    for chain in &rig.soft_chains {
        if !chain.enabled || chain.bones.len() < 2 {
            continue;
        }
        let n_bones = chain.bones.len();
        let n = n_bones + 1;
        let use_sim = chain.sim.is_initialized() && chain.sim.positions().len() == n;
        let scale: f32 = chain.lengths.iter().copied().sum::<f32>() + chain.tip_length;
        let radius = (scale * 0.12).clamp(0.02, 0.12);

        for i in 1..n {
            let p = if use_sim {
                chain.sim.positions()[i]
            } else if i < n_bones {
                scene
                    .world_matrix(chain.bones[i].node)
                    .transform_point3(Vec3::ZERO)
            } else {
                let tip = chain.bones[n_bones - 1];
                let m = scene.world_matrix(tip.node);
                let origin = m.transform_point3(Vec3::ZERO);
                let axis = m.transform_vector3(Vec3::Y).normalize_or_zero();
                origin + axis * chain.tip_length.max(1e-4)
            };
            let Some(t) = ray_sphere(ray, p, radius) else {
                continue;
            };
            let bone = if i < n_bones {
                chain.bones[i]
            } else {
                chain.bones[n_bones - 1]
            };
            if best.is_none_or(|(bt, ..)| t < bt) {
                best = Some((t, chain.id, i, bone, p));
            }
        }
    }
    best.map(|(_, cid, i, bone, p)| (cid, i, bone, p))
}

fn ray_sphere(ray: &Ray, center: Vec3, radius: f32) -> Option<f32> {
    let oc = ray.origin - center;
    let a = ray.dir.dot(ray.dir);
    let b = 2.0 * oc.dot(ray.dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let t0 = (-b - s) / (2.0 * a);
    let t1 = (-b + s) / (2.0 * a);
    if t0 >= 0.0 {
        Some(t0)
    } else if t1 >= 0.0 {
        Some(t1)
    } else {
        None
    }
}

fn ray_plane(ray: &Ray, point: Vec3, normal: Vec3) -> Option<Vec3> {
    let n = normal.normalize_or_zero();
    if n.length_squared() < 1e-8 {
        return None;
    }
    let denom = ray.dir.dot(n);
    if denom.abs() < 1e-8 {
        return None;
    }
    let t = (point - ray.origin).dot(n) / denom;
    if t < 0.0 {
        return None;
    }
    Some(ray.origin + ray.dir * t)
}

/// Kill grab-induced Verlet velocity so release doesn't shoot the chain.
pub fn release_soft_grab(rig: &mut RigDocument, grab: &SoftGrabDrag) {
    let Some(chain) = rig.soft_chains.iter_mut().find(|c| c.id == grab.chain_id) else {
        return;
    };
    chain.sim.zero_velocity();
    // Soften spring snap-back for a short window after release.
    chain.sim.begin_relax();
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

fn isometry_from_mat4(m: glam::Mat4) -> Isometry {
    let (_, rotation, translation) = m.to_scale_rotation_translation();
    Isometry::new(translation, rotation)
}

fn prepare_chain(
    scene: &mut Scene,
    chain: &mut SoftChain,
    dt: f32,
) -> (Option<PreparedChain>, Option<ChainAxis>) {
    let n_bones = chain.bones.len();
    if n_bones < 2 {
        return (None, None);
    }
    let n = n_bones + 1;

    let mut rest = Vec::with_capacity(n);
    for &b in &chain.bones {
        rest.push(scene.world_matrix(b.node).transform_point3(Vec3::ZERO));
    }
    {
        let mut tip_dir = (rest[n_bones - 1] - rest[n_bones - 2]).normalize_or_zero();
        if tip_dir.length_squared() < 1e-8 {
            let tip = chain.bones[n_bones - 1];
            tip_dir = scene
                .world_matrix(tip.node)
                .transform_vector3(Vec3::Y)
                .normalize_or_zero();
        }
        if tip_dir.length_squared() < 1e-8 {
            tip_dir = Vec3::Y;
        }
        rest.push(rest[n_bones - 1] + tip_dir * chain.tip_length.max(1e-4));
    }

    let mut seg_len = Vec::with_capacity(n);
    seg_len.push(0.0);
    for i in 1..n_bones {
        seg_len.push(chain.lengths[i].max(1e-4));
    }
    seg_len.push(chain.tip_length.max(1e-4));

    let root = isometry_from_mat4(scene.world_matrix(chain.bones[0].node));

    let anchor_rot = quat_from_matrix(scene.world_matrix(chain.anchor.node));
    let mut support_n = (anchor_rot * chain.support_normal_local).normalize_or_zero();
    if support_n.length_squared() < 1e-8 {
        support_n = Vec3::Z;
    }

    let tip_local_axis = {
        let tip = chain.bones[n_bones - 1];
        let tip_m = scene.world_matrix(tip.node);
        let (_, tip_rot, _) = tip_m.to_scale_rotation_translation();
        let inv = tip_rot.normalize().inverse();
        let rest_tip_dir = (rest[n - 1] - rest[n - 2]).normalize_or_zero();
        let axis = inv * rest_tip_dir;
        if axis.length_squared() > 1e-8 {
            axis.normalize()
        } else {
            Vec3::Y
        }
    };

    let support_point = rest[0];
    let frame = ChainFrame {
        rest,
        seg_len,
        root,
        gravity: chain.gravity,
        stiffness: chain.stiffness,
        damping: chain.damping,
        inertia: chain.inertia,
        max_angle: chain.max_angle,
        support_point,
        support_normal: support_n,
    };
    let scratch = chain.sim.prepare(&frame, dt, SUBSTEPS);

    (Some(scratch), Some(ChainAxis { tip_local_axis }))
}

fn write_chain_bones(scene: &mut Scene, chain: &SoftChain, _scratch: &PreparedChain, axis: &ChainAxis) {
    let n_bones = chain.bones.len();
    let pos = chain.sim.positions();
    let n = pos.len();
    for i in 0..n_bones {
        if i + 1 < n_bones {
            swing_bone_to(scene, chain.bones[i], chain.bones[i + 1], pos[i + 1]);
        } else {
            let tip = chain.bones[i];
            let m = scene.world_matrix(tip.node);
            let origin = m.transform_point3(Vec3::ZERO);
            let (_, tip_rot, _) = m.to_scale_rotation_translation();
            let from = (tip_rot.normalize() * axis.tip_local_axis).normalize_or_zero();
            let to = (pos[n - 1] - origin).normalize_or_zero();
            apply_swing(scene, tip, from, to);
        }
    }
}

/// Build world capsules from soft particles (live) or bone matrices (static bones).
fn build_live_capsules(
    scene: &Scene,
    chains: &[SoftChain],
    scratches: &[Option<PreparedChain>],
    colliders: &[BoneCollider],
) -> Vec<ChainCapsule> {
    let mut out = Vec::new();
    for col in colliders {
        if !col.enabled {
            continue;
        }
        let key = col.bone.node.key();
        let mut placed = false;
        for (ci, chain) in chains.iter().enumerate() {
            if scratches[ci].is_none() || !chain.enabled {
                continue;
            }
            if let Some(bi) = chain.bones.iter().position(|b| b.node.key() == key) {
                if let Some(cap) = capsule_from_soft_bone(chain, bi, col, ci) {
                    out.push(cap);
                    placed = true;
                    break;
                }
            }
        }
        if !placed {
            let m = scene.world_matrix(col.bone.node);
            out.push(ChainCapsule {
                a: m.transform_point3(col.a_local),
                b: m.transform_point3(col.b_local),
                radius: col.radius.max(1e-4),
                softness: col.softness.clamp(0.0, 1.0),
                chain: None,
            });
        }
    }
    out
}

fn capsule_from_soft_bone(
    chain: &SoftChain,
    bone_idx: usize,
    col: &BoneCollider,
    chain_idx: usize,
) -> Option<ChainCapsule> {
    let positions = chain.sim.positions();
    if bone_idx >= positions.len() {
        return None;
    }
    let origin = positions[bone_idx];
    let next = positions
        .get(bone_idx + 1)
        .copied()
        .unwrap_or(origin + Vec3::Y * 0.01);
    let mut world_axis = (next - origin).normalize_or_zero();
    if world_axis.length_squared() < 1e-8 {
        world_axis = Vec3::Y;
    }
    let mut local_axis = (col.b_local - col.a_local).normalize_or_zero();
    if local_axis.length_squared() < 1e-8 {
        local_axis = Vec3::Y;
    }
    let rot = Quat::from_rotation_arc(local_axis, world_axis);
    Some(ChainCapsule {
        a: origin + rot * col.a_local,
        b: origin + rot * col.b_local,
        radius: col.radius.max(1e-4),
        softness: col.softness.clamp(0.0, 1.0),
        chain: Some(chain_idx),
    })
}

fn resolve_capsule_capsule(
    scene: &Scene,
    chains: &mut [SoftChain],
    scratches: &[Option<PreparedChain>],
    colliders: &[BoneCollider],
) {
    let caps = build_live_capsules(scene, chains, scratches, colliders);
    if caps.len() < 2 {
        return;
    }
    // `chain::resolve_capsule_collisions` operates on `Chain` + scratch slices
    // aligned by index; unpack/repack the `sim` field to satisfy borrowing.
    let mut sims: Vec<Chain> = chains.iter_mut().map(|c| std::mem::take(&mut c.sim)).collect();
    chain::resolve_capsule_collisions(&mut sims, scratches, &caps, COLLISION_PASSES);
    for (c, sim) in chains.iter_mut().zip(sims.into_iter()) {
        c.sim = sim;
    }
}

fn swing_bone_to(scene: &mut Scene, bone: BoneId, child: BoneId, desired_child: Vec3) {
    let origin = scene.world_matrix(bone.node).transform_point3(Vec3::ZERO);
    let cur_child = scene.world_matrix(child.node).transform_point3(Vec3::ZERO);
    let from = (cur_child - origin).normalize_or_zero();
    let to = (desired_child - origin).normalize_or_zero();
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
    let q = if dot < -0.999999 {
        let axis = from.any_orthonormal_vector();
        Quat::from_axis_angle(axis, std::f32::consts::PI)
    } else {
        Quat::from_rotation_arc(from, to)
    };
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
        let pos = chain.sim.positions();
        if !chain.enabled || pos.len() < 2 {
            continue;
        }
        let col = [0.35, 0.85, 1.0, 0.85];
        for w in pos.windows(2) {
            scene.debug.line(
                w[0],
                w[1],
                LineOpts::color(col).width(2.5).overlay(),
            );
        }
        let anchor_rot = quat_from_matrix(scene.world_matrix(chain.anchor.node));
        let n = (anchor_rot * chain.support_normal_local).normalize_or_zero();
        if n.length_squared() > 1e-8 && !pos.is_empty() {
            let p = pos[0];
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

/// Soft Grab feedback: particle → cursor target.
pub fn draw_soft_grab(scene: &mut Scene, rig: &RigDocument, grab: &SoftGrabDrag) {
    let Some(chain) = rig.soft_chains.iter().find(|c| c.id == grab.chain_id) else {
        return;
    };
    let pos = chain.sim.positions();
    if grab.particle >= pos.len() {
        return;
    }
    let p = pos[grab.particle];
    let col = [1.0, 0.85, 0.25, 0.95];
    scene.debug.sphere(p, 0.018, col, false);
    scene.debug.line(
        p,
        grab.target,
        LineOpts::color(col).width(2.0).overlay(),
    );
    scene.debug.sphere(grab.target, 0.012, [1.0, 0.55, 0.15, 0.9], false);
}

/// Debug wire capsules for bone colliders (Edit + Pose).
pub fn draw_bone_colliders(scene: &mut Scene, rig: &RigDocument) {
    if !rig.show_colliders {
        return;
    }
    for c in &rig.colliders {
        if !c.enabled {
            continue;
        }
        let selected = rig.selection == Some(c.bone);
        let col = if selected {
            [1.0, 0.55, 0.15, 0.95]
        } else {
            [0.95, 0.45, 0.55, 0.75]
        };
        let m = scene.world_matrix(c.bone.node);
        let a = m.transform_point3(c.a_local);
        let b = m.transform_point3(c.b_local);
        draw_capsule_wire(scene, a, b, c.radius.max(1e-4), col);
    }
}

fn draw_capsule_wire(scene: &mut Scene, a: Vec3, b: Vec3, radius: f32, color: [f32; 4]) {
    let opts = LineOpts::color(color).width(1.6).overlay();
    scene.debug.sphere(a, radius, color, false);
    scene.debug.sphere(b, radius, color, false);

    let ab = b - a;
    let len = ab.length();
    let axis = if len > 1e-8 { ab / len } else { Vec3::Y };
    let mut side = axis.cross(Vec3::Y);
    if side.length_squared() < 1e-8 {
        side = axis.cross(Vec3::X);
    }
    let side = side.normalize_or_zero();
    let up = axis.cross(side).normalize_or_zero();
    if side.length_squared() < 1e-8 || up.length_squared() < 1e-8 {
        return;
    }

    for k in 0..4 {
        let ang = TAU * k as f32 / 4.0;
        let off = (side * ang.cos() + up * ang.sin()) * radius;
        scene.debug.line(a + off, b + off, opts);
    }

    let rings = 3u32;
    let segs = 16u32;
    for r in 0..=rings {
        let t = r as f32 / rings as f32;
        let center = a + ab * t;
        for i in 0..segs {
            let a0 = TAU * i as f32 / segs as f32;
            let a1 = TAU * (i + 1) as f32 / segs as f32;
            let p0 = center + (side * a0.cos() + up * a0.sin()) * radius;
            let p1 = center + (side * a1.cos() + up * a1.sin()) * radius;
            scene.debug.line(p0, p1, opts);
        }
    }
}
