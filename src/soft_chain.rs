//! Soft bone chains: world-gravity Verlet + spring to animated pose + support plane.
//! Capsule↔capsule soft collision (no particle↔capsule).

use glam::{Quat, Vec3};
use mega_render::{LineOpts, Scene, Transform};
use std::collections::HashMap;
use std::f32::consts::TAU;

use crate::ik::quat_from_matrix;
use crate::rig::{AppMode, BoneCollider, BoneId, RigDocument, SoftChain};

const CONSTRAINT_ITERS: u32 = 6;
const SUBSTEPS: u32 = 4;

struct WorldCapsule {
    a: Vec3,
    b: Vec3,
    radius: f32,
    softness: f32,
    /// Index into the soft-chains slice being solved, if collider bone is soft.
    chain_idx: Option<usize>,
}

struct ChainScratch {
    rest: Vec<Vec3>,
    rest_dir: Vec<Vec3>,
    seg_len: Vec<f32>,
    support_n: Vec3,
    max_angle: f32,
    accel: Vec3,
    tip_local_axis: Vec3,
    h: f32,
    h2: f32,
    vel_keep: f32,
    stiff: f32,
}

/// Run all enabled soft chains (Pose only). Call after IK.
pub fn evaluate_soft_chains(scene: &mut Scene, rig: &mut RigDocument, dt: f32) {
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

    let mut scratches: Vec<Option<ChainScratch>> = (0..chains.len())
        .map(|i| {
            if chains[i].enabled {
                prepare_chain(scene, &mut chains[i], dt)
            } else {
                None
            }
        })
        .collect();

    for _ in 0..SUBSTEPS {
        for i in 0..chains.len() {
            if let Some(scratch) = scratches[i].as_ref() {
                integrate_chain(&mut chains[i], scratch);
            }
        }
        for i in 0..chains.len() {
            if let Some(scratch) = scratches[i].as_ref() {
                constrain_chain(&mut chains[i], scratch);
            }
        }
        // Capsule↔capsule from *current* particle poses (lockstep) — stable, no particle hits.
        resolve_capsule_capsule(scene, &mut chains, &scratches, &colliders);
    }

    for i in 0..chains.len() {
        if let Some(scratch) = scratches[i].as_ref() {
            write_chain_bones(scene, &chains[i], scratch);
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

fn prepare_chain(scene: &mut Scene, chain: &mut SoftChain, dt: f32) -> Option<ChainScratch> {
    let n_bones = chain.bones.len();
    if n_bones < 2 {
        return None;
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

    let mut rest_dir = Vec::with_capacity(n);
    rest_dir.push(Vec3::ZERO);
    for i in 1..n {
        rest_dir.push((rest[i] - rest[i - 1]).normalize_or_zero());
    }

    let root_world = scene.world_matrix(chain.bones[0].node);
    let (_, _, curr_t) = root_world.to_scale_rotation_translation();
    let mut fictitious = Vec3::ZERO;

    if !chain.initialized || chain.curr_pos.len() != n {
        chain.prev_pos = rest.clone();
        chain.curr_pos = rest.clone();
        chain.prev_root_world = root_world;
        chain.prev_root_vel = Vec3::ZERO;
        chain.initialized = true;
    } else {
        let (_, _, prev_t) = chain.prev_root_world.to_scale_rotation_translation();
        let trans = curr_t - prev_t;
        let dt_safe = dt.max(1e-4);
        let root_vel = trans / dt_safe;
        let mut root_accel = (root_vel - chain.prev_root_vel) / dt_safe;
        chain.prev_root_vel = root_vel;

        let scale: f32 = chain.lengths.iter().copied().sum::<f32>() + chain.tip_length;
        let inertia = chain.inertia.clamp(0.0, 20.0);
        let teleport = trans.length() > (scale * 2.5).max(0.2);

        if teleport {
            chain.prev_pos = rest.clone();
            chain.curr_pos = rest.clone();
            chain.prev_root_vel = Vec3::ZERO;
            root_accel = Vec3::ZERO;
        } else {
            let delta = root_world * chain.prev_root_world.inverse();
            for i in 1..n {
                chain.curr_pos[i] = delta.transform_point3(chain.curr_pos[i]);
                chain.prev_pos[i] = delta.transform_point3(chain.prev_pos[i]);
            }

            let t_len = trans.length();
            if inertia > 1e-6 && t_len > 1e-8 {
                let max_lag = (scale * 0.4 * inertia.max(1.0)).max(0.012);
                let mut lag = trans * inertia;
                if lag.length() > max_lag {
                    lag = lag.normalize() * max_lag;
                }
                for i in 1..n {
                    chain.curr_pos[i] -= lag;
                    chain.prev_pos[i] -= lag;
                }
            }
        }

        let max_a = ((scale * 80.0).max(15.0)).min(100.0);
        if root_accel.length_squared() > max_a * max_a {
            root_accel = root_accel.normalize() * max_a;
        }
        fictitious = -root_accel * inertia;
        chain.prev_root_world = root_world;
    }

    chain.curr_pos[0] = rest[0];
    chain.prev_pos[0] = rest[0];

    let gravity = Vec3::new(0.0, -chain.gravity, 0.0);
    let h = dt / SUBSTEPS as f32;
    let h2 = h * h;
    let vel_keep = (-chain.damping.max(0.0) * h).exp();
    let stiff = chain.stiffness.max(0.0);

    let anchor_rot = quat_from_matrix(scene.world_matrix(chain.anchor.node));
    let mut support_n = (anchor_rot * chain.support_normal_local).normalize_or_zero();
    if support_n.length_squared() < 1e-8 {
        support_n = Vec3::Z;
    }

    let mut seg_len = Vec::with_capacity(n);
    seg_len.push(0.0);
    for i in 1..n_bones {
        seg_len.push(chain.lengths[i].max(1e-4));
    }
    seg_len.push(chain.tip_length.max(1e-4));

    let tip_local_axis = {
        let tip = chain.bones[n_bones - 1];
        let tip_m = scene.world_matrix(tip.node);
        let (_, tip_rot, _) = tip_m.to_scale_rotation_translation();
        let inv = tip_rot.normalize().inverse();
        let axis = inv * rest_dir[n_bones];
        if axis.length_squared() > 1e-8 {
            axis.normalize()
        } else {
            Vec3::Y
        }
    };

    Some(ChainScratch {
        rest,
        rest_dir,
        seg_len,
        support_n,
        max_angle: chain.max_angle.max(0.05),
        accel: gravity + fictitious,
        tip_local_axis,
        h,
        h2,
        vel_keep,
        stiff,
    })
}

fn integrate_chain(chain: &mut SoftChain, scratch: &ChainScratch) {
    let n = chain.curr_pos.len();
    chain.curr_pos[0] = scratch.rest[0];
    chain.prev_pos[0] = scratch.rest[0];

    for i in 1..n {
        let parent = chain.curr_pos[i - 1];
        let rest_target = if scratch.rest_dir[i].length_squared() > 1e-8 {
            parent + scratch.rest_dir[i] * scratch.seg_len[i]
        } else {
            scratch.rest[i]
        };
        let vel = chain.curr_pos[i] - chain.prev_pos[i];
        let spring = (rest_target - chain.curr_pos[i]) * scratch.stiff;
        let next = chain.curr_pos[i] + vel * scratch.vel_keep + (scratch.accel + spring) * scratch.h2;
        chain.prev_pos[i] = chain.curr_pos[i];
        chain.curr_pos[i] = next;
    }
}

fn constrain_chain(chain: &mut SoftChain, scratch: &ChainScratch) {
    let n = chain.curr_pos.len();
    let plane_p = scratch.rest[0];
    let max_angle = scratch.max_angle;
    let support_n = scratch.support_n;

    for _ in 0..CONSTRAINT_ITERS {
        chain.curr_pos[0] = scratch.rest[0];
        for i in 1..n {
            let parent = chain.curr_pos[i - 1];
            let len = scratch.seg_len[i];
            let rd = scratch.rest_dir[i];
            let old = chain.curr_pos[i];

            let mut p = project_length(old, parent, len, rd, support_n);

            let side = (p - plane_p).dot(support_n);
            if side < 0.0 {
                p += support_n * (-side);
                p = project_length(p, parent, len, rd, support_n);
            }

            if rd.length_squared() > 1e-8 {
                let mut dir = (p - parent).normalize_or_zero();
                if dir.length_squared() > 1e-8 {
                    let dot = rd.dot(dir).clamp(-1.0, 1.0);
                    let ang = dot.acos();
                    if ang > max_angle {
                        let axis = rd.cross(dir);
                        if axis.length_squared() > 1e-10 {
                            let q = Quat::from_axis_angle(axis.normalize(), max_angle);
                            dir = q * rd;
                            p = parent + dir * len;
                        } else {
                            p = parent + rd * len;
                        }
                        p = clamp_to_halfspace(p, plane_p, support_n);
                        p = project_length(p, parent, len, rd, support_n);
                    }
                }
            }

            chain.curr_pos[i] = p;
            let corr = p - old;
            let clen = corr.length();
            let shock = (len * 0.2).max(1e-4);
            if clen > shock {
                chain.prev_pos[i] += corr * ((clen - shock) / clen);
            }

            if (p - plane_p).dot(support_n) < 1e-4 {
                let vel = chain.curr_pos[i] - chain.prev_pos[i];
                let vn = vel.dot(support_n);
                if vn < 0.0 {
                    chain.prev_pos[i] = p - (vel - support_n * vn);
                }
            }
        }
    }
}

fn write_chain_bones(scene: &mut Scene, chain: &SoftChain, scratch: &ChainScratch) {
    let n_bones = chain.bones.len();
    let n = chain.curr_pos.len();
    for i in 0..n_bones {
        if i + 1 < n_bones {
            swing_bone_to(scene, chain.bones[i], chain.bones[i + 1], chain.curr_pos[i + 1]);
        } else {
            let tip = chain.bones[i];
            let m = scene.world_matrix(tip.node);
            let origin = m.transform_point3(Vec3::ZERO);
            let (_, tip_rot, _) = m.to_scale_rotation_translation();
            let from = (tip_rot.normalize() * scratch.tip_local_axis).normalize_or_zero();
            let to = (chain.curr_pos[n - 1] - origin).normalize_or_zero();
            apply_swing(scene, tip, from, to);
        }
    }
}

/// Build world capsules from soft particles (live) or bone matrices (static bones).
fn build_live_capsules(
    scene: &Scene,
    chains: &[SoftChain],
    scratches: &[Option<ChainScratch>],
    colliders: &[BoneCollider],
) -> Vec<WorldCapsule> {
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
            out.push(WorldCapsule {
                a: m.transform_point3(col.a_local),
                b: m.transform_point3(col.b_local),
                radius: col.radius.max(1e-4),
                softness: col.softness.clamp(0.0, 1.0),
                chain_idx: None,
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
) -> Option<WorldCapsule> {
    if bone_idx >= chain.curr_pos.len() {
        return None;
    }
    let origin = chain.curr_pos[bone_idx];
    let next = chain
        .curr_pos
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
    Some(WorldCapsule {
        a: origin + rot * col.a_local,
        b: origin + rot * col.b_local,
        radius: col.radius.max(1e-4),
        softness: col.softness.clamp(0.0, 1.0),
        chain_idx: Some(chain_idx),
    })
}

fn resolve_capsule_capsule(
    scene: &Scene,
    chains: &mut [SoftChain],
    scratches: &[Option<ChainScratch>],
    colliders: &[BoneCollider],
) {
    // A couple of soft passes so large radii actually settle within the substep.
    for _ in 0..3 {
        resolve_capsule_capsule_once(scene, chains, scratches, colliders);
    }
}

fn resolve_capsule_capsule_once(
    scene: &Scene,
    chains: &mut [SoftChain],
    scratches: &[Option<ChainScratch>],
    colliders: &[BoneCollider],
) {
    let caps = build_live_capsules(scene, chains, scratches, colliders);
    if caps.len() < 2 {
        return;
    }

    let mut pushes = vec![Vec3::ZERO; chains.len()];

    for i in 0..caps.len() {
        for j in (i + 1)..caps.len() {
            let a = &caps[i];
            let b = &caps[j];
            if a.chain_idx.is_some() && a.chain_idx == b.chain_idx {
                continue;
            }
            if a.chain_idx.is_none() && b.chain_idx.is_none() {
                continue;
            }

            let (pa, pb) = closest_points_segments(a.a, a.b, b.a, b.b);
            let delta = pa - pb;
            let dist = delta.length();
            let min_d = a.radius + b.radius;
            // Tiny slack only — radius must dominate overlap depth.
            let slack = (0.004 * min_d).min(0.002);
            if dist + 1e-6 >= min_d - slack {
                continue;
            }
            let depth = (min_d - slack) - dist;
            if depth <= 1e-6 {
                continue;
            }

            let n = if dist > 1e-5 {
                delta / dist
            } else {
                let mid_a = (a.a + a.b) * 0.5;
                let mid_b = (b.a + b.b) * 0.5;
                let mut d = mid_a - mid_b;
                if d.length_squared() < 1e-10 {
                    d = Vec3::X;
                }
                d.normalize()
            };

            // Softness = how much of the overlap to peel off this pass.
            // 0 → gentle (~20%), 1 → firm (~75%). Still not a hard snap.
            let soft = (0.5 * (a.softness + b.softness)).clamp(0.0, 1.0);
            let gain = 0.20 + 0.55 * soft;
            let push = depth * gain;

            match (a.chain_idx, b.chain_idx) {
                (Some(ca), Some(cb)) => {
                    pushes[ca] += n * (push * 0.5);
                    pushes[cb] -= n * (push * 0.5);
                }
                (Some(ca), None) => {
                    pushes[ca] += n * push;
                }
                (None, Some(cb)) => {
                    pushes[cb] -= n * push;
                }
                _ => {}
            }
        }
    }

    for (ci, push) in pushes.into_iter().enumerate() {
        if push.length_squared() < 1e-12 {
            continue;
        }
        let Some(scratch) = scratches[ci].as_ref() else {
            continue;
        };
        apply_chain_separation(&mut chains[ci], scratch, push);
    }
}

fn apply_chain_separation(chain: &mut SoftChain, scratch: &ChainScratch, push: Vec3) {
    let n = chain.curr_pos.len();
    if n < 2 {
        return;
    }
    let plane_p = scratch.rest[0];
    for i in 1..n {
        let w = i as f32 / (n - 1) as f32;
        let old = chain.curr_pos[i];
        let mut p = old + push * (0.35 + 0.65 * w);
        let parent = chain.curr_pos[i - 1];
        let len = scratch.seg_len[i];
        let rd = scratch.rest_dir[i];
        p = project_length(p, parent, len, rd, scratch.support_n);
        p = clamp_to_halfspace(p, plane_p, scratch.support_n);
        // Full absorb — separation must not inject Verlet velocity (that thrashed).
        let corr = p - old;
        chain.curr_pos[i] = p;
        chain.prev_pos[i] += corr;
    }
    chain.curr_pos[0] = scratch.rest[0];
    chain.prev_pos[0] = scratch.rest[0];
}

fn project_length(p: Vec3, parent: Vec3, len: f32, fallback: Vec3, support_n: Vec3) -> Vec3 {
    let mut d = p - parent;
    if d.length_squared() < 1e-12 {
        d = if fallback.length_squared() > 1e-8 {
            fallback * len
        } else {
            support_n * len
        };
    } else {
        d = d.normalize() * len;
    }
    parent + d
}

fn closest_points_segments(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3) -> (Vec3, Vec3) {
    let da = a1 - a0;
    let db = b1 - b0;
    let r = a0 - b0;
    let aa = da.dot(da).max(1e-12);
    let ee = db.dot(db).max(1e-12);
    let bb = da.dot(db);
    let cc = da.dot(r);
    let ff = db.dot(r);

    let denom = aa * ee - bb * bb;
    let (mut s, mut t) = if denom.abs() > 1e-10 {
        (
            ((bb * ff - cc * ee) / denom).clamp(0.0, 1.0),
            ((aa * ff - bb * cc) / denom).clamp(0.0, 1.0),
        )
    } else {
        (0.0, (ff / ee).clamp(0.0, 1.0))
    };

    let pa = a0 + da * s;
    t = ((pa - b0).dot(db) / ee).clamp(0.0, 1.0);
    let pb = b0 + db * t;
    s = ((pb - a0).dot(da) / aa).clamp(0.0, 1.0);
    (a0 + da * s, b0 + db * t)
}

fn clamp_to_halfspace(p: Vec3, plane_p: Vec3, n: Vec3) -> Vec3 {
    let side = (p - plane_p).dot(n);
    if side < 0.0 {
        p + n * (-side)
    } else {
        p
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

/// Debug wire capsules for bone colliders (Edit + Pose).
pub fn draw_bone_colliders(scene: &mut Scene, rig: &RigDocument) {
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
