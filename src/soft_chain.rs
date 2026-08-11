//! Soft bone chains: world-gravity Verlet + spring to animated pose + support plane.

use glam::{Quat, Vec3};
use mega_render::{LineOpts, Scene, Transform};
use std::collections::HashMap;

use crate::ik::quat_from_matrix;
use crate::rig::{AppMode, BoneId, RigDocument, SoftChain};

const CONSTRAINT_ITERS: u32 = 8;
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
    // Particles: one per bone + virtual tip continuing the last segment.
    let n = n_bones + 1;

    let mut rest = Vec::with_capacity(n);
    for &b in &chain.bones {
        rest.push(scene.world_matrix(b.node).transform_point3(Vec3::ZERO));
    }
    // Tip extends the authored chain direction (not raw +Y) so the last bone
    // doesn't kink relative to its parent segment.
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

    // Rest segment directions (animated bind). Angle cones & spring targets use these.
    let mut rest_dir = Vec::with_capacity(n);
    rest_dir.push(Vec3::ZERO);
    for i in 1..n {
        rest_dir.push((rest[i] - rest[i - 1]).normalize_or_zero());
    }

    let root_world = scene.world_matrix(chain.bones[0].node);

    if !chain.initialized || chain.curr_pos.len() != n {
        chain.prev_pos = rest.clone();
        chain.curr_pos = rest.clone();
        chain.prev_root_world = root_world;
        chain.initialized = true;
    } else {
        // Motion inheritance: move/rotate particles with the soft root so body
        // animation isn't treated as a teleport (fixes thrashing while posing).
        let delta = root_world * chain.prev_root_world.inverse();
        for i in 1..n {
            chain.curr_pos[i] = delta.transform_point3(chain.curr_pos[i]);
            chain.prev_pos[i] = delta.transform_point3(chain.prev_pos[i]);
        }
        chain.prev_root_world = root_world;
    }

    // Pin root after inheritance.
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
    let max_angle = chain.max_angle.max(0.05);

    let mut seg_len = Vec::with_capacity(n);
    seg_len.push(0.0);
    for i in 1..n_bones {
        seg_len.push(chain.lengths[i].max(1e-4));
    }
    seg_len.push(chain.tip_length.max(1e-4));

    // Tip rest axis in the tip bone's local space (captured in bind pose).
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

    for _ in 0..SUBSTEPS {
        chain.curr_pos[0] = rest[0];
        chain.prev_pos[0] = rest[0];

        for i in 1..n {
            // Spring toward bind direction from the *current* parent particle
            // (not absolute rest world pos — that shears the chain and bends kids weirdly).
            let parent = chain.curr_pos[i - 1];
            let rest_target = if rest_dir[i].length_squared() > 1e-8 {
                parent + rest_dir[i] * seg_len[i]
            } else {
                rest[i]
            };

            let vel = chain.curr_pos[i] - chain.prev_pos[i];
            let spring = (rest_target - chain.curr_pos[i]) * stiff;
            let next = chain.curr_pos[i] + vel * vel_keep + (gravity + spring) * h2;
            chain.prev_pos[i] = chain.curr_pos[i];
            chain.curr_pos[i] = next;
        }

        let plane_p = chain.curr_pos[0];
        for _ in 0..CONSTRAINT_ITERS {
            chain.curr_pos[0] = rest[0];
            for i in 1..n {
                let parent = chain.curr_pos[i - 1];
                let len = seg_len[i];
                let rd = rest_dir[i];

                let mut p = project_length(chain.curr_pos[i], parent, len, rd, support_n);

                // Support half-space: stay outside the chest.
                let side = (p - plane_p).dot(support_n);
                if side < 0.0 {
                    p += support_n * (-side);
                    p = project_length(p, parent, len, rd, support_n);

                    // Prefer the rest direction flattened onto the plane when gravity
                    // is pushing into the body (lying on back) — avoids random sideways slides.
                    if rd.length_squared() > 1e-8 {
                        let mut prefer = rd - support_n * rd.dot(support_n);
                        if prefer.length_squared() < 1e-10 {
                            // Rest is along the normal; pick a stable tangent from gravity.
                            prefer = gravity - support_n * gravity.dot(support_n);
                        }
                        if prefer.length_squared() > 1e-10 {
                            let into = (p - parent).dot(support_n);
                            if into < 1e-4 {
                                let tang = prefer.normalize();
                                // Blend toward in-plane rest so kids don't crab-walk.
                                let cur = (p - parent).normalize_or_zero();
                                let blended = (cur + tang * 0.35).normalize_or_zero();
                                if blended.length_squared() > 1e-8 {
                                    p = parent + blended * len;
                                    p = clamp_to_halfspace(p, plane_p, support_n);
                                    p = project_length(p, parent, len, rd, support_n);
                                }
                            }
                        }
                    }

                    // Kill velocity into the plane (stop buzzing / tunneling feel).
                    let vel = chain.curr_pos[i] - chain.prev_pos[i];
                    let vn = vel.dot(support_n);
                    if vn < 0.0 {
                        let v_tang = vel - support_n * vn;
                        chain.prev_pos[i] = p - v_tang;
                    }
                }

                // Angle cone around *bind segment* direction (not rest[i]-curr[i-1]).
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
                            // Keep outside chest after angle clamp.
                            p = clamp_to_halfspace(p, plane_p, support_n);
                            p = project_length(p, parent, len, rd, support_n);
                        }
                    }
                }

                chain.curr_pos[i] = p;
            }
        }
    }

    // Write rotations from particle chain (each bone aims at the next particle).
    for i in 0..n_bones {
        let target = chain.curr_pos[i + 1];
        if i + 1 < n_bones {
            swing_bone_to(scene, chain.bones[i], chain.bones[i + 1], target);
        } else {
            let tip = chain.bones[i];
            let m = scene.world_matrix(tip.node);
            let origin = m.transform_point3(Vec3::ZERO);
            let (_, tip_rot, _) = m.to_scale_rotation_translation();
            let from = (tip_rot.normalize() * tip_local_axis).normalize_or_zero();
            let to = (chain.curr_pos[n - 1] - origin).normalize_or_zero();
            apply_swing(scene, tip, from, to);
        }
    }
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
