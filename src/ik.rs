//! Auto-IK pull via CCD (stable interactive posing).
//!
//! The actual joint-rotation math lives in `mega_render::ik`; this module
//! only handles viewport picking / dragging and translates `BoneId` chains
//! into calls against the render engine's node-based solver.

use glam::{Quat, Vec2, Vec3};
use mega_render::{ccd_rotate_joint as render_ccd_rotate_joint, translate_bone_world, GizmoAxis, Scene};
use mega_ui::Rect;

use crate::gizmo::{gizmo_basis, pick_translate_axis};
use crate::pick::{ray_from_viewport, Ray};
use crate::rig::{BoneId, RigDocument, TransformSpace};

const CCD_ITERS: u32 = 16;
/// Effector + this many ancestors (Blender-like short Auto-IK).
const IK_DEPTH: usize = 5;

#[derive(Clone)]
pub struct IkPullDrag {
    pub bone: BoneId,
    pub axis: GizmoAxis,
    /// Root-of-chain → … → effector.
    pub chain: Vec<BoneId>,
    pub start_locals: Vec<Quat>,
    /// Local translation of chain root at grab (restored each frame).
    pub start_root_translation: Vec3,
    pub effector_start: Vec3,
    pub origin: Vec3,
    pub axis_dir: Vec3,
    pub plane_n: Vec3,
    pub plane_u: Vec3,
    pub plane_v: Vec3,
    pub grab: Vec3,
}

/// Ancestors of `effector` including itself, root first. Caps at `max_len`.
pub fn build_chain(rig: &RigDocument, effector: BoneId, max_len: usize) -> Vec<BoneId> {
    let mut chain = Vec::new();
    let mut cur = Some(effector);
    while let Some(id) = cur {
        chain.push(id);
        if chain.len() >= max_len {
            break;
        }
        cur = rig.bone(id).and_then(|b| b.parent);
    }
    chain.reverse();
    chain
}

pub fn begin_ik_pull(
    scene: &Scene,
    rig: &RigDocument,
    bone: BoneId,
    pivot: Vec3,
    space: TransformSpace,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
) -> Option<IkPullDrag> {
    let plane = pick_move_plane(scene, bone, pivot, space, viewport, cursor, radius)?;
    let chain = build_chain(rig, bone, IK_DEPTH);
    if chain.len() < 2 {
        return None;
    }

    let mut start_locals = Vec::with_capacity(chain.len());
    for &id in &chain {
        let q = scene.nodes.get(id.node)?.local.rotation;
        start_locals.push(q);
    }
    let start_root_translation = scene.nodes.get(chain[0].node)?.local.translation;
    let effector_start = scene.world_matrix(bone.node).transform_point3(Vec3::ZERO);

    Some(IkPullDrag {
        bone,
        axis: plane.axis,
        chain,
        start_locals,
        start_root_translation,
        effector_start,
        origin: pivot,
        axis_dir: plane.axis_dir,
        plane_n: plane.plane_n,
        plane_u: plane.plane_u,
        plane_v: plane.plane_v,
        grab: plane.grab,
    })
}

pub fn apply_ik_pull(scene: &mut Scene, drag: &IkPullDrag, viewport: Rect, cursor: Vec2) {
    let Some(target) = move_target(scene, drag, viewport, cursor) else {
        return;
    };

    restore_chain_pose(scene, drag);
    let tip = *drag.chain.last().unwrap();
    for _ in 0..CCD_ITERS {
        for i in (0..drag.chain.len() - 1).rev() {
            render_ccd_rotate_joint(scene, drag.chain[i].node, tip.node, target, 1.0);
        }
    }
    // Free the root so tip can reach anywhere (no hard stop at chain length).
    pull_root_to_close_gap(scene, drag.chain[0], tip, target);
}

fn restore_chain_pose(scene: &mut Scene, drag: &IkPullDrag) {
    for (&id, &q) in drag.chain.iter().zip(drag.start_locals.iter()) {
        if let Some(n) = scene.nodes.get_mut(id.node) {
            n.local.rotation = q;
        }
    }
    if let Some(n) = scene.nodes.get_mut(drag.chain[0].node) {
        n.local.translation = drag.start_root_translation;
    }
}

/// Translate chain root in world so tip lands on target.
pub(crate) fn pull_root_to_close_gap(
    scene: &mut Scene,
    root: BoneId,
    tip: BoneId,
    target: Vec3,
) {
    let tip_pos = scene.world_matrix(tip.node).transform_point3(Vec3::ZERO);
    let residual = target - tip_pos;
    if residual.length_squared() < 1e-12 {
        return;
    }
    translate_bone_world(scene, root.node, residual);
}

pub(crate) struct MovePlane {
    pub axis: GizmoAxis,
    pub axis_dir: Vec3,
    pub plane_n: Vec3,
    pub plane_u: Vec3,
    pub plane_v: Vec3,
    pub grab: Vec3,
}

pub(crate) fn pick_move_plane(
    scene: &Scene,
    bone: BoneId,
    pivot: Vec3,
    space: TransformSpace,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
) -> Option<MovePlane> {
    let axis = pick_translate_axis(scene, bone, pivot, space, viewport, cursor, radius)?;
    let (_, basis) = gizmo_basis(scene, bone, space)?;
    let x = basis.x_axis;
    let y = basis.y_axis;
    let z = basis.z_axis;
    let ray = ray_from_viewport(scene, viewport, cursor)?;

    let (axis_dir, plane_n, plane_u, plane_v) = match axis {
        GizmoAxis::X => (x, x, y, z),
        GizmoAxis::Y => (y, y, x, z),
        GizmoAxis::Z => (z, z, x, y),
        GizmoAxis::Xy => ((x + y).normalize_or_zero(), z, x, y),
        GizmoAxis::Yz => ((y + z).normalize_or_zero(), x, y, z),
        GizmoAxis::Zx => ((z + x).normalize_or_zero(), y, z, x),
        GizmoAxis::Uniform => return None,
    };

    let grab = match axis {
        GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
            let view = (scene.camera.eye - pivot).normalize_or_zero();
            let n = axis_dir.cross(view.cross(axis_dir)).normalize_or_zero();
            ray_plane_point(&ray, pivot, n).unwrap_or(pivot)
        }
        _ => ray_plane_point(&ray, pivot, plane_n).unwrap_or(pivot),
    };

    Some(MovePlane {
        axis,
        axis_dir,
        plane_n,
        plane_u,
        plane_v,
        grab,
    })
}

fn move_target(
    scene: &Scene,
    drag: &IkPullDrag,
    viewport: Rect,
    cursor: Vec2,
) -> Option<Vec3> {
    target_from_plane(
        scene,
        viewport,
        cursor,
        drag.axis,
        drag.origin,
        drag.axis_dir,
        drag.plane_n,
        drag.plane_u,
        drag.plane_v,
        drag.grab,
        drag.effector_start,
    )
}

pub(crate) fn target_from_plane(
    scene: &Scene,
    viewport: Rect,
    cursor: Vec2,
    axis: GizmoAxis,
    origin: Vec3,
    axis_dir: Vec3,
    plane_n: Vec3,
    plane_u: Vec3,
    plane_v: Vec3,
    grab: Vec3,
    effector_start: Vec3,
) -> Option<Vec3> {
    let ray = ray_from_viewport(scene, viewport, cursor)?;
    let hit = match axis {
        GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
            let view = (scene.camera.eye - origin).normalize_or_zero();
            let n = axis_dir.cross(view.cross(axis_dir)).normalize_or_zero();
            ray_plane_point(&ray, origin, n)?
        }
        _ => ray_plane_point(&ray, origin, plane_n)?,
    };
    let delta = hit - grab;
    Some(match axis {
        GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
            effector_start + axis_dir * delta.dot(axis_dir)
        }
        GizmoAxis::Xy | GizmoAxis::Yz | GizmoAxis::Zx => {
            effector_start + plane_u * delta.dot(plane_u) + plane_v * delta.dot(plane_v)
        }
        GizmoAxis::Uniform => effector_start,
    })
}

/// Rotate `joint` so the chain tip moves toward `target`. `weight` 0..1 scales the step.
/// Thin `BoneId` wrapper over the render engine's node-based CCD step.
pub(crate) fn ccd_rotate_joint(scene: &mut Scene, joint: BoneId, tip: BoneId, target: Vec3, weight: f32) {
    render_ccd_rotate_joint(scene, joint.node, tip.node, target, weight);
}

fn ray_plane_point(ray: &Ray, point: Vec3, normal: Vec3) -> Option<Vec3> {
    let n = normal.normalize_or_zero();
    if n.length_squared() < 1e-8 {
        return None;
    }
    let denom = ray.dir.dot(n);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (point - ray.origin).dot(n) / denom;
    if t < 0.0 {
        return None;
    }
    Some(ray.origin + ray.dir * t)
}
