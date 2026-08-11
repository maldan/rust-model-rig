//! Soft whole-chain pull: CCD with falloff toward the root + free root slide.

use glam::{Quat, Vec2, Vec3};
use mega_render::{GizmoAxis, Scene};
use mega_ui::Rect;

use crate::ik::{
    build_chain, ccd_rotate_joint, pick_move_plane, pull_root_to_close_gap, target_from_plane,
};
use crate::rig::{BoneId, RigDocument, TransformSpace};

const CCD_ITERS: u32 = 20;
const SOFT_DEPTH: usize = 32;
const FALLOFF: f32 = 1.75;

#[derive(Clone)]
pub struct VerletDrag {
    pub bone: BoneId,
    pub axis: GizmoAxis,
    pub chain: Vec<BoneId>,
    pub start_locals: Vec<Quat>,
    pub start_root_translation: Vec3,
    pub effector_start: Vec3,
    pub origin: Vec3,
    pub axis_dir: Vec3,
    pub plane_n: Vec3,
    pub plane_u: Vec3,
    pub plane_v: Vec3,
    pub grab: Vec3,
}

pub fn begin_verlet_pull(
    scene: &Scene,
    rig: &RigDocument,
    bone: BoneId,
    pivot: Vec3,
    space: TransformSpace,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
) -> Option<VerletDrag> {
    let plane = pick_move_plane(scene, bone, pivot, space, viewport, cursor, radius)?;
    let chain = build_chain(rig, bone, SOFT_DEPTH);
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

    Some(VerletDrag {
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

pub fn apply_verlet_pull(scene: &mut Scene, drag: &VerletDrag, viewport: Rect, cursor: Vec2) {
    let Some(target) = target_from_plane(
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
    ) else {
        return;
    };

    for (&id, &q) in drag.chain.iter().zip(drag.start_locals.iter()) {
        if let Some(n) = scene.nodes.get_mut(id.node) {
            n.local.rotation = q;
        }
    }
    if let Some(n) = scene.nodes.get_mut(drag.chain[0].node) {
        n.local.translation = drag.start_root_translation;
    }

    let tip = *drag.chain.last().unwrap();
    let n = drag.chain.len();
    for _ in 0..CCD_ITERS {
        for i in (0..n - 1).rev() {
            let t = if n <= 2 {
                1.0
            } else {
                i as f32 / (n - 2) as f32
            };
            let weight = t.powf(FALLOFF).clamp(0.05, 1.0);
            ccd_rotate_joint(scene, drag.chain[i], tip, target, weight);
        }
    }
    pull_root_to_close_gap(scene, drag.chain[0], tip, target);
}
