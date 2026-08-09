//! Rotation gizmo (RGB axis rings). FK = change selected bone local rotation;
//! children follow via the scene-graph parent chain.

use std::f32::consts::TAU;

use glam::{Mat3, Quat, Vec2, Vec3};
use mega_render::Scene;
use mega_ui::Rect;

use crate::pick::{project_world_to_screen, ray_from_viewport, Ray};
use crate::rig::BoneId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub fn color(self) -> [f32; 4] {
        match self {
            Self::X => [0.95, 0.25, 0.22, 1.0],
            Self::Y => [0.35, 0.9, 0.3, 1.0],
            Self::Z => [0.3, 0.55, 1.0, 1.0],
        }
    }

    pub fn all() -> [Axis; 3] {
        [Self::X, Self::Y, Self::Z]
    }
}

#[derive(Clone, Copy)]
pub struct RotateDrag {
    pub bone: BoneId,
    pub axis: Axis,
    pub world_axis: Vec3,
    pub origin: Vec3,
    pub start_local_rot: Quat,
    pub start_angle: f32,
}

pub fn gizmo_radius(_scene: &Scene, _bone: BoneId, hint_len: f32) -> f32 {
    (hint_len * 0.55).clamp(0.04, 0.6)
}

pub fn bone_world_basis(scene: &Scene, bone: BoneId) -> Option<(Vec3, Mat3)> {
    let m = scene.world_matrix(bone.node);
    let origin = m.transform_point3(Vec3::ZERO);
    let x = m.transform_vector3(Vec3::X).normalize_or_zero();
    let mut y = m.transform_vector3(Vec3::Y).normalize_or_zero();
    let z = x.cross(y).normalize_or_zero();
    y = z.cross(x).normalize_or_zero();
    if x.length_squared() < 1e-8 || y.length_squared() < 1e-8 {
        return None;
    }
    Some((origin, Mat3::from_cols(x, y, z)))
}

fn world_axis_of(basis: Mat3, axis: Axis) -> Vec3 {
    match axis {
        Axis::X => basis.x_axis,
        Axis::Y => basis.y_axis,
        Axis::Z => basis.z_axis,
    }
}

pub fn draw_gizmo(
    scene: &mut Scene,
    bone: BoneId,
    radius: f32,
    hover: Option<Axis>,
    active: Option<Axis>,
) {
    let Some((origin, basis)) = bone_world_basis(scene, bone) else {
        return;
    };

    for axis in Axis::all() {
        let dir = world_axis_of(basis, axis);
        let mut col = axis.color();
        if active == Some(axis) || hover == Some(axis) {
            col[0] = (col[0] * 1.2).min(1.0);
            col[1] = (col[1] * 1.2).min(1.0);
            col[2] = (col[2] * 1.2).min(1.0);
            col[3] = 1.0;
        } else {
            col[3] = 0.85;
        }
        scene
            .debug
            .line_overlay(origin, origin + dir * radius * 1.15, col);
    }

    const SEGMENTS: usize = 48;
    for axis in Axis::all() {
        let n = world_axis_of(basis, axis);
        let (t, b) = ring_basis(n);
        let mut col = axis.color();
        let thick = active == Some(axis) || hover == Some(axis);
        col[3] = if thick { 1.0 } else { 0.75 };
        let mut prev = origin + t * radius;
        for i in 1..=SEGMENTS {
            let a = (i as f32 / SEGMENTS as f32) * TAU;
            let p = origin + (t * a.cos() + b * a.sin()) * radius;
            scene.debug.line_overlay(prev, p, col);
            if thick {
                let a0 = ((i - 1) as f32 / SEGMENTS as f32) * TAU;
                let prev2 = origin + (t * a0.cos() + b * a0.sin()) * (radius * 1.02);
                let p2 = origin + (t * a.cos() + b * a.sin()) * (radius * 1.02);
                scene.debug.line_overlay(prev2, p2, col);
            }
            prev = p;
        }
    }
}

fn ring_basis(axis: Vec3) -> (Vec3, Vec3) {
    let axis = axis.normalize_or_zero();
    let helper = if axis.cross(Vec3::Y).length_squared() > 1e-4 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let t = axis.cross(helper).normalize_or_zero();
    let b = axis.cross(t).normalize_or_zero();
    (t, b)
}

pub fn pick_axis(
    scene: &Scene,
    bone: BoneId,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
) -> Option<(Axis, f32)> {
    let ray = ray_from_viewport(scene, viewport, cursor)?;
    let (origin, basis) = bone_world_basis(scene, bone)?;

    let mut best: Option<(f32, Axis, f32)> = None;
    for axis in Axis::all() {
        let n = world_axis_of(basis, axis);
        let Some((hit, angle)) = ray_ring_hit(&ray, origin, n) else {
            continue;
        };
        let depth = (hit - ray.origin).length();
        let radial = (hit - origin - n * (hit - origin).dot(n)).length();
        let radial_err = (radial - radius).abs() / radius.max(1e-4);
        if radial_err > 0.35 {
            continue;
        }
        let score = depth + radial_err * radius * 4.0;
        if best.is_none_or(|(s, _, _)| score < s) {
            best = Some((score, axis, angle));
        }
    }
    best.map(|(_, a, ang)| (a, ang))
}

fn ray_ring_hit(ray: &Ray, origin: Vec3, axis: Vec3) -> Option<(Vec3, f32)> {
    let axis = axis.normalize_or_zero();
    if axis.length_squared() < 1e-8 {
        return None;
    }
    let denom = ray.dir.dot(axis);
    if denom.abs() < 1e-5 {
        return None;
    }
    let t = (origin - ray.origin).dot(axis) / denom;
    if t < 0.0 {
        return None;
    }
    let hit = ray.origin + ray.dir * t;
    let (tangent, bitangent) = ring_basis(axis);
    let v = hit - origin;
    let angle = v.dot(bitangent).atan2(v.dot(tangent));
    Some((hit, angle))
}

pub fn begin_rotate(
    scene: &Scene,
    bone: BoneId,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
) -> Option<RotateDrag> {
    let (axis, angle) = pick_axis(scene, bone, viewport, cursor, radius)?;
    let (origin, basis) = bone_world_basis(scene, bone)?;
    let local = scene.nodes.get(bone.node)?.local.rotation;
    Some(RotateDrag {
        bone,
        axis,
        world_axis: world_axis_of(basis, axis),
        origin,
        start_local_rot: local,
        start_angle: angle,
    })
}

pub fn hover_axis(
    scene: &Scene,
    bone: BoneId,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
) -> Option<Axis> {
    pick_axis(scene, bone, viewport, cursor, radius).map(|(a, _)| a)
}

/// Rotate selected bone only; children follow through node parents (FK).
pub fn apply_rotate(scene: &mut Scene, drag: &RotateDrag, viewport: Rect, cursor: Vec2) {
    let Some(ray) = ray_from_viewport(scene, viewport, cursor) else {
        return;
    };
    let Some((_, angle)) = ray_ring_hit(&ray, drag.origin, drag.world_axis) else {
        return;
    };
    let mut delta = angle - drag.start_angle;
    while delta > std::f32::consts::PI {
        delta -= TAU;
    }
    while delta < -std::f32::consts::PI {
        delta += TAU;
    }

    let world_delta = Quat::from_axis_angle(drag.world_axis, delta);
    let parent_world = scene
        .nodes
        .get(drag.bone.node)
        .and_then(|n| n.parent)
        .map(|p| scene.world_matrix(p))
        .unwrap_or(glam::Mat4::IDENTITY);
    let parent_rot = Quat::from_mat3(&Mat3::from_mat4(parent_world)).normalize();
    let new_local =
        (parent_rot.inverse() * world_delta * parent_rot * drag.start_local_rot).normalize();

    if let Some(n) = scene.nodes.get_mut(drag.bone.node) {
        n.local.rotation = new_local;
    }
}

#[allow(dead_code)]
pub fn gizmo_screen_center(scene: &Scene, bone: BoneId, viewport: Rect) -> Option<Vec2> {
    let world = scene.world_matrix(bone.node).transform_point3(Vec3::ZERO);
    project_world_to_screen(scene, viewport, world)
}
