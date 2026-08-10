//! Rotation gizmo: draw via mega-render [`DebugDraw::gizmo`], FK via local bone rotation.

use std::f32::consts::TAU;

use glam::{Mat3, Quat, Vec2, Vec3};
use mega_render::{
    gizmo_ring_basis, gizmo_screen_size, GizmoAxis, GizmoMode, GizmoOpts, GizmoRotateArc, Scene,
};
use mega_ui::Rect;

use crate::pick::{project_world_to_screen, ray_from_viewport, Ray};
use crate::rig::BoneId;

#[derive(Clone, Copy)]
pub struct RotateDrag {
    pub bone: BoneId,
    pub axis: GizmoAxis,
    pub world_axis: Vec3,
    pub origin: Vec3,
    pub start_local_rot: Quat,
    pub start_angle: f32,
    pub current_angle: f32,
    /// Frozen ring basis at grab (for rotate-arc feedback).
    pub u: Vec3,
    pub v: Vec3,
}

pub fn gizmo_radius(scene: &Scene, bone: BoneId, viewport_h: f32) -> f32 {
    let origin = scene.world_matrix(bone.node).transform_point3(Vec3::ZERO);
    let dist = (scene.camera.eye - origin).length();
    gizmo_screen_size(dist, scene.camera.fov_y, viewport_h, 110.0).clamp(0.06, 2.0)
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

fn world_axis_of(basis: Mat3, axis: GizmoAxis) -> Vec3 {
    match axis {
        GizmoAxis::X => basis.x_axis,
        GizmoAxis::Y => basis.y_axis,
        GizmoAxis::Z => basis.z_axis,
        _ => basis.y_axis,
    }
}

fn rotate_axes() -> [GizmoAxis; 3] {
    [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z]
}

pub fn draw_gizmo(
    scene: &mut Scene,
    bone: BoneId,
    radius: f32,
    hover: Option<GizmoAxis>,
    drag: Option<&RotateDrag>,
) {
    let Some((origin, basis)) = bone_world_basis(scene, bone) else {
        return;
    };
    let rotation = Quat::from_mat3(&basis);
    let highlight = drag.map(|d| d.axis).or(hover);
    let rotate_arc = drag.map(|d| GizmoRotateArc {
        axis: d.axis,
        u: d.u,
        v: d.v,
        start: d.start_angle,
        current: d.current_angle,
    });
    let eye = scene.camera.eye;
    scene.debug.gizmo(
        origin,
        rotation,
        GizmoOpts {
            mode: GizmoMode::Rotate,
            size: radius,
            highlight,
            eye: Some(eye),
            rotate_arc,
            depth_test: false,
        },
    );
}

pub fn pick_axis(
    scene: &Scene,
    bone: BoneId,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
) -> Option<(GizmoAxis, f32, Vec3, Vec3)> {
    let ray = ray_from_viewport(scene, viewport, cursor)?;
    let (origin, basis) = bone_world_basis(scene, bone)?;

    let mut best: Option<(f32, GizmoAxis, f32, Vec3, Vec3)> = None;
    for axis in rotate_axes() {
        let n = world_axis_of(basis, axis);
        let (u, v) = gizmo_ring_basis(axis, basis.x_axis, basis.y_axis, basis.z_axis);
        let Some((hit, angle)) = ray_ring_hit(&ray, origin, n, u, v) else {
            continue;
        };
        let depth = (hit - ray.origin).length();
        let radial = (hit - origin - n * (hit - origin).dot(n)).length();
        let radial_err = (radial - radius).abs() / radius.max(1e-4);
        if radial_err > 0.45 {
            continue;
        }
        let score = depth + radial_err * radius * 4.0;
        if best.is_none_or(|(s, ..)| score < s) {
            best = Some((score, axis, angle, u, v));
        }
    }
    best.map(|(_, a, ang, u, v)| (a, ang, u, v))
}

fn ray_ring_hit(ray: &Ray, origin: Vec3, axis: Vec3, u: Vec3, v: Vec3) -> Option<(Vec3, f32)> {
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
    let rel = hit - origin;
    let angle = v.dot(rel).atan2(u.dot(rel));
    Some((hit, angle))
}

pub fn begin_rotate(
    scene: &Scene,
    bone: BoneId,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
) -> Option<RotateDrag> {
    let (axis, angle, u, v) = pick_axis(scene, bone, viewport, cursor, radius)?;
    let (origin, basis) = bone_world_basis(scene, bone)?;
    let local = scene.nodes.get(bone.node)?.local.rotation;
    Some(RotateDrag {
        bone,
        axis,
        world_axis: world_axis_of(basis, axis),
        origin,
        start_local_rot: local,
        start_angle: angle,
        current_angle: angle,
        u,
        v,
    })
}

pub fn hover_axis(
    scene: &Scene,
    bone: BoneId,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
) -> Option<GizmoAxis> {
    pick_axis(scene, bone, viewport, cursor, radius).map(|(a, ..)| a)
}

/// Rotate selected bone only; children follow through node parents (FK).
pub fn apply_rotate(scene: &mut Scene, drag: &mut RotateDrag, viewport: Rect, cursor: Vec2) {
    let Some(ray) = ray_from_viewport(scene, viewport, cursor) else {
        return;
    };
    let Some((_, angle)) = ray_ring_hit(&ray, drag.origin, drag.world_axis, drag.u, drag.v) else {
        return;
    };
    drag.current_angle = angle;

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
