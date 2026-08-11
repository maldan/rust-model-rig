//! Bone gizmos: rotate (pose/edit) + translate (edit). FK via local transforms.

use std::f32::consts::TAU;

use glam::{Mat3, Quat, Vec2, Vec3};
use mega_render::{
    gizmo_ring_basis, gizmo_screen_size, GizmoAxis, GizmoMode, GizmoOpts, GizmoRotateArc, Scene,
};
use mega_ui::Rect;

use crate::pick::{project_world_to_screen, ray_from_viewport, Ray};
use crate::rig::BoneId;

#[derive(Clone)]
pub struct RotateDrag {
    pub bone: BoneId,
    pub axis: GizmoAxis,
    pub world_axis: Vec3,
    /// Shared pivot (median of selection roots).
    pub origin: Vec3,
    /// `(bone, start_local_rot, start_world_pos, start_parent_world)`.
    pub bones: Vec<(BoneId, Quat, Vec3, glam::Mat4)>,
    pub start_angle: f32,
    pub current_angle: f32,
    /// Frozen ring basis at grab (for rotate-arc feedback).
    pub u: Vec3,
    pub v: Vec3,
}

#[derive(Clone)]
pub struct TranslateDrag {
    pub bone: BoneId,
    pub axis: GizmoAxis,
    pub origin: Vec3,
    pub axis_dir: Vec3,
    pub plane_n: Vec3,
    pub plane_u: Vec3,
    pub plane_v: Vec3,
    pub grab: Vec3,
    /// `(bone, start_world, start_parent_world)` — selection roots only.
    pub bones: Vec<(BoneId, Vec3, glam::Mat4)>,
}

pub fn gizmo_radius_at(scene: &Scene, origin: Vec3, viewport_h: f32) -> f32 {
    let dist = (scene.camera.eye - origin).length();
    gizmo_screen_size(dist, scene.camera.fov_y, viewport_h, 110.0).clamp(0.06, 2.0)
}

pub fn gizmo_radius(scene: &Scene, bone: BoneId, viewport_h: f32) -> f32 {
    let origin = scene.world_matrix(bone.node).transform_point3(Vec3::ZERO);
    gizmo_radius_at(scene, origin, viewport_h)
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

pub fn draw_rotate_gizmo(
    scene: &mut Scene,
    bone: BoneId,
    pivot: Vec3,
    radius: f32,
    hover: Option<GizmoAxis>,
    drag: Option<&RotateDrag>,
) {
    let Some((_, basis)) = bone_world_basis(scene, bone) else {
        return;
    };
    let origin = drag.map(|d| d.origin).unwrap_or(pivot);
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

pub fn draw_translate_gizmo(
    scene: &mut Scene,
    bone: BoneId,
    pivot: Vec3,
    radius: f32,
    hover: Option<GizmoAxis>,
    drag: Option<&TranslateDrag>,
) {
    let Some((_, basis)) = bone_world_basis(scene, bone) else {
        return;
    };
    let origin = drag.map(|d| d.origin).unwrap_or(pivot);
    let rotation = Quat::from_mat3(&basis);
    let highlight = drag.map(|d| d.axis).or(hover);
    scene.debug.gizmo(
        origin,
        rotation,
        GizmoOpts {
            mode: GizmoMode::Translate,
            size: radius,
            highlight,
            eye: Some(scene.camera.eye),
            rotate_arc: None,
            depth_test: false,
        },
    );
}

pub fn pick_rotate_axis(
    scene: &Scene,
    bone: BoneId,
    origin: Vec3,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
) -> Option<(GizmoAxis, f32, Vec3, Vec3)> {
    let ray = ray_from_viewport(scene, viewport, cursor)?;
    let (_, basis) = bone_world_basis(scene, bone)?;

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

fn ray_plane_point(ray: &Ray, plane_o: Vec3, plane_n: Vec3) -> Option<Vec3> {
    let n = plane_n.normalize_or_zero();
    if n.length_squared() < 1e-8 {
        return None;
    }
    let denom = ray.dir.dot(n);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (plane_o - ray.origin).dot(n) / denom;
    if t < 0.0 {
        return None;
    }
    Some(ray.origin + ray.dir * t)
}

fn ray_segment_dist(ray: &Ray, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let ao = a - ray.origin;
    let ab_len_sq = ab.length_squared();
    if ab_len_sq < 1e-12 {
        let t = ao.dot(ray.dir).max(0.0);
        let p = ray.origin + ray.dir * t;
        return (p - a).length();
    }
    let d = ray.dir;
    let r = ab;
    let w0 = ray.origin - a;
    let aa = d.dot(d);
    let bb = r.dot(r);
    let cc = d.dot(r);
    let det = aa * bb - cc * cc;
    let (t, s) = if det.abs() < 1e-10 {
        let s = (w0.dot(r) / bb).clamp(0.0, 1.0);
        let t = (r * s - w0).dot(d) / aa;
        (t.max(0.0), s)
    } else {
        let _t = ((cc * w0.dot(r) - bb * w0.dot(d)) / det).max(0.0);
        let s = ((aa * w0.dot(r) - cc * w0.dot(d)) / det).clamp(0.0, 1.0);
        let t = (r * s - w0).dot(d) / aa;
        (t.max(0.0), s)
    };
    let p_ray = ray.origin + d * t;
    let p_seg = a + r * s;
    (p_ray - p_seg).length()
}

fn ray_plane_quad_dist(ray: &Ray, origin: Vec3, u: Vec3, v: Vec3, size: f32) -> Option<f32> {
    let n = u.cross(v).normalize_or_zero();
    let hit = ray_plane_point(ray, origin, n)?;
    let rel = hit - origin;
    let su = rel.dot(u);
    let sv = rel.dot(v);
    let pad = size * 0.22;
    if su < pad || sv < pad || su > size * 0.55 || sv > size * 0.55 {
        return None;
    }
    Some((hit - ray.origin).length())
}

pub fn pick_translate_axis(
    scene: &Scene,
    bone: BoneId,
    origin: Vec3,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
) -> Option<GizmoAxis> {
    let ray = ray_from_viewport(scene, viewport, cursor)?;
    let (_, basis) = bone_world_basis(scene, bone)?;
    let x = basis.x_axis;
    let y = basis.y_axis;
    let z = basis.z_axis;
    let thresh = radius * 0.12;

    let mut best: Option<(f32, GizmoAxis)> = None;
    let mut consider = |d: f32, axis: GizmoAxis| {
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, axis));
        }
    };

    for (axis, dir) in [(GizmoAxis::X, x), (GizmoAxis::Y, y), (GizmoAxis::Z, z)] {
        let d = ray_segment_dist(&ray, origin, origin + dir * radius);
        if d < thresh {
            consider(d, axis);
        }
    }
    for (axis, a, b) in [
        (GizmoAxis::Xy, x, y),
        (GizmoAxis::Yz, y, z),
        (GizmoAxis::Zx, z, x),
    ] {
        if let Some(d) = ray_plane_quad_dist(&ray, origin, a, b, radius) {
            consider(d, axis);
        }
    }
    best.map(|(_, a)| a)
}

pub fn begin_rotate(
    scene: &Scene,
    bone: BoneId,
    roots: &[BoneId],
    pivot: Vec3,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
) -> Option<RotateDrag> {
    let (axis, angle, u, v) = pick_rotate_axis(scene, bone, pivot, viewport, cursor, radius)?;
    let (_, basis) = bone_world_basis(scene, bone)?;
    let mut bones = Vec::new();
    let ids = if roots.is_empty() {
        std::slice::from_ref(&bone)
    } else {
        roots
    };
    for &id in ids {
        let Some(n) = scene.nodes.get(id.node) else {
            continue;
        };
        let parent_world = n
            .parent
            .map(|p| scene.world_matrix(p))
            .unwrap_or(glam::Mat4::IDENTITY);
        let world = scene.world_matrix(id.node).transform_point3(Vec3::ZERO);
        bones.push((id, n.local.rotation, world, parent_world));
    }
    if bones.is_empty() {
        return None;
    }
    Some(RotateDrag {
        bone,
        axis,
        world_axis: world_axis_of(basis, axis),
        origin: pivot,
        bones,
        start_angle: angle,
        current_angle: angle,
        u,
        v,
    })
}

pub fn begin_translate(
    scene: &Scene,
    bone: BoneId,
    roots: &[BoneId],
    pivot: Vec3,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
) -> Option<TranslateDrag> {
    let axis = pick_translate_axis(scene, bone, pivot, viewport, cursor, radius)?;
    let (_, basis) = bone_world_basis(scene, bone)?;
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

    let mut bones = Vec::new();
    let ids = if roots.is_empty() {
        std::slice::from_ref(&bone)
    } else {
        roots
    };
    for &id in ids {
        let Some(n) = scene.nodes.get(id.node) else {
            continue;
        };
        let parent_world = n
            .parent
            .map(|p| scene.world_matrix(p))
            .unwrap_or(glam::Mat4::IDENTITY);
        let world = scene.world_matrix(id.node).transform_point3(Vec3::ZERO);
        bones.push((id, world, parent_world));
    }
    if bones.is_empty() {
        return None;
    }
    Some(TranslateDrag {
        bone,
        axis,
        origin: pivot,
        axis_dir,
        plane_n,
        plane_u,
        plane_v,
        grab,
        bones,
    })
}

pub fn hover_axis(
    scene: &Scene,
    bone: BoneId,
    origin: Vec3,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
) -> Option<GizmoAxis> {
    pick_rotate_axis(scene, bone, origin, viewport, cursor, radius).map(|(a, ..)| a)
}

pub fn hover_translate_axis(
    scene: &Scene,
    bone: BoneId,
    origin: Vec3,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
) -> Option<GizmoAxis> {
    pick_translate_axis(scene, bone, origin, viewport, cursor, radius)
}

/// Rotate selection roots around shared pivot (children follow via FK).
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
    for &(bone, start_local, start_world, parent_world) in &drag.bones {
        let parent_rot = Quat::from_mat3(&Mat3::from_mat4(parent_world)).normalize();
        let new_local_rot =
            (parent_rot.inverse() * world_delta * parent_rot * start_local).normalize();
        let new_world = drag.origin + world_delta * (start_world - drag.origin);
        let new_local_pos = parent_world.inverse().transform_point3(new_world);
        if let Some(n) = scene.nodes.get_mut(bone.node) {
            n.local.rotation = new_local_rot;
            n.local.translation = new_local_pos;
        }
    }
}

pub fn apply_translate(scene: &mut Scene, drag: &mut TranslateDrag, viewport: Rect, cursor: Vec2) {
    let Some(ray) = ray_from_viewport(scene, viewport, cursor) else {
        return;
    };
    let hit = match drag.axis {
        GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
            let view = (scene.camera.eye - drag.origin).normalize_or_zero();
            let n = drag
                .axis_dir
                .cross(view.cross(drag.axis_dir))
                .normalize_or_zero();
            ray_plane_point(&ray, drag.origin, n)
        }
        _ => ray_plane_point(&ray, drag.origin, drag.plane_n),
    };
    let Some(hit) = hit else {
        return;
    };
    let delta = hit - drag.grab;
    let world_new_primary = match drag.axis {
        GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
            drag.origin + drag.axis_dir * delta.dot(drag.axis_dir)
        }
        GizmoAxis::Xy | GizmoAxis::Yz | GizmoAxis::Zx => {
            drag.origin
                + drag.plane_u * delta.dot(drag.plane_u)
                + drag.plane_v * delta.dot(drag.plane_v)
        }
        GizmoAxis::Uniform => drag.origin,
    };
    let world_delta = world_new_primary - drag.origin;

    for &(bone, start_world, parent_world) in &drag.bones {
        let world_new = start_world + world_delta;
        let local_pos = parent_world.inverse().transform_point3(world_new);
        if let Some(n) = scene.nodes.get_mut(bone.node) {
            n.local.translation = local_pos;
        }
    }
}

#[allow(dead_code)]
pub fn gizmo_screen_center(scene: &Scene, bone: BoneId, viewport: Rect) -> Option<Vec2> {
    let world = scene.world_matrix(bone.node).transform_point3(Vec3::ZERO);
    project_world_to_screen(scene, viewport, world)
}

/// Hit Y=0 ground (or a fallback plane facing camera through origin).
pub fn ray_ground_hit(scene: &Scene, viewport: Rect, cursor: Vec2) -> Option<Vec3> {
    let ray = ray_from_viewport(scene, viewport, cursor)?;
    let denom = ray.dir.y;
    if denom.abs() > 1e-5 {
        let t = -ray.origin.y / denom;
        if t > 0.0 {
            return Some(ray.origin + ray.dir * t);
        }
    }
    // Fallback: plane through origin facing camera.
    let n = (scene.camera.eye).normalize_or_zero();
    ray_plane_point(&ray, Vec3::ZERO, n)
}
