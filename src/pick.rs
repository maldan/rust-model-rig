//! Viewport ray helpers and bone picking.

use glam::{Mat4, Vec2, Vec3, Vec4};
use mega_render::Scene;
use mega_ui::Rect;

use crate::rig::{BoneId, RigDocument};

pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

/// Build a world ray from a cursor position inside the viewport rect.
pub fn ray_from_viewport(scene: &Scene, viewport: Rect, cursor: Vec2) -> Option<Ray> {
    let w = viewport.width();
    let h = viewport.height();
    if w < 1.0 || h < 1.0 {
        return None;
    }
    if !viewport.contains(cursor) {
        return None;
    }
    let u = ((cursor.x - viewport.min.x) / w) * 2.0 - 1.0;
    let v = 1.0 - ((cursor.y - viewport.min.y) / h) * 2.0;
    let aspect = w / h;
    let view_proj = scene.camera.view_proj(aspect);
    let inv = view_proj.inverse();

    let near = unproject(inv, Vec3::new(u, v, 0.0));
    let far = unproject(inv, Vec3::new(u, v, 1.0));
    let dir = (far - near).normalize_or_zero();
    if dir.length_squared() < 1e-8 {
        return None;
    }
    Some(Ray {
        origin: near,
        dir,
    })
}

fn unproject(inv_view_proj: Mat4, ndc: Vec3) -> Vec3 {
    let clip = Vec4::new(ndc.x, ndc.y, ndc.z, 1.0);
    let world = inv_view_proj * clip;
    if world.w.abs() < 1e-8 {
        return world.truncate();
    }
    world.truncate() / world.w
}

/// Pick closest bone segment / joint sphere to the ray.
pub fn pick_bone(scene: &Scene, rig: &RigDocument, ray: &Ray) -> Option<BoneId> {
    let segs = rig.bone_segments(scene);
    if segs.is_empty() {
        return None;
    }
    let avg = segs
        .iter()
        .map(|(a, b, _)| (*b - *a).length())
        .sum::<f32>()
        / segs.len() as f32;
    let radius = (avg * 0.14).clamp(0.01, 0.08);

    let mut best: Option<(f32, BoneId)> = None;
    for (from, to, id) in segs {
        if let Some(t) = ray_sphere(ray, to, radius) {
            consider(&mut best, t, id);
        }
        if let Some((t, dist)) = ray_segment_hit(ray, from, to) {
            if dist <= radius * 1.5 {
                consider(&mut best, t, id);
            }
        }
    }
    best.map(|(_, id)| id)
}

fn consider(best: &mut Option<(f32, BoneId)>, t: f32, id: BoneId) {
    if t < 0.0 {
        return;
    }
    if best.is_none_or(|(bt, _)| t < bt) {
        *best = Some((t, id));
    }
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

fn ray_segment_hit(ray: &Ray, a: Vec3, b: Vec3) -> Option<(f32, f32)> {
    let ab = b - a;
    let ao = a - ray.origin;
    let ab_len_sq = ab.length_squared();
    if ab_len_sq < 1e-12 {
        let t = ao.dot(ray.dir).max(0.0);
        let p = ray.origin + ray.dir * t;
        return Some((t, (p - a).length()));
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
    Some((t, (p_ray - p_seg).length()))
}

pub fn project_world_to_screen(scene: &Scene, viewport: Rect, world: Vec3) -> Option<Vec2> {
    let w = viewport.width();
    let h = viewport.height();
    if w < 1.0 || h < 1.0 {
        return None;
    }
    let aspect = w / h;
    let clip = scene.camera.view_proj(aspect) * world.extend(1.0);
    if clip.w.abs() < 1e-8 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if ndc.z < 0.0 || ndc.z > 1.0 {
        return None;
    }
    let x = viewport.min.x + (ndc.x * 0.5 + 0.5) * w;
    let y = viewport.min.y + (1.0 - (ndc.y * 0.5 + 0.5)) * h;
    Some(Vec2::new(x, y))
}
