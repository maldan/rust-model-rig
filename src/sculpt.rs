//! Shape-key sculpt: mesh picking + Grab / Inflate / Smooth brushes.

use glam::{Mat4, Vec2, Vec3};
use mega_render::{skin_mesh_matrix, skin_mesh_point, Handle, Mesh, Scene};
use mega_ui::Rect;

use crate::pick::{self, Ray};
use crate::rig::{BrushKind, RigDocument};

pub struct SculptDrag {
    pub mesh: Handle<Mesh>,
    pub node: mega_render::Handle<mega_render::Node>,
    pub shape: usize,
    pub kind: BrushKind,
    /// Vertex indices + falloff (Grab: frozen at stroke start; others refreshed).
    pub verts: Vec<(usize, f32)>,
    /// Neighbor lists for Smooth (built once per stroke).
    pub adjacency: Vec<Vec<usize>>,
    pub last_cursor: Vec2,
    pub plane_normal: Vec3,
    pub plane_point: Vec3,
}

/// Closest ray–mesh hit (skinned if the node has a skin).
pub struct MeshHit {
    pub node: mega_render::Handle<mega_render::Node>,
    pub mesh: Handle<Mesh>,
    pub point: Vec3,
    pub normal: Vec3,
    pub t: f32,
}

pub fn pick_mesh(scene: &Scene, ray: &Ray) -> Option<MeshHit> {
    let world = scene.world_matrices();
    let mut best: Option<MeshHit> = None;

    for (node_h, node) in scene.nodes.iter() {
        if !node.visible {
            continue;
        }
        let Some(mesh_h) = node.mesh else {
            continue;
        };
        let Some(mesh) = scene.meshes.get(mesh_h) else {
            continue;
        };
        if mesh.positions.len() < 3 || mesh.indices.len() < 3 {
            continue;
        }

        let mesh_world = world
            .get(&node_h.key())
            .copied()
            .unwrap_or_else(|| scene.world_matrix(node_h));

        let skin_mats = node
            .skin
            .map(|skin_h| scene.joint_matrices_with_cache(skin_h, node_h, &world))
            .filter(|m| !m.is_empty());
        let mode = scene.skinning_mode;

        let nverts = mesh.positions.len();
        let mut pos_w = vec![Vec3::ZERO; nverts];
        for i in 0..nverts {
            let p = Vec3::from_array(mesh.positions[i]);
            let local = if let Some(ref mats) = skin_mats {
                skin_mesh_point(mesh, mats, i, p, mode)
            } else {
                p
            };
            pos_w[i] = mesh_world.transform_point3(local);
        }

        for tri in mesh.indices.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;
            if i0 >= nverts || i1 >= nverts || i2 >= nverts {
                continue;
            }
            let Some((t, bary)) = ray_triangle(ray, pos_w[i0], pos_w[i1], pos_w[i2]) else {
                continue;
            };
            if t < 0.0 {
                continue;
            }
            if best.as_ref().is_some_and(|b| t >= b.t) {
                continue;
            }
            let n0 = mesh
                .normals
                .get(i0)
                .map(|n| Vec3::from_array(*n))
                .unwrap_or(Vec3::Y);
            let n1 = mesh
                .normals
                .get(i1)
                .map(|n| Vec3::from_array(*n))
                .unwrap_or(Vec3::Y);
            let n2 = mesh
                .normals
                .get(i2)
                .map(|n| Vec3::from_array(*n))
                .unwrap_or(Vec3::Y);
            let n_local = n0 * bary.x + n1 * bary.y + n2 * bary.z;
            let n_world = mesh_world
                .inverse()
                .transpose()
                .transform_vector3(n_local)
                .normalize_or_zero();
            best = Some(MeshHit {
                node: node_h,
                mesh: mesh_h,
                point: ray.origin + ray.dir * t,
                normal: if n_world.length_squared() > 1e-8 {
                    n_world
                } else {
                    (pos_w[i1] - pos_w[i0])
                        .cross(pos_w[i2] - pos_w[i0])
                        .normalize_or_zero()
                },
                t,
            });
        }
    }

    best
}

fn ray_triangle(ray: &Ray, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<(f32, Vec3)> {
    const EPS: f32 = 1e-7;
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let pvec = ray.dir.cross(e2);
    let det = e1.dot(pvec);
    if det.abs() < EPS {
        return None;
    }
    let inv = 1.0 / det;
    let tvec = ray.origin - v0;
    let u = tvec.dot(pvec) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(e1);
    let v = ray.dir.dot(qvec) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(qvec) * inv;
    if t < 0.0 {
        return None;
    }
    Some((t, Vec3::new(1.0 - u - v, u, v)))
}

fn collect_brush_verts(
    scene: &Scene,
    mesh_h: Handle<Mesh>,
    node_h: mega_render::Handle<mega_render::Node>,
    center: Vec3,
    radius: f32,
) -> Vec<(usize, f32)> {
    let Some(mesh) = scene.meshes.get(mesh_h) else {
        return Vec::new();
    };
    let world = scene.world_matrices();
    let mesh_world = world
        .get(&node_h.key())
        .copied()
        .unwrap_or_else(|| scene.world_matrix(node_h));
    let skin_mats = scene
        .nodes
        .get(node_h)
        .and_then(|n| n.skin)
        .map(|skin_h| scene.joint_matrices_with_cache(skin_h, node_h, &world))
        .filter(|m| !m.is_empty());
    let mode = scene.skinning_mode;

    let radius = radius.max(1e-4);
    let nverts = mesh.positions.len();
    let mut verts = Vec::new();
    for i in 0..nverts {
        let p = Vec3::from_array(mesh.positions[i]);
        let local = if let Some(ref mats) = skin_mats {
            skin_mesh_point(mesh, mats, i, p, mode)
        } else {
            p
        };
        let pw = mesh_world.transform_point3(local);
        let d = (pw - center).length();
        if d > radius {
            continue;
        }
        let t = (1.0 - d / radius).clamp(0.0, 1.0);
        let falloff = t * t * (3.0 - 2.0 * t);
        verts.push((i, falloff));
    }
    verts
}

fn build_adjacency(mesh: &Mesh) -> Vec<Vec<usize>> {
    let n = mesh.positions.len();
    let mut adj = vec![Vec::new(); n];
    for tri in mesh.indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        if i0 >= n || i1 >= n || i2 >= n {
            continue;
        }
        for (a, b) in [(i0, i1), (i1, i2), (i2, i0)] {
            if !adj[a].contains(&b) {
                adj[a].push(b);
            }
            if !adj[b].contains(&a) {
                adj[b].push(a);
            }
        }
    }
    adj
}

pub fn begin_stroke(
    scene: &Scene,
    rig: &RigDocument,
    viewport: Rect,
    cursor: Vec2,
) -> Option<SculptDrag> {
    let mesh_h = rig.active_mesh?;
    let shape = rig.active_shape?;
    let ray = pick::ray_from_viewport(scene, viewport, cursor)?;
    let hit = pick_mesh(scene, &ray)?;
    if hit.mesh.key() != mesh_h.key() {
        return None;
    }

    let mesh = scene.meshes.get(hit.mesh)?;
    if shape >= mesh.morph_targets.len() {
        return None;
    }

    let verts = collect_brush_verts(scene, hit.mesh, hit.node, hit.point, rig.brush_radius);
    if verts.is_empty() {
        return None;
    }

    let adjacency = if rig.brush_kind == BrushKind::Smooth {
        build_adjacency(mesh)
    } else {
        Vec::new()
    };

    let plane_normal = (scene.camera.target - scene.camera.eye).normalize_or_zero();
    let plane_normal = if plane_normal.length_squared() < 1e-8 {
        -ray.dir
    } else {
        plane_normal
    };

    Some(SculptDrag {
        mesh: hit.mesh,
        node: hit.node,
        shape,
        kind: rig.brush_kind,
        verts,
        adjacency,
        last_cursor: cursor,
        plane_normal,
        plane_point: hit.point,
    })
}

pub fn apply_stroke(
    scene: &mut Scene,
    drag: &mut SculptDrag,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
    strength: f32,
    invert: bool,
) {
    match drag.kind {
        BrushKind::Grab => apply_grab(scene, drag, viewport, cursor, strength),
        BrushKind::Inflate => {
            apply_inflate(scene, drag, viewport, cursor, radius, strength, invert)
        }
        BrushKind::Smooth => apply_smooth(scene, drag, viewport, cursor, radius, strength),
    }
}

fn apply_grab(
    scene: &mut Scene,
    drag: &mut SculptDrag,
    viewport: Rect,
    cursor: Vec2,
    strength: f32,
) {
    if (cursor - drag.last_cursor).length_squared() < 1e-8 {
        return;
    }
    let Some(prev_ray) = pick::ray_from_viewport(scene, viewport, drag.last_cursor) else {
        drag.last_cursor = cursor;
        return;
    };
    let Some(cur_ray) = pick::ray_from_viewport(scene, viewport, cursor) else {
        drag.last_cursor = cursor;
        return;
    };
    let Some(prev_hit) = ray_plane(&prev_ray, drag.plane_point, drag.plane_normal) else {
        drag.last_cursor = cursor;
        return;
    };
    let Some(cur_hit) = ray_plane(&cur_ray, drag.plane_point, drag.plane_normal) else {
        drag.last_cursor = cursor;
        return;
    };
    let world_delta = (cur_hit - prev_hit) * strength.clamp(0.0, 1.0);
    drag.last_cursor = cursor;
    if world_delta.length_squared() < 1e-14 {
        return;
    }

    let world = scene.world_matrices();
    let mesh_world = world
        .get(&drag.node.key())
        .copied()
        .unwrap_or_else(|| scene.world_matrix(drag.node));
    let skin_mats = scene
        .nodes
        .get(drag.node)
        .and_then(|n| n.skin)
        .map(|skin_h| scene.joint_matrices_with_cache(skin_h, drag.node, &world))
        .filter(|m| !m.is_empty());
    let mode = scene.skinning_mode;

    let deltas: Vec<(usize, [f32; 3])> = {
        let Some(mesh) = scene.meshes.get(drag.mesh) else {
            return;
        };
        drag.verts
            .iter()
            .filter_map(|&(vi, falloff)| {
                let skin = if let Some(ref mats) = skin_mats {
                    skin_mesh_matrix(mesh, mats, vi, mode)
                } else {
                    Mat4::IDENTITY
                };
                let t = mesh_world * skin;
                let inv = t.inverse();
                let local = inv.transform_vector3(world_delta * falloff);
                if local.length_squared() < 1e-16 {
                    None
                } else {
                    Some((vi, local.to_array()))
                }
            })
            .collect()
    };

    let Some(mesh) = scene.meshes.get_mut(drag.mesh) else {
        return;
    };
    for (vi, d) in deltas {
        mesh.add_shape_delta(drag.shape, vi, d);
    }
    mesh.apply_morphs();
}

fn refresh_brush_under_cursor(
    scene: &Scene,
    drag: &mut SculptDrag,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
) {
    drag.last_cursor = cursor;
    let Some(ray) = pick::ray_from_viewport(scene, viewport, cursor) else {
        return;
    };
    let Some(hit) = pick_mesh(scene, &ray) else {
        return;
    };
    if hit.mesh.key() != drag.mesh.key() {
        return;
    }
    drag.plane_point = hit.point;
    drag.verts = collect_brush_verts(scene, drag.mesh, drag.node, hit.point, radius);
}

fn apply_inflate(
    scene: &mut Scene,
    drag: &mut SculptDrag,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
    strength: f32,
    invert: bool,
) {
    refresh_brush_under_cursor(scene, drag, viewport, cursor, radius);
    if drag.verts.is_empty() {
        return;
    }

    let sign = if invert { -1.0 } else { 1.0 };
    // World-ish step scaled by brush size so strength feels consistent across scales.
    let step = radius.max(1e-4) * 0.012 * strength.clamp(0.0, 1.0) * sign;

    let deltas: Vec<(usize, [f32; 3])> = {
        let Some(mesh) = scene.meshes.get(drag.mesh) else {
            return;
        };
        drag.verts
            .iter()
            .filter_map(|&(vi, falloff)| {
                let n = mesh
                    .normals
                    .get(vi)
                    .map(|n| Vec3::from_array(*n))
                    .unwrap_or(Vec3::Y)
                    .normalize_or_zero();
                if n.length_squared() < 1e-10 {
                    return None;
                }
                let local = n * (step * falloff);
                Some((vi, local.to_array()))
            })
            .collect()
    };

    let Some(mesh) = scene.meshes.get_mut(drag.mesh) else {
        return;
    };
    for (vi, d) in deltas {
        mesh.add_shape_delta(drag.shape, vi, d);
    }
    mesh.apply_morphs();
}

fn apply_smooth(
    scene: &mut Scene,
    drag: &mut SculptDrag,
    viewport: Rect,
    cursor: Vec2,
    radius: f32,
    strength: f32,
) {
    refresh_brush_under_cursor(scene, drag, viewport, cursor, radius);
    if drag.verts.is_empty() {
        return;
    }
    if drag.adjacency.is_empty() {
        if let Some(mesh) = scene.meshes.get(drag.mesh) {
            drag.adjacency = build_adjacency(mesh);
        }
    }

    let amount = 0.15 * strength.clamp(0.0, 1.0);
    let deltas: Vec<(usize, [f32; 3])> = {
        let Some(mesh) = scene.meshes.get(drag.mesh) else {
            return;
        };
        drag.verts
            .iter()
            .filter_map(|&(vi, falloff)| {
                let neighbors = drag.adjacency.get(vi)?;
                if neighbors.is_empty() {
                    return None;
                }
                let mut avg = Vec3::ZERO;
                let mut count = 0u32;
                for &ni in neighbors {
                    if let Some(p) = mesh.positions.get(ni) {
                        avg += Vec3::from_array(*p);
                        count += 1;
                    }
                }
                if count == 0 {
                    return None;
                }
                avg /= count as f32;
                let cur = Vec3::from_array(mesh.positions[vi]);
                let local = (avg - cur) * (amount * falloff);
                if local.length_squared() < 1e-16 {
                    None
                } else {
                    Some((vi, local.to_array()))
                }
            })
            .collect()
    };

    let Some(mesh) = scene.meshes.get_mut(drag.mesh) else {
        return;
    };
    for (vi, d) in deltas {
        mesh.add_shape_delta(drag.shape, vi, d);
    }
    mesh.apply_morphs();
}

fn ray_plane(ray: &Ray, point: Vec3, normal: Vec3) -> Option<Vec3> {
    let denom = ray.dir.dot(normal);
    if denom.abs() < 1e-8 {
        return None;
    }
    let t = (point - ray.origin).dot(normal) / denom;
    if t < 0.0 {
        return None;
    }
    Some(ray.origin + ray.dir * t)
}

/// Draw a simple brush ring at the surface under the cursor.
pub fn draw_brush_cursor(scene: &mut Scene, viewport: Rect, cursor: Vec2, radius: f32) {
    let Some(ray) = pick::ray_from_viewport(scene, viewport, cursor) else {
        return;
    };
    let Some(hit) = pick_mesh(scene, &ray) else {
        return;
    };
    let n = hit.normal.normalize_or_zero();
    if n.length_squared() < 1e-8 {
        return;
    }
    let (u, v) = ring_basis(n);
    const SEGS: usize = 48;
    let r = radius.max(1e-4);
    let mut prev = hit.point + u * r;
    for i in 1..=SEGS {
        let a = (i as f32 / SEGS as f32) * std::f32::consts::TAU;
        let p = hit.point + (u * a.cos() + v * a.sin()) * r;
        scene.debug.line(
            prev,
            p,
            mega_render::LineOpts::color([0.35, 0.85, 1.0, 0.9])
                .width(1.5)
                .overlay(),
        );
        prev = p;
    }
}

fn ring_basis(n: Vec3) -> (Vec3, Vec3) {
    let helper = if n.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let u = n.cross(helper).normalize_or_zero();
    let v = n.cross(u).normalize_or_zero();
    (u, v)
}
