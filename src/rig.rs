//! Rig document: bones + bind pose on top of a mega-render [`Scene`].

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use glam::{Mat4, Vec3};
use mega_render::{
    load_gltf, Camera, DirectionalLight, Handle, Light, Node, PointLight, Scene, Transform,
};

#[derive(Clone, Copy)]
pub struct BoneId {
    pub node: Handle<Node>,
}

impl std::fmt::Debug for BoneId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BoneId({}:{})",
            self.node.index, self.node.generation
        )
    }
}

impl PartialEq for BoneId {
    fn eq(&self, other: &Self) -> bool {
        self.node.key() == other.node.key()
    }
}

impl Eq for BoneId {}

impl Hash for BoneId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.key().hash(state);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tool {
    #[default]
    Select,
    Rotate,
}

pub struct BoneInfo {
    pub id: BoneId,
    pub name: String,
    pub parent: Option<BoneId>,
    pub children: Vec<BoneId>,
    /// Joint participates in skin deformation.
    pub deform: bool,
}

pub struct RigDocument {
    pub source_path: Option<PathBuf>,
    pub model_root: Option<Handle<Node>>,
    pub bones: Vec<BoneInfo>,
    pub bone_index: HashMap<(u32, u32), usize>,
    pub selection: Option<BoneId>,
    /// Local transforms at bind (load) time.
    pub bind_locals: HashMap<(u32, u32), Transform>,
    pub show_skeleton: bool,
    pub show_mesh: bool,
    pub tool: Tool,
    /// Screen-space rect of the 3D viewport (updated each UI frame).
    pub viewport_rect: mega_ui::Rect,
}

impl Default for RigDocument {
    fn default() -> Self {
        Self {
            source_path: None,
            model_root: None,
            bones: Vec::new(),
            bone_index: HashMap::new(),
            selection: None,
            bind_locals: HashMap::new(),
            show_skeleton: true,
            show_mesh: true,
            tool: Tool::Rotate,
            viewport_rect: mega_ui::Rect {
                min: glam::Vec2::ZERO,
                max: glam::Vec2::ZERO,
            },
        }
    }
}

impl RigDocument {
    pub fn clear_model(&mut self, scene: &mut Scene) {
        *scene = empty_scene();

        self.source_path = None;
        self.model_root = None;
        self.bones.clear();
        self.bone_index.clear();
        self.selection = None;
        self.bind_locals.clear();
    }

    pub fn load_path(&mut self, scene: &mut Scene, path: &Path) -> Result<(), String> {
        self.clear_model(scene);
        let root = load_gltf(scene, path, None)?;
        for a in &mut scene.animators {
            a.playing = false;
        }
        scene.animators.clear();
        self.source_path = Some(path.to_path_buf());
        self.model_root = Some(root);
        self.rebuild_bones(scene);
        Self::relink_joint_parents(scene, &self.bones);
        // Parents changed — refresh bind from the relinked locals.
        self.capture_bind_pose(scene);
        self.fit_camera(scene);
        Ok(())
    }

    /// Make each joint's `Node.parent` the logical joint parent (skip empties/ORG).
    /// Bakes world transform so the pose stays identical.
    fn relink_joint_parents(scene: &mut Scene, bones: &[BoneInfo]) {
        let mut order: Vec<&BoneInfo> = bones.iter().collect();
        order.sort_by_key(|b| {
            let mut d = 0u32;
            let mut p = b.parent;
            while let Some(pp) = p {
                d += 1;
                p = bones
                    .iter()
                    .find(|x| x.id == pp)
                    .and_then(|x| x.parent);
            }
            d
        });

        for b in order {
            let Some(lp) = b.parent else {
                continue;
            };
            let cur_parent = scene.nodes.get(b.id.node).and_then(|n| n.parent);
            if cur_parent.is_some_and(|p| p.key() == lp.node.key()) {
                continue;
            }
            let world = scene.world_matrix(b.id.node);
            let parent_world = scene.world_matrix(lp.node);
            let local_mat = parent_world.inverse() * world;
            let (scale, rotation, translation) = local_mat.to_scale_rotation_translation();
            if let Some(n) = scene.nodes.get_mut(b.id.node) {
                n.parent = Some(lp.node);
                n.local = Transform {
                    translation,
                    rotation,
                    scale,
                };
            }
        }
    }

    pub fn rebuild_bones(&mut self, scene: &Scene) {
        self.bones.clear();
        self.bone_index.clear();

        let mut joint_set: HashSet<(u32, u32)> = HashSet::new();
        for (_, skin) in scene.skins.iter() {
            for &j in &skin.joints {
                joint_set.insert(j.key());
            }
        }

        let mut candidates: Vec<Handle<Node>> = if joint_set.is_empty() {
            scene.nodes.iter().map(|(h, _)| h).collect()
        } else {
            joint_set
                .iter()
                .filter_map(|&key| {
                    scene
                        .nodes
                        .iter()
                        .find(|(h, _)| h.key() == key)
                        .map(|(h, _)| h)
                })
                .collect()
        };
        candidates.sort_by_key(|h| h.index);

        for &node in &candidates {
            let Some(n) = scene.nodes.get(node) else {
                continue;
            };
            let parent = n.parent.and_then(|p| {
                if joint_set.is_empty() || joint_set.contains(&p.key()) {
                    Some(BoneId { node: p })
                } else {
                    let mut cur = Some(p);
                    while let Some(c) = cur {
                        if joint_set.contains(&c.key()) {
                            return Some(BoneId { node: c });
                        }
                        cur = scene.nodes.get(c).and_then(|nn| nn.parent);
                    }
                    None
                }
            });
            let id = BoneId { node };
            let idx = self.bones.len();
            self.bone_index.insert(node.key(), idx);
            self.bones.push(BoneInfo {
                id,
                name: if n.name.is_empty() {
                    format!("Bone_{}", node.index)
                } else {
                    n.name.clone()
                },
                parent,
                children: Vec::new(),
                deform: !joint_set.is_empty() && joint_set.contains(&node.key()),
            });
        }

        let parents: Vec<(BoneId, Option<BoneId>)> =
            self.bones.iter().map(|b| (b.id, b.parent)).collect();
        for (id, parent) in parents {
            if let Some(p) = parent {
                if let Some(&pi) = self.bone_index.get(&p.node.key()) {
                    self.bones[pi].children.push(id);
                }
            }
        }

        if let Some(sel) = self.selection {
            if !self.bone_index.contains_key(&sel.node.key()) {
                self.selection = None;
            }
        }
    }

    pub fn capture_bind_pose(&mut self, scene: &Scene) {
        self.bind_locals.clear();
        for b in &self.bones {
            if let Some(n) = scene.nodes.get(b.id.node) {
                self.bind_locals.insert(b.id.node.key(), n.local);
            }
        }
    }

    pub fn reset_pose(&self, scene: &mut Scene) {
        for b in &self.bones {
            if let Some(local) = self.bind_locals.get(&b.id.node.key()).copied() {
                if let Some(n) = scene.nodes.get_mut(b.id.node) {
                    n.local = local;
                }
            }
        }
    }

    pub fn reset_selected(&self, scene: &mut Scene) {
        let Some(sel) = self.selection else {
            return;
        };
        let Some(local) = self.bind_locals.get(&sel.node.key()).copied() else {
            return;
        };
        if let Some(n) = scene.nodes.get_mut(sel.node) {
            n.local = local;
        }
    }

    pub fn bone(&self, id: BoneId) -> Option<&BoneInfo> {
        self.bone_index
            .get(&id.node.key())
            .and_then(|&i| self.bones.get(i))
    }

    pub fn roots(&self) -> Vec<BoneId> {
        self.bones
            .iter()
            .filter(|b| b.parent.is_none())
            .map(|b| b.id)
            .collect()
    }

    pub fn apply_mesh_visibility(&self, scene: &mut Scene) {
        for b_node in scene.nodes.iter().map(|(h, _)| h).collect::<Vec<_>>() {
            if let Some(n) = scene.nodes.get_mut(b_node) {
                if n.mesh.is_some() {
                    n.visible = self.show_mesh;
                }
            }
        }
    }

    fn fit_camera(&self, scene: &mut Scene) {
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        let mut any = false;

        let mesh_nodes: Vec<_> = scene
            .nodes
            .iter()
            .filter_map(|(h, n)| n.mesh.map(|m| (h, m)))
            .collect();
        for (h, mh) in mesh_nodes {
            let Some(mesh) = scene.meshes.get(mh) else {
                continue;
            };
            let world = scene.world_matrix(h);
            for p in &mesh.positions {
                let wp = world.transform_point3(Vec3::from_array(*p));
                min = min.min(wp);
                max = max.max(wp);
                any = true;
            }
        }
        for b in &self.bones {
            let wp = scene.world_matrix(b.id.node).transform_point3(Vec3::ZERO);
            min = min.min(wp);
            max = max.max(wp);
            any = true;
        }
        if !any {
            return;
        }
        let center = (min + max) * 0.5;
        let radius = ((max - min).length() * 0.5).max(0.25);
        let dist = (radius / (45f32.to_radians() * 0.5).tan()).max(1.0) * 1.35;
        scene.camera = Camera::orbit(0.9, 0.35, dist, center);
        scene.camera.far = (dist + radius * 4.0).max(50.0);
        scene.camera.near = (dist * 0.001).clamp(0.01, 0.1);
    }

    /// Bone diamonds: from this joint → each child (outgoing).
    /// Rotating a bone swings *its* diamonds; old parent→self drawing looked
    /// like "only children move" because that segment's endpoints stay fixed.
    pub fn bone_segments(&self, scene: &Scene) -> Vec<(Vec3, Vec3, BoneId)> {
        let mut out = Vec::new();
        let fallback_len = {
            let mut sum = 0.0f32;
            let mut n = 0u32;
            for b in &self.bones {
                if let Some(p) = b.parent {
                    let a = scene.world_matrix(p.node).transform_point3(Vec3::ZERO);
                    let c = scene.world_matrix(b.id.node).transform_point3(Vec3::ZERO);
                    let d = (c - a).length();
                    if d > 1e-5 {
                        sum += d;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                (sum / n as f32).max(0.05)
            } else {
                0.12
            }
        };

        for b in &self.bones {
            let from = scene.world_matrix(b.id.node).transform_point3(Vec3::ZERO);
            if b.children.is_empty() {
                let m = scene.world_matrix(b.id.node);
                let axis = m.transform_vector3(Vec3::Y).normalize_or_zero();
                if axis.length_squared() > 1e-8 {
                    out.push((from, from + axis * fallback_len * 0.65, b.id));
                }
            } else {
                for &child in &b.children {
                    let to = scene.world_matrix(child.node).transform_point3(Vec3::ZERO);
                    if (to - from).length_squared() > 1e-10 {
                        out.push((from, to, b.id));
                    }
                }
            }
        }
        out
    }

    pub fn average_bone_length(&self, scene: &Scene) -> f32 {
        let segs = self.bone_segments(scene);
        if segs.is_empty() {
            return 0.15;
        }
        let sum: f32 = segs.iter().map(|(a, b, _)| (*b - *a).length()).sum();
        (sum / segs.len() as f32).max(0.05)
    }
}

/// Draw skeleton with selection highlight (always overlay so mesh doesn't hide it).
pub fn draw_rig_debug(scene: &mut Scene, rig: &RigDocument) {
    if !rig.show_skeleton {
        return;
    }
    let base = [0.25, 0.95, 0.35, 1.0];
    let sel_col = [0.15, 0.95, 1.0, 1.0];
    let avg = rig.average_bone_length(scene);
    let thickness = (avg * 0.08).clamp(0.008, 0.04);
    let joint = thickness * 1.35;
    let segments = rig.bone_segments(scene);
    for (from, to, id) in &segments {
        let color = if rig.selection == Some(*id) {
            sel_col
        } else {
            base
        };
        if let Some(m) = bone_cuboid_matrix(*from, *to, thickness) {
            scene.debug.box_transform(m, color, false);
        }
        // Small joint cube at the bone origin.
        scene.debug.box_aabb(
            *from - Vec3::splat(joint * 0.5),
            *from + Vec3::splat(joint * 0.5),
            color,
            false,
        );
    }
}

/// Unit box `[-0.5, 0.5]³` → thin cuboid from `from` to `to` (Y along the bone).
fn bone_cuboid_matrix(from: Vec3, to: Vec3, thickness: f32) -> Option<Mat4> {
    let dir = to - from;
    let len = dir.length();
    if len < 1e-6 {
        return None;
    }
    let axis = dir / len;
    let hint = if axis.cross(Vec3::Y).length_squared() < 1e-4 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let side = axis.cross(hint).normalize();
    let up = side.cross(axis).normalize();
    let mid = (from + to) * 0.5;
    Some(Mat4::from_cols(
        (side * thickness).extend(0.0),
        (axis * len).extend(0.0),
        (up * thickness).extend(0.0),
        mid.extend(1.0),
    ))
}

pub fn empty_scene() -> Scene {
    let mut scene = Scene::new();
    scene.ambient = [0.04, 0.04, 0.05];
    scene.clear_color = [0.0, 0.0, 0.0, 1.0];
    if let Some(Light::Directional(d)) = scene.lights.first_mut() {
        *d = DirectionalLight {
            direction: Vec3::new(0.35, -0.55, 0.75).normalize(),
            color: [1.0, 0.98, 0.94],
            intensity: 2.8,
            enabled: true,
            cast_shadows: true,
        };
    }
    scene.lights.push(Light::Point(PointLight {
        position: Vec3::new(-2.0, 2.5, 1.5),
        color: [0.55, 0.7, 1.0],
        intensity: 4.0,
        range: 12.0,
        enabled: true,
    }));
    scene
}
