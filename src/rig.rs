//! Rig document: bones + bind pose on top of a mega-render [`Scene`].

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use glam::Vec3;
use mega_render::{
    load_gltf, Camera, DirectionalLight, Handle, Light, LineOpts, Node, PointLight, Scene,
    Transform,
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
pub enum AppMode {
    /// Create / reparent / rest-pose transforms.
    Edit,
    /// FK pose on top of bind (current behaviour).
    #[default]
    Pose,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tool {
    #[default]
    Select,
    /// Edit: place / extrude a child bone.
    AddBone,
    Translate,
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
    pub mode: AppMode,
    pub tool: Tool,
    /// Screen-space rect of the 3D viewport (updated each UI frame).
    pub viewport_rect: mega_ui::Rect,
    /// Counter for default bone names.
    next_bone_serial: u32,
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
            mode: AppMode::Pose,
            tool: Tool::Rotate,
            viewport_rect: mega_ui::Rect {
                min: glam::Vec2::ZERO,
                max: glam::Vec2::ZERO,
            },
            next_bone_serial: 1,
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
        self.next_bone_serial = 1;
    }

    /// Empty scene + root bone (+ tip child so a segment is visible).
    pub fn new_skeleton(&mut self, scene: &mut Scene) {
        self.clear_model(scene);
        let root = self.insert_bone(scene, "Root", None, Transform::default());
        let tip = self.insert_bone(
            scene,
            "Bone",
            Some(root),
            Transform::from_translation(Vec3::Y * 0.35),
        );
        self.selection = Some(tip);
        self.capture_bind_pose(scene);
        self.mode = AppMode::Edit;
        self.tool = Tool::Translate;
        self.fit_camera(scene);
    }

    pub fn set_mode(&mut self, scene: &mut Scene, mode: AppMode) {
        if self.mode == mode {
            return;
        }
        match mode {
            AppMode::Edit => {
                // Edit rest pose, not a temporary FK pose.
                self.reset_pose(scene);
                if !matches!(self.tool, Tool::Translate | Tool::AddBone | Tool::Rotate) {
                    self.tool = Tool::Translate;
                }
            }
            AppMode::Pose => {
                self.capture_bind_pose(scene);
                self.tool = Tool::Rotate;
            }
        }
        self.mode = mode;
    }

    pub fn write_bind_for(&mut self, scene: &Scene, id: BoneId) {
        if let Some(n) = scene.nodes.get(id.node) {
            self.bind_locals.insert(id.node.key(), n.local);
        }
    }

    /// Extrude a child from `parent` along local +Y.
    pub fn extrude_bone(&mut self, scene: &mut Scene, parent: BoneId) -> Option<BoneId> {
        if self.bone(parent).is_none() {
            return None;
        }
        let len = self.average_bone_length(scene).max(0.12);
        let name = format!("Bone_{}", self.next_bone_serial);
        self.next_bone_serial += 1;
        let id = self.insert_bone(
            scene,
            &name,
            Some(parent),
            Transform::from_translation(Vec3::Y * len),
        );
        self.selection = Some(id);
        self.write_bind_for(scene, id);
        Some(id)
    }

    /// New root (or orphan) at a world position.
    pub fn add_root_at(&mut self, scene: &mut Scene, world_pos: Vec3) -> BoneId {
        let name = format!("Bone_{}", self.next_bone_serial);
        self.next_bone_serial += 1;
        let id = self.insert_bone(
            scene,
            &name,
            None,
            Transform::from_translation(world_pos),
        );
        self.selection = Some(id);
        self.write_bind_for(scene, id);
        id
    }

    /// Delete bone and all descendants. Children are not reparented.
    pub fn delete_bone_subtree(&mut self, scene: &mut Scene, id: BoneId) {
        if self.bone(id).is_none() {
            return;
        }
        let mut stack = vec![id];
        let mut kill = Vec::new();
        while let Some(cur) = stack.pop() {
            if let Some(b) = self.bone(cur) {
                stack.extend(b.children.iter().copied());
            }
            kill.push(cur);
        }

        let kill_set: HashSet<(u32, u32)> = kill.iter().map(|b| b.node.key()).collect();

        // Detach from surviving parents' children lists.
        for b in &mut self.bones {
            b.children.retain(|c| !kill_set.contains(&c.node.key()));
        }

        if self.selection.is_some_and(|s| kill_set.contains(&s.node.key())) {
            self.selection = None;
        }

        for dead in &kill {
            self.bind_locals.remove(&dead.node.key());
            self.bone_index.remove(&dead.node.key());
            scene.nodes.remove(dead.node);
        }
        self.bones
            .retain(|b| !kill_set.contains(&b.id.node.key()));
        self.reindex_bones();
    }

    fn insert_bone(
        &mut self,
        scene: &mut Scene,
        name: &str,
        parent: Option<BoneId>,
        local: Transform,
    ) -> BoneId {
        let node = scene.nodes.insert(Node {
            name: name.to_string(),
            parent: parent.map(|p| p.node),
            local,
            mesh: None,
            material: None,
            skin: None,
            visible: true,
        });
        let id = BoneId { node };
        let idx = self.bones.len();
        self.bone_index.insert(node.key(), idx);
        self.bones.push(BoneInfo {
            id,
            name: name.to_string(),
            parent,
            children: Vec::new(),
            deform: false,
        });
        if let Some(p) = parent {
            if let Some(&pi) = self.bone_index.get(&p.node.key()) {
                self.bones[pi].children.push(id);
            }
        }
        id
    }

    fn reindex_bones(&mut self) {
        self.bone_index.clear();
        for (i, b) in self.bones.iter().enumerate() {
            self.bone_index.insert(b.id.node.key(), i);
        }
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
        self.mode = AppMode::Pose;
        self.tool = Tool::Rotate;
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
        scene.camera = Camera::orbit(std::f32::consts::PI, 0.35, dist, center);
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

/// Stick bone display (screen-space px). Later: expose via settings UI.
const BONE_LINE_W: f32 = 3.5;
const BONE_OUTLINE_W: f32 = 7.5;
const BONE_JOINT: f32 = 8.0;
const BONE_JOINT_OUTLINE: f32 = 12.0;

const BONE_FILL: [f32; 4] = [0.86, 0.86, 0.90, 1.0];
const BONE_OUTLINE: [f32; 4] = [0.02, 0.02, 0.04, 1.0];
/// Selected: bright fill + darker blue outline (both blue so selection pops).
const BONE_SEL_FILL: [f32; 4] = [0.35, 0.72, 1.0, 1.0];
const BONE_SEL_OUTLINE: [f32; 4] = [0.06, 0.22, 0.55, 1.0];

/// Stick skeleton: thick outline + thinner fill, joint dots. Overlay so mesh never hides it.
pub fn draw_rig_debug(scene: &mut Scene, rig: &RigDocument) {
    if !rig.show_skeleton {
        return;
    }
    let segments = rig.bone_segments(scene);

    // Outlines first (drawn under fills in submission order).
    for (from, to, id) in &segments {
        let outline = if rig.selection == Some(*id) {
            BONE_SEL_OUTLINE
        } else {
            BONE_OUTLINE
        };
        scene.debug.line(
            *from,
            *to,
            LineOpts::color(outline).width(BONE_OUTLINE_W).overlay(),
        );
    }
    for (from, to, id) in &segments {
        let fill = if rig.selection == Some(*id) {
            BONE_SEL_FILL
        } else {
            BONE_FILL
        };
        scene.debug.line(
            *from,
            *to,
            LineOpts::color(fill).width(BONE_LINE_W).overlay(),
        );
    }

    // Joint dots at bone origins (points render after lines → sit on top).
    for b in &rig.bones {
        let pos = scene.world_matrix(b.id.node).transform_point3(Vec3::ZERO);
        let selected = rig.selection == Some(b.id);
        let (outline, fill) = if selected {
            (BONE_SEL_OUTLINE, BONE_SEL_FILL)
        } else {
            (BONE_OUTLINE, BONE_FILL)
        };
        scene
            .debug
            .point_ex(pos, outline, BONE_JOINT_OUTLINE, false);
        scene.debug.point_ex(pos, fill, BONE_JOINT, false);
    }
}

pub fn empty_scene() -> Scene {
    let mut scene = Scene::new();
    scene.ambient = [0.04, 0.04, 0.05];
    scene.clear_color = [0.0, 0.0, 0.0, 1.0];
    // Z+ = forward: start on −Z looking toward origin / +Z.
    scene.camera = Camera::orbit(std::f32::consts::PI, 0.35, 5.0, Vec3::new(0.0, 0.5, 0.0));
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
