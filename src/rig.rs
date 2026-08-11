//! Rig document: bones + bind pose on top of a mega-render [`Scene`].

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use glam::Vec3;
use mega_render::{
    load_gltf, Camera, DirectionalLight, Handle, Light, LineOpts, Mesh, Node, PointLight, PolyOpts,
    Scene, Skin, Transform,
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

#[derive(Clone)]
struct WeightOverlayCache {
    dirty: bool,
    tris: Vec<(Vec3, Vec3, Vec3, [f32; 4])>,
}

impl Default for WeightOverlayCache {
    fn default() -> Self {
        Self {
            dirty: true,
            tris: Vec::new(),
        }
    }
}

impl WeightOverlayCache {
    fn clear(&mut self) {
        self.tris.clear();
        self.dirty = true;
    }
}

pub struct RigDocument {
    pub source_path: Option<PathBuf>,
    pub model_root: Option<Handle<Node>>,
    pub bones: Vec<BoneInfo>,
    pub bone_index: HashMap<(u32, u32), usize>,
    /// Active bone (gizmo / inspector). Always ∈ `selected` when `Some`.
    pub selection: Option<BoneId>,
    /// Multi-selection set.
    pub selected: HashSet<BoneId>,
    /// Local transforms at bind (load) time.
    pub bind_locals: HashMap<(u32, u32), Transform>,
    pub show_skeleton: bool,
    pub show_mesh: bool,
    /// Heat-map overlay for selected bone influence on skinned meshes.
    pub show_weights: bool,
    /// Cached weight tris; rebuilt only when dirty (selection / pose / load).
    weight_overlay: WeightOverlayCache,
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
            selected: HashSet::new(),
            bind_locals: HashMap::new(),
            show_skeleton: true,
            show_mesh: true,
            show_weights: false,
            weight_overlay: WeightOverlayCache::default(),
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
    /// Mark weight overlay for rebuild (selection / pose / mesh change).
    pub fn invalidate_weight_overlay(&mut self) {
        self.weight_overlay.dirty = true;
    }

    pub fn clear_selection(&mut self) {
        self.selected.clear();
        self.selection = None;
        self.invalidate_weight_overlay();
    }

    pub fn set_selection(&mut self, id: BoneId) {
        self.selected.clear();
        self.selected.insert(id);
        self.selection = Some(id);
        self.invalidate_weight_overlay();
    }

    pub fn toggle_selection(&mut self, id: BoneId) {
        if self.selected.contains(&id) {
            self.selected.remove(&id);
            if self.selection == Some(id) {
                self.selection = self.selected.iter().copied().next();
            }
        } else {
            self.selected.insert(id);
            self.selection = Some(id);
        }
        self.invalidate_weight_overlay();
    }

    /// Replace or union selection with `ids`. Active = last id (if any).
    pub fn select_bones(&mut self, ids: &[BoneId], additive: bool) {
        if !additive {
            self.selected.clear();
            self.selection = None;
        }
        for &id in ids {
            self.selected.insert(id);
        }
        if let Some(&last) = ids.last() {
            self.selection = Some(last);
        }
        self.invalidate_weight_overlay();
    }

    pub fn is_selected(&self, id: BoneId) -> bool {
        self.selected.contains(&id)
    }

    pub fn selected_bones(&self) -> Vec<BoneId> {
        let mut out: Vec<_> = self.selected.iter().copied().collect();
        // Active first, then by node index.
        if let Some(active) = self.selection {
            out.sort_by_key(|id| (*id != active, id.node.index, id.node.generation));
        } else {
            out.sort_by_key(|id| (id.node.index, id.node.generation));
        }
        out
    }

    /// Bones in the selection whose parent is not selected (Blender-style transform targets).
    /// Children stay selected for highlight but follow via FK.
    pub fn selection_roots(&self) -> Vec<BoneId> {
        let mut roots: Vec<_> = self
            .selected
            .iter()
            .copied()
            .filter(|id| match self.bone(*id).and_then(|b| b.parent) {
                Some(p) => !self.selected.contains(&p),
                None => true,
            })
            .collect();
        if let Some(active) = self.selection {
            roots.sort_by_key(|id| (*id != active, id.node.index, id.node.generation));
        } else {
            roots.sort_by_key(|id| (id.node.index, id.node.generation));
        }
        roots
    }

    /// Median of selection-root joint positions (shared pivot for multi-root transforms).
    pub fn selection_pivot(&self, scene: &Scene) -> Option<Vec3> {
        let roots = self.selection_roots();
        if roots.is_empty() {
            return None;
        }
        let mut sum = Vec3::ZERO;
        for id in &roots {
            sum += scene.world_matrix(id.node).transform_point3(Vec3::ZERO);
        }
        Some(sum / roots.len() as f32)
    }

    pub fn clear_model(&mut self, scene: &mut Scene) {
        *scene = empty_scene();

        self.source_path = None;
        self.model_root = None;
        self.bones.clear();
        self.bone_index.clear();
        self.clear_selection();
        self.bind_locals.clear();
        self.next_bone_serial = 1;
        self.weight_overlay.clear();
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
        self.set_selection(tip);
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
        self.set_selection(id);
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
        self.set_selection(id);
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

        self.selected
            .retain(|s| !kill_set.contains(&s.node.key()));
        if self
            .selection
            .is_some_and(|s| kill_set.contains(&s.node.key()))
        {
            self.selection = self.selected.iter().copied().next();
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

        self.selected
            .retain(|s| self.bone_index.contains_key(&s.node.key()));
        if let Some(sel) = self.selection {
            if !self.bone_index.contains_key(&sel.node.key()) {
                self.selection = self.selected.iter().copied().next();
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

    pub fn reset_pose(&mut self, scene: &mut Scene) {
        for b in &self.bones {
            if let Some(local) = self.bind_locals.get(&b.id.node.key()).copied() {
                if let Some(n) = scene.nodes.get_mut(b.id.node) {
                    n.local = local;
                }
            }
        }
        self.invalidate_weight_overlay();
    }

    pub fn reset_selected(&mut self, scene: &mut Scene) {
        for id in &self.selected {
            let Some(local) = self.bind_locals.get(&id.node.key()).copied() else {
                continue;
            };
            if let Some(n) = scene.nodes.get_mut(id.node) {
                n.local = local;
            }
        }
        self.invalidate_weight_overlay();
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

/// Blender-ish weight ramp: blue → cyan → green → yellow → red.
fn weight_heat(w: f32) -> [f32; 4] {
    let t = w.clamp(0.0, 1.0);
    const STOPS: [(f32, [f32; 3]); 5] = [
        (0.0, [0.05, 0.15, 0.95]),
        (0.25, [0.05, 0.75, 0.85]),
        (0.5, [0.15, 0.85, 0.12]),
        (0.75, [0.95, 0.85, 0.08]),
        (1.0, [0.95, 0.12, 0.08]),
    ];
    let mut i = 0;
    while i + 1 < STOPS.len() && t > STOPS[i + 1].0 {
        i += 1;
    }
    let (t0, c0) = STOPS[i];
    let (t1, c1) = STOPS[(i + 1).min(STOPS.len() - 1)];
    let u = if (t1 - t0).abs() < 1e-6 {
        0.0
    } else {
        ((t - t0) / (t1 - t0)).clamp(0.0, 1.0)
    };
    [
        c0[0] + (c1[0] - c0[0]) * u,
        c0[1] + (c1[1] - c0[1]) * u,
        c0[2] + (c1[2] - c0[2]) * u,
        0.95,
    ]
}

const WEIGHT_EPS: f32 = 0.001;
/// Push overlay slightly along normals to avoid z-fight with the mesh.
const WEIGHT_ZBIAS_FRAC: f32 = 0.012;

fn rebuild_weight_overlay(scene: &Scene, rig: &RigDocument) -> Vec<(Vec3, Vec3, Vec3, [f32; 4])> {
    let selected_keys: HashSet<(u32, u32)> =
        rig.selected.iter().map(|b| b.node.key()).collect();
    let zbias = (rig.average_bone_length(scene) * WEIGHT_ZBIAS_FRAC).clamp(0.0004, 0.02);

    let skinned: Vec<(Handle<Node>, Handle<Mesh>, Handle<Skin>)> = scene
        .nodes
        .iter()
        .filter_map(|(h, n)| match (n.mesh, n.skin) {
            (Some(m), Some(s)) => Some((h, m, s)),
            _ => None,
        })
        .collect();
    if skinned.is_empty() {
        return Vec::new();
    }

    let world = scene.world_matrices();
    let mut out_tris: Vec<(Vec3, Vec3, Vec3, [f32; 4])> = Vec::new();

    for (node_h, mesh_h, skin_h) in skinned {
        let joint_keys: Vec<(u32, u32)> = {
            let Some(skin) = scene.skins.get(skin_h) else {
                continue;
            };
            skin.joints.iter().map(|j| j.key()).collect()
        };
        let mut joint_sel = vec![false; joint_keys.len()];
        let mut any = false;
        for (i, key) in joint_keys.iter().enumerate() {
            if selected_keys.contains(key) {
                joint_sel[i] = true;
                any = true;
            }
        }
        if !any {
            continue;
        }

        let mats = scene.joint_matrices_with_cache(skin_h, node_h, &world);
        if mats.is_empty() {
            continue;
        }
        let mesh_world = world
            .get(&node_h.key())
            .copied()
            .unwrap_or_else(|| scene.world_matrix(node_h));
        let normal_world = mesh_world.inverse().transpose();

        let Some(mesh) = scene.meshes.get(mesh_h) else {
            continue;
        };
        let (Some(joints), Some(weights)) = (&mesh.joints, &mesh.weights) else {
            continue;
        };
        let nverts = mesh
            .positions
            .len()
            .min(joints.len())
            .min(weights.len());
        if nverts == 0 || mesh.indices.len() < 3 {
            continue;
        }

        let mut w_sel = vec![0.0f32; nverts];
        for i in 0..nverts {
            let ji = joints[i];
            let wi = weights[i];
            let mut w_max = 0.0f32;
            for k in 0..4 {
                let idx = ji[k] as usize;
                if idx < joint_sel.len() && joint_sel[idx] {
                    w_max = w_max.max(wi[k]);
                }
            }
            w_sel[i] = w_max;
        }

        let mut need = vec![false; nverts];
        let mut influenced: Vec<[usize; 3]> = Vec::new();
        for tri in mesh.indices.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;
            if i0 >= nverts || i1 >= nverts || i2 >= nverts {
                continue;
            }
            let w = w_sel[i0].max(w_sel[i1]).max(w_sel[i2]);
            if w < WEIGHT_EPS {
                continue;
            }
            need[i0] = true;
            need[i1] = true;
            need[i2] = true;
            influenced.push([i0, i1, i2]);
        }
        if influenced.is_empty() {
            continue;
        }

        let mut pos_w = vec![Vec3::ZERO; nverts];
        let mut nrm_w = vec![Vec3::Y; nverts];
        for i in 0..nverts {
            if !need[i] {
                continue;
            }
            let ji = joints[i];
            let wi = weights[i];
            let p = Vec3::from_array(mesh.positions[i]);
            let n = mesh
                .normals
                .get(i)
                .map(|n| Vec3::from_array(*n))
                .unwrap_or(Vec3::Y);
            let mut skinned_p = Vec3::ZERO;
            let mut skinned_n = Vec3::ZERO;
            let mut w_sum = 0.0f32;
            for k in 0..4 {
                let idx = ji[k] as usize;
                let w = wi[k];
                if w <= 0.0 || idx >= mats.len() {
                    continue;
                }
                skinned_p += mats[idx].transform_point3(p) * w;
                skinned_n += mats[idx].transform_vector3(n) * w;
                w_sum += w;
            }
            if w_sum < 1e-6 {
                skinned_p = p;
                skinned_n = n;
            } else if (w_sum - 1.0).abs() > 1e-3 {
                skinned_p /= w_sum;
                skinned_n /= w_sum;
            }
            pos_w[i] = mesh_world.transform_point3(skinned_p);
            let nw = normal_world.transform_vector3(skinned_n);
            nrm_w[i] = if nw.length_squared() > 1e-10 {
                nw.normalize()
            } else {
                Vec3::Y
            };
        }

        out_tris.reserve(out_tris.len() + influenced.len());
        for [i0, i1, i2] in influenced {
            let w = w_sel[i0].max(w_sel[i1]).max(w_sel[i2]);
            out_tris.push((
                pos_w[i0] + nrm_w[i0] * zbias,
                pos_w[i1] + nrm_w[i1] * zbias,
                pos_w[i2] + nrm_w[i2] * zbias,
                weight_heat(w),
            ));
        }
    }

    out_tris
}

/// Selected-bone weight influence as coloured tris (cached while idle).
pub fn draw_weight_debug(scene: &mut Scene, rig: &mut RigDocument) {
    if !rig.show_weights || rig.selected.is_empty() {
        return;
    }

    if rig.weight_overlay.dirty {
        rig.weight_overlay.tris = rebuild_weight_overlay(scene, rig);
        rig.weight_overlay.dirty = false;
    }

    for &(a, b, c, color) in &rig.weight_overlay.tris {
        scene.debug.tri(
            a,
            b,
            c,
            PolyOpts {
                color,
                depth_test: true,
            },
        );
    }
}

/// Stick skeleton: thick outline + thinner fill, joint dots. Overlay so mesh never hides it.
pub fn draw_rig_debug(scene: &mut Scene, rig: &RigDocument) {
    if !rig.show_skeleton {
        return;
    }
    let segments = rig.bone_segments(scene);

    // Outlines first (drawn under fills in submission order).
    for (from, to, id) in &segments {
        let outline = if rig.is_selected(*id) {
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
        let fill = if rig.is_selected(*id) {
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
        let selected = rig.is_selected(b.id);
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
