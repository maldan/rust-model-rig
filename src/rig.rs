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
    /// Shape keys / morph sculpt (bind pose).
    Shape,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tool {
    #[default]
    Select,
    /// Edit: place / extrude a child bone.
    AddBone,
    Translate,
    Rotate,
    /// Shape mode: sculpt brush (`brush_kind`).
    Brush,
    /// Pose: soft-grab a soft-chain particle (test / poke physics).
    SoftGrab,
}

/// Sculpt brush mode (Shape + `Tool::Brush`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BrushKind {
    #[default]
    Grab,
    /// Push / pull along vertex normals.
    Inflate,
    /// Laplacian relax toward neighbors.
    Smooth,
}

/// How the Move tool applies translation (Pose). Edit always uses FK.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MoveMode {
    /// Local/world translate of selection roots (current behaviour).
    #[default]
    Fk,
    /// CCD Auto-IK: short chain pull.
    AutoIk,
    /// Soft CCD with root falloff (whole ancestor chain).
    Verlet,
}

/// Gizmo / transform axis space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransformSpace {
    #[default]
    Local,
    World,
}

pub struct BoneInfo {
    pub id: BoneId,
    pub name: String,
    pub parent: Option<BoneId>,
    pub children: Vec<BoneId>,
    /// Joint participates in skin deformation.
    pub deform: bool,
}

/// Configured IK: deform chain + control target/pole bones.
#[derive(Clone, Debug)]
pub struct IkChain {
    pub id: u32,
    pub name: String,
    /// Effector (hand / foot / chain end) — position we reach for.
    pub tip: BoneId,
    /// Bones that rotate, root-first (any length ≥ 1).
    pub bones: Vec<BoneId>,
    /// Control bone (`deform: false`) — drag this in Pose.
    pub target: BoneId,
    /// Control bone — bend / pole preference.
    pub pole: BoneId,
    pub enabled: bool,
    /// Segment lengths bones[i]→bones[i+1], last → tip (for 2-bone analytic).
    pub lengths: Vec<f32>,
    /// `tip_world_rot = target_world_rot * tip_rot_offset` (set at Create IK).
    pub tip_rot_offset: glam::Quat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IkControlKind {
    Target,
    Pole,
}

/// Soft capsule collider attached to a bone (local medial axis + radius).
#[derive(Clone, Debug)]
pub struct BoneCollider {
    pub id: u32,
    pub bone: BoneId,
    pub enabled: bool,
    /// Capsule segment start in bone local space.
    pub a_local: Vec3,
    /// Capsule segment end in bone local space.
    pub b_local: Vec3,
    pub radius: f32,
    /// Soft push amount (0 = jelly, 1 = firmer — still penetrable).
    pub softness: f32,
}

/// Secondary dynamics: gravity + spring-to-pose + support plane (hair / cloth / soft tissue).
#[derive(Clone, Debug)]
pub struct SoftChain {
    pub id: u32,
    pub name: String,
    /// Fixed parent of `bones[0]` (follows body, not simulated).
    pub anchor: BoneId,
    /// Soft joints root→tip (linear). `bones[0]` is pinned to animated pose.
    pub bones: Vec<BoneId>,
    /// Rest length bones[i-1]→bones[i] (index 0 unused).
    pub lengths: Vec<f32>,
    /// Virtual particle past the tip (along tip +Y), so the last bone also swings.
    pub tip_length: f32,
    /// Support plane normal in `anchor` local space (points "out" of the body).
    pub support_normal_local: Vec3,
    pub enabled: bool,
    /// World gravity acceleration (m/s²-ish in scene units).
    pub gravity: f32,
    /// Spring accel toward animated rest (1/s²). Sag ≈ gravity / stiffness.
    pub stiffness: f32,
    /// Velocity decay rate (1/s). Frame-rate independent.
    pub damping: f32,
    /// How much motion trails the soft root (0 = glued, 1 = normal, >1 = exaggerated).
    /// Uses stable positional lag + accel response; safe with constraint absorb.
    pub inertia: f32,
    /// Radians from animated rest direction (cone).
    pub max_angle: f32,
    /// Runtime Verlet state (world). Includes virtual tip as last particle.
    pub(crate) prev_pos: Vec<Vec3>,
    pub(crate) curr_pos: Vec<Vec3>,
    /// Previous frame world matrix of soft root (`bones[0]`) for motion inheritance.
    pub(crate) prev_root_world: glam::Mat4,
    /// Soft-root world velocity (for inertia / fictitious force).
    pub(crate) prev_root_vel: Vec3,
    pub(crate) initialized: bool,
    /// 1 just after Soft Grab release; decays each frame to soften spring snap-back.
    pub(crate) grab_relax: f32,
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
    /// Debug wire capsules for bone colliders.
    pub show_colliders: bool,
    /// Cached weight tris; rebuilt only when dirty (selection / pose / load).
    weight_overlay: WeightOverlayCache,
    pub mode: AppMode,
    pub tool: Tool,
    pub move_mode: MoveMode,
    pub transform_space: TransformSpace,
    /// Configured IK limbs (arms / legs).
    pub ik_chains: Vec<IkChain>,
    /// Soft / secondary bone chains (gravity + support).
    pub soft_chains: Vec<SoftChain>,
    /// Soft capsule colliders on bones (body / breast volume, etc.).
    pub colliders: Vec<BoneCollider>,
    /// Mesh used for shape-key editing.
    pub active_mesh: Option<Handle<Mesh>>,
    /// Active morph target index on `active_mesh`.
    pub active_shape: Option<usize>,
    /// Grab brush radius in world units.
    pub brush_radius: f32,
    /// Grab brush strength multiplier (0–1).
    pub brush_strength: f32,
    pub brush_kind: BrushKind,
    /// Screen-space rect of the 3D viewport (updated each UI frame).
    pub viewport_rect: mega_ui::Rect,
    /// Counter for default bone names.
    next_bone_serial: u32,
    next_ik_serial: u32,
    next_soft_serial: u32,
    next_collider_serial: u32,
    next_shape_serial: u32,
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
            show_colliders: true,
            weight_overlay: WeightOverlayCache::default(),
            mode: AppMode::Pose,
            tool: Tool::Rotate,
            move_mode: MoveMode::Fk,
            transform_space: TransformSpace::Local,
            ik_chains: Vec::new(),
            soft_chains: Vec::new(),
            colliders: Vec::new(),
            active_mesh: None,
            active_shape: None,
            brush_radius: 0.08,
            brush_strength: 0.65,
            brush_kind: BrushKind::Grab,
            viewport_rect: mega_ui::Rect {
                min: glam::Vec2::ZERO,
                max: glam::Vec2::ZERO,
            },
            next_bone_serial: 1,
            next_ik_serial: 1,
            next_soft_serial: 1,
            next_collider_serial: 1,
            next_shape_serial: 1,
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
        self.ik_chains.clear();
        self.soft_chains.clear();
        self.colliders.clear();
        self.active_mesh = None;
        self.active_shape = None;
        self.brush_radius = 0.08;
        self.brush_strength = 0.65;
        self.brush_kind = BrushKind::Grab;
        self.next_bone_serial = 1;
        self.next_ik_serial = 1;
        self.next_soft_serial = 1;
        self.next_collider_serial = 1;
        self.next_shape_serial = 1;
        self.weight_overlay.clear();
        self.show_weights = false;
        self.move_mode = MoveMode::Fk;
        self.transform_space = TransformSpace::Local;
    }

    /// Empty scene + root bone (+ tip child so a segment is visible).
    pub fn new_skeleton(&mut self, scene: &mut Scene) {
        self.clear_model(scene);
        let root = self.insert_bone(scene, "Root", None, Transform::default(), true);
        let tip = self.insert_bone(
            scene,
            "Bone",
            Some(root),
            Transform::from_translation(Vec3::Y * 0.35),
            true,
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
            AppMode::Shape => {
                // Sculpt against bind geometry.
                self.reset_pose(scene);
                self.tool = Tool::Brush;
                if self.active_mesh.is_none() {
                    self.active_mesh = first_mesh(scene);
                }
            }
        }
        self.mode = mode;
    }

    pub fn create_shape_key(&mut self, scene: &mut Scene) -> Option<usize> {
        let mesh_h = self.active_mesh.or_else(|| {
            let h = first_mesh(scene);
            self.active_mesh = h;
            h
        })?;
        let mesh = scene.meshes.get_mut(mesh_h)?;
        let name = format!("Key_{}", self.next_shape_serial);
        self.next_shape_serial += 1;
        let idx = mesh.add_shape_key(name);
        mesh.set_morph_weight(idx, 1.0);
        self.active_shape = Some(idx);
        Some(idx)
    }

    pub fn delete_active_shape(&mut self, scene: &mut Scene) {
        let (Some(mesh_h), Some(idx)) = (self.active_mesh, self.active_shape) else {
            return;
        };
        let Some(mesh) = scene.meshes.get_mut(mesh_h) else {
            return;
        };
        mesh.remove_shape_key(idx);
        self.active_shape = if mesh.morph_targets.is_empty() {
            None
        } else {
            Some(idx.min(mesh.morph_targets.len() - 1))
        };
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
            true,
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
            true,
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

        let mut kill_set: HashSet<(u32, u32)> = kill.iter().map(|b| b.node.key()).collect();

        // Remove IK entries; include their control bones in the delete set.
        let mut keep_chains = Vec::new();
        for c in self.ik_chains.drain(..) {
            let hit = kill_set.contains(&c.tip.node.key())
                || kill_set.contains(&c.target.node.key())
                || kill_set.contains(&c.pole.node.key())
                || c.bones.iter().any(|b| kill_set.contains(&b.node.key()));
            if hit {
                if kill_set.insert(c.target.node.key()) {
                    kill.push(c.target);
                }
                if kill_set.insert(c.pole.node.key()) {
                    kill.push(c.pole);
                }
            } else {
                keep_chains.push(c);
            }
        }
        self.ik_chains = keep_chains;

        self.soft_chains.retain(|c| {
            !kill_set.contains(&c.anchor.node.key())
                && !c.bones.iter().any(|b| kill_set.contains(&b.node.key()))
        });
        self.colliders
            .retain(|c| !kill_set.contains(&c.bone.node.key()));

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

    /// Role of a bone if it is an IK control handle.
    pub fn ik_control_kind(&self, id: BoneId) -> Option<IkControlKind> {
        for c in &self.ik_chains {
            if c.target == id {
                return Some(IkControlKind::Target);
            }
            if c.pole == id {
                return Some(IkControlKind::Pole);
            }
        }
        None
    }

    pub fn ik_chain_for_tip(&self, tip: BoneId) -> Option<&IkChain> {
        self.ik_chains.iter().find(|c| c.tip == tip)
    }

    /// Create IK from tip. `rotate_count` = how many ancestors of tip rotate (1..=32).
    /// Tip itself is the effector (not counted in rotate_count).
    pub fn create_ik_from_tip(
        &mut self,
        scene: &mut Scene,
        tip: BoneId,
        rotate_count: usize,
    ) -> Result<u32, &'static str> {
        if self.ik_chains.iter().any(|c| c.tip == tip) {
            return Err("IK already exists on this tip");
        }
        if self.ik_control_kind(tip).is_some() {
            return Err("Cannot create IK on a control bone");
        }
        let rotate_count = rotate_count.clamp(1, 32);
        let mut bones_rev = Vec::new();
        let mut cur = self.bone(tip).and_then(|b| b.parent);
        while let Some(id) = cur {
            bones_rev.push(id);
            if bones_rev.len() >= rotate_count {
                break;
            }
            cur = self.bone(id).and_then(|b| b.parent);
        }
        if bones_rev.is_empty() {
            return Err("Tip needs at least one parent bone");
        }
        if bones_rev.len() < rotate_count {
            return Err("Not enough ancestors for this chain length");
        }
        bones_rev.reverse();
        let bones = bones_rev;

        let tip_pos = scene.world_matrix(tip.node).transform_point3(Vec3::ZERO);
        let mut lengths = Vec::with_capacity(bones.len());
        for i in 0..bones.len() {
            let a = scene
                .world_matrix(bones[i].node)
                .transform_point3(Vec3::ZERO);
            let b = if i + 1 < bones.len() {
                scene
                    .world_matrix(bones[i + 1].node)
                    .transform_point3(Vec3::ZERO)
            } else {
                tip_pos
            };
            lengths.push((b - a).length().max(1e-3));
        }

        let mid_id = bones[bones.len() / 2];
        let root_pos = scene
            .world_matrix(bones[0].node)
            .transform_point3(Vec3::ZERO);
        let mid_pos = scene.world_matrix(mid_id.node).transform_point3(Vec3::ZERO);
        let avg = (lengths.iter().sum::<f32>() / lengths.len() as f32).max(0.08);

        let axis = (tip_pos - root_pos).normalize_or_zero();
        let mut bend = mid_pos - root_pos - axis * (mid_pos - root_pos).dot(axis);
        if bend.length_squared() < 1e-8 {
            let mut s = axis.cross(Vec3::Y);
            if s.length_squared() < 1e-8 {
                s = axis.cross(Vec3::X);
            }
            bend = s;
        }
        let pole_pos = mid_pos + bend.normalize() * avg * 1.15;

        let tip_name = self
            .bone(tip)
            .map(|b| b.name.clone())
            .unwrap_or_else(|| "IK".into());
        let n = bones.len();
        let chain_name = format!("IK_{tip_name}×{n}");
        let target_name = format!("{tip_name}.IK");
        let pole_name = format!("{tip_name}.Pole");

        // Target inherits tip world rotation so Rotate on target drives the hand.
        let tip_world_rot = {
            let m = scene.world_matrix(tip.node);
            let (_, r, _) = m.to_scale_rotation_translation();
            r.normalize()
        };
        let target = self.insert_control_bone(scene, &target_name, tip_pos, tip_world_rot);
        let pole = self.insert_control_bone(scene, &pole_name, pole_pos, glam::Quat::IDENTITY);

        let id = self.next_ik_serial;
        self.next_ik_serial += 1;
        self.ik_chains.push(IkChain {
            id,
            name: chain_name,
            tip,
            bones,
            target,
            pole,
            enabled: true,
            lengths,
            // Matched at create → identity offset.
            tip_rot_offset: glam::Quat::IDENTITY,
        });
        self.set_selection(target);
        self.write_bind_for(scene, target);
        self.write_bind_for(scene, pole);
        Ok(id)
    }

    pub fn remove_ik_chain(&mut self, scene: &mut Scene, chain_id: u32) {
        let Some(idx) = self.ik_chains.iter().position(|c| c.id == chain_id) else {
            return;
        };
        let chain = self.ik_chains.remove(idx);
        self.delete_single_bone(scene, chain.target);
        self.delete_single_bone(scene, chain.pole);
    }

    pub fn set_ik_enabled(&mut self, chain_id: u32, enabled: bool) {
        if let Some(c) = self.ik_chains.iter_mut().find(|c| c.id == chain_id) {
            c.enabled = enabled;
        }
    }

    pub fn soft_chain_containing(&self, id: BoneId) -> Option<&SoftChain> {
        self.soft_chains.iter().find(|c| c.bones.iter().any(|b| *b == id))
    }

    /// Soft chain from a selected bone: expands along the linear parent/child path.
    /// Anchor = first branching (or multi-child) parent; needs ≥2 soft bones.
    pub fn create_soft_from_bone(
        &mut self,
        scene: &Scene,
        selected: BoneId,
    ) -> Result<u32, &'static str> {
        if self.ik_control_kind(selected).is_some() {
            return Err("Cannot create Soft on an IK control bone");
        }
        if self.soft_chain_containing(selected).is_some() {
            return Err("Bone already in a Soft chain");
        }

        let (anchor, bones) = collect_linear_soft_chain(self, selected)?;
        if bones.len() < 2 {
            return Err("Need at least 2 bones (root + tip). Extrude a child first");
        }
        if bones
            .iter()
            .any(|b| self.soft_chain_containing(*b).is_some())
        {
            return Err("Overlaps an existing Soft chain");
        }

        let mut lengths = Vec::with_capacity(bones.len());
        lengths.push(0.0); // unused for pinned root
        for i in 1..bones.len() {
            let a = scene
                .world_matrix(bones[i - 1].node)
                .transform_point3(Vec3::ZERO);
            let b = scene
                .world_matrix(bones[i].node)
                .transform_point3(Vec3::ZERO);
            lengths.push((b - a).length().max(1e-3));
        }
        let tip_length = {
            let avg = lengths.iter().skip(1).copied().sum::<f32>()
                / (lengths.len() - 1).max(1) as f32;
            avg.max(1e-3) * 0.65
        };

        let root_pos = scene
            .world_matrix(bones[0].node)
            .transform_point3(Vec3::ZERO);
        let tip_pos = scene
            .world_matrix(bones[bones.len() - 1].node)
            .transform_point3(Vec3::ZERO);
        let mut n_world = (tip_pos - root_pos).normalize_or_zero();
        if n_world.length_squared() < 1e-8 {
            let tip = bones[bones.len() - 1];
            n_world = scene
                .world_matrix(tip.node)
                .transform_vector3(Vec3::Y)
                .normalize_or_zero();
        }
        if n_world.length_squared() < 1e-8 {
            n_world = Vec3::Z;
        }
        let anchor_rot = {
            let m = scene.world_matrix(anchor.node);
            let (_, r, _) = m.to_scale_rotation_translation();
            r.normalize()
        };
        let support_normal_local = (anchor_rot.inverse() * n_world).normalize_or_zero();

        let root_name = self
            .bone(bones[0])
            .map(|b| b.name.clone())
            .unwrap_or_else(|| "Soft".into());
        let n = bones.len();
        let name = format!("Soft_{root_name}×{n}");

        let id = self.next_soft_serial;
        self.next_soft_serial += 1;
        self.soft_chains.push(SoftChain {
            id,
            name,
            anchor,
            bones,
            lengths,
            tip_length,
            support_normal_local,
            enabled: true,
            gravity: 9.8,
            stiffness: 45.0,
            damping: 2.5,
            inertia: 0.0,
            max_angle: 95f32.to_radians(),
            prev_pos: Vec::new(),
            curr_pos: Vec::new(),
            prev_root_world: glam::Mat4::IDENTITY,
            prev_root_vel: Vec3::ZERO,
            initialized: false,
            grab_relax: 0.0,
        });
        Ok(id)
    }

    pub fn remove_soft_chain(&mut self, chain_id: u32) {
        self.soft_chains.retain(|c| c.id != chain_id);
    }

    pub fn set_soft_enabled(&mut self, chain_id: u32, enabled: bool) {
        if let Some(c) = self.soft_chains.iter_mut().find(|c| c.id == chain_id) {
            c.enabled = enabled;
            if enabled {
                c.initialized = false;
            }
        }
    }

    pub fn collider_on_bone(&self, id: BoneId) -> Option<&BoneCollider> {
        self.colliders.iter().find(|c| c.bone == id)
    }

    /// Capsule along bone→child (or +Y), soft by default.
    pub fn create_capsule_collider(
        &mut self,
        scene: &Scene,
        bone: BoneId,
    ) -> Result<u32, &'static str> {
        if self.bone(bone).is_none() {
            return Err("Unknown bone");
        }
        if self.collider_on_bone(bone).is_some() {
            return Err("Bone already has a collider");
        }

        let bone_m = scene.world_matrix(bone.node);
        let inv = bone_m.inverse();

        let tip_world = self
            .bone(bone)
            .map(|b| b.children.clone())
            .unwrap_or_default()
            .into_iter()
            .next()
            .map(|ch| scene.world_matrix(ch.node).transform_point3(Vec3::ZERO))
            .unwrap_or_else(|| bone_m.transform_point3(Vec3::Y * 0.12));

        let mut tip_local = inv.transform_point3(tip_world);
        if tip_local.length_squared() < 1e-8 {
            tip_local = Vec3::Y * 0.12;
        }
        let len = tip_local.length().max(1e-3);
        let axis = tip_local.normalize();
        // Keep a bit of stem so the capsule sits on the bone segment.
        let a_local = axis * (len * 0.05);
        let b_local = axis * (len * 0.95);
        let radius = (len * 0.35).max(0.01);

        let id = self.next_collider_serial;
        self.next_collider_serial += 1;
        self.colliders.push(BoneCollider {
            id,
            bone,
            enabled: true,
            a_local,
            b_local,
            radius,
            softness: 0.35,
        });
        Ok(id)
    }

    pub fn remove_collider(&mut self, collider_id: u32) {
        self.colliders.retain(|c| c.id != collider_id);
    }

    fn insert_control_bone(
        &mut self,
        scene: &mut Scene,
        name: &str,
        world_pos: Vec3,
        world_rot: glam::Quat,
    ) -> BoneId {
        self.insert_bone(
            scene,
            name,
            None,
            Transform {
                translation: world_pos,
                rotation: world_rot.normalize(),
                scale: Vec3::ONE,
            },
            false,
        )
    }

    /// Delete one bone (no descendants). Used for IK controls.
    fn delete_single_bone(&mut self, scene: &mut Scene, id: BoneId) {
        if self.bone(id).is_none() {
            return;
        }
        let key = id.node.key();
        if let Some(parent) = self.bone(id).and_then(|b| b.parent) {
            if let Some(&pi) = self.bone_index.get(&parent.node.key()) {
                self.bones[pi].children.retain(|c| c.node.key() != key);
            }
        }
        let children: Vec<BoneId> = self
            .bone(id)
            .map(|b| b.children.clone())
            .unwrap_or_default();
        for child in children {
            if let Some(&ci) = self.bone_index.get(&child.node.key()) {
                self.bones[ci].parent = None;
            }
            if let Some(n) = scene.nodes.get_mut(child.node) {
                n.parent = None;
            }
        }

        self.selected.remove(&id);
        if self.selection == Some(id) {
            self.selection = self.selected.iter().copied().next();
        }
        self.bind_locals.remove(&key);
        self.bone_index.remove(&key);
        scene.nodes.remove(id.node);
        self.bones.retain(|b| b.id.node.key() != key);
        self.reindex_bones();
    }

    fn insert_bone(
        &mut self,
        scene: &mut Scene,
        name: &str,
        parent: Option<BoneId>,
        local: Transform,
        deform: bool,
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
            deform,
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
        self.reset_soft_sim();
        self.invalidate_weight_overlay();
    }

    pub fn reset_soft_sim(&mut self) {
        for c in &mut self.soft_chains {
            c.initialized = false;
        }
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
            // IK controls: joint markers only (no stick stub).
            if self.ik_control_kind(b.id).is_some() {
                continue;
            }
            let from = scene.world_matrix(b.id.node).transform_point3(Vec3::ZERO);
            if b.children.is_empty() {
                let m = scene.world_matrix(b.id.node);
                let axis = m.transform_vector3(Vec3::Y).normalize_or_zero();
                if axis.length_squared() > 1e-8 {
                    out.push((from, from + axis * fallback_len * 0.65, b.id));
                }
            } else {
                for &child in &b.children {
                    if self.ik_control_kind(child).is_some() {
                        continue;
                    }
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

impl BoneCollider {
    pub fn length(&self) -> f32 {
        (self.b_local - self.a_local).length()
    }

    /// Midpoint projected along the capsule axis, from bone origin along that axis.
    pub fn axis_offset(&self) -> f32 {
        let mut axis = (self.b_local - self.a_local).normalize_or_zero();
        if axis.length_squared() < 1e-8 {
            axis = Vec3::Y;
        }
        let mid = (self.a_local + self.b_local) * 0.5;
        mid.dot(axis)
    }

    pub fn set_length_offset(&mut self, length: f32, offset: f32) {
        let mut axis = (self.b_local - self.a_local).normalize_or_zero();
        if axis.length_squared() < 1e-8 {
            axis = Vec3::Y;
        }
        let len = length.max(1e-4);
        let mid = axis * offset;
        self.a_local = mid - axis * (len * 0.5);
        self.b_local = mid + axis * (len * 0.5);
    }
}

/// Soft chain = deepest path under `selected`, walking up to (but not including)
/// the first multi-child / root parent — that parent is the fixed anchor.
fn collect_linear_soft_chain(
    rig: &RigDocument,
    selected: BoneId,
) -> Result<(BoneId, Vec<BoneId>), &'static str> {
    if rig.bone(selected).is_none() {
        return Err("Invalid bone");
    }

    let tip = deepest_descendant(rig, selected);

    // Path tip → … → selected (and possibly above selected while single-child).
    let mut bones_rev = vec![tip];
    let mut cur = tip;
    while cur != selected {
        let Some(p) = rig.bone(cur).and_then(|b| b.parent) else {
            return Err("Selection is not an ancestor of its tip path");
        };
        bones_rev.push(p);
        cur = p;
        if bones_rev.len() > 32 {
            return Err("Soft chain too long");
        }
    }

    // Continue upward through a linear run so soft root sits under a branch/anchor.
    loop {
        let Some(p) = rig.bone(cur).and_then(|b| b.parent) else {
            return Err("Soft chain needs a parent anchor");
        };
        let p_child_count = rig.bone(p).map(|b| b.children.len()).unwrap_or(0);
        if p_child_count != 1 {
            bones_rev.reverse();
            return Ok((p, bones_rev));
        }
        bones_rev.push(p);
        cur = p;
        if bones_rev.len() > 32 {
            return Err("Soft chain too long");
        }
    }
}

fn chain_depth(rig: &RigDocument, id: BoneId) -> usize {
    let kids = rig.bone(id).map(|b| b.children.clone()).unwrap_or_default();
    if kids.is_empty() {
        return 1;
    }
    1 + kids
        .into_iter()
        .map(|c| chain_depth(rig, c))
        .max()
        .unwrap_or(0)
}

fn deepest_descendant(rig: &RigDocument, id: BoneId) -> BoneId {
    let kids = rig.bone(id).map(|b| b.children.clone()).unwrap_or_default();
    match kids.len() {
        0 => id,
        1 => deepest_descendant(rig, kids[0]),
        _ => {
            let best = kids
                .into_iter()
                .max_by_key(|c| chain_depth(rig, *c))
                .unwrap();
            deepest_descendant(rig, best)
        }
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
const IK_TARGET_FILL: [f32; 4] = [1.0, 0.55, 0.12, 1.0];
const IK_TARGET_OUTLINE: [f32; 4] = [0.45, 0.18, 0.02, 1.0];
const IK_POLE_FILL: [f32; 4] = [0.85, 0.35, 0.95, 1.0];
const IK_POLE_OUTLINE: [f32; 4] = [0.35, 0.08, 0.45, 1.0];

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
        let (outline, fill, joint, joint_out) = match (rig.ik_control_kind(b.id), selected) {
            (Some(IkControlKind::Target), false) => {
                (IK_TARGET_OUTLINE, IK_TARGET_FILL, BONE_JOINT * 1.35, BONE_JOINT_OUTLINE * 1.35)
            }
            (Some(IkControlKind::Pole), false) => {
                (IK_POLE_OUTLINE, IK_POLE_FILL, BONE_JOINT * 1.2, BONE_JOINT_OUTLINE * 1.2)
            }
            (Some(IkControlKind::Target), true) | (Some(IkControlKind::Pole), true) => {
                (BONE_SEL_OUTLINE, BONE_SEL_FILL, BONE_JOINT * 1.4, BONE_JOINT_OUTLINE * 1.4)
            }
            (None, true) => (BONE_SEL_OUTLINE, BONE_SEL_FILL, BONE_JOINT, BONE_JOINT_OUTLINE),
            (None, false) => (BONE_OUTLINE, BONE_FILL, BONE_JOINT, BONE_JOINT_OUTLINE),
        };
        scene
            .debug
            .point_ex(pos, outline, joint_out, false);
        scene.debug.point_ex(pos, fill, joint, false);
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

fn first_mesh(scene: &Scene) -> Option<Handle<Mesh>> {
    scene.nodes.iter().find_map(|(_, n)| n.mesh)
}
