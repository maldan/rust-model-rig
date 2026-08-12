//! Node-graph drivers: bone/morph I/O + vec/quat math, evaluated in Pose.

use std::collections::HashMap;

use glam::{EulerRot, Mat4, Quat, Vec2, Vec3};
use mega_render::{Handle, Mesh, Scene, Transform};
use mega_ui::{NodeLink, NodeSpace};

use crate::rig::{AppMode, BoneId, RigDocument};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverSpace {
    Local,
    World,
    /// Local relative to bind: `bind⁻¹ * local` (rot), `local - bind` (pos), `local / bind` (scale).
    BindOffset,
}

impl DriverSpace {
    pub const ALL: [DriverSpace; 3] = [
        DriverSpace::Local,
        DriverSpace::World,
        DriverSpace::BindOffset,
    ];

    pub fn index(self) -> usize {
        match self {
            DriverSpace::Local => 0,
            DriverSpace::World => 1,
            DriverSpace::BindOffset => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            DriverSpace::Local => "Local",
            DriverSpace::World => "World",
            DriverSpace::BindOffset => "Offset",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "local" => Some(Self::Local),
            "world" => Some(Self::World),
            "offset" | "bind_offset" | "bindoffset" => Some(Self::BindOffset),
            _ => None,
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::World => "world",
            Self::BindOffset => "offset",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverNodeKind {
    BoneGet,
    BoneSet,
    MorphSet,
    Float,
    Vec3,
    QuatEuler,
    /// Quaternion → Euler degrees (pitch, yaw, roll), YXZ — inverse of Quat Euler.
    QuatToEuler,
    Remap,
    MapRange,
    Clamp,
    Add,
    Mul,
    /// Pack three floats into a Vec3.
    CombineVec3,
    /// Split Vec3 into X / Y / Z floats.
    SplitVec3,
    Vec3Add,
    Vec3Scale,
    Vec3Length,
    Vec3Normalize,
    QuatMul,
    QuatRotateVec,
    QuatInvert,
    /// `slerp(identity, q, t)` — scale rotation influence (clavicle follow weight).
    QuatScale,
    /// Angle between two quaternions, in degrees.
    QuatAngle,
}

impl DriverNodeKind {
    pub fn title(self) -> &'static str {
        match self {
            DriverNodeKind::BoneGet => "Get Bone",
            DriverNodeKind::BoneSet => "Set Bone",
            DriverNodeKind::MorphSet => "Set Morph",
            DriverNodeKind::Float => "Float",
            DriverNodeKind::Vec3 => "Vec3",
            DriverNodeKind::QuatEuler => "Quat Euler",
            DriverNodeKind::QuatToEuler => "Quat → Euler",
            DriverNodeKind::Remap => "Remap",
            DriverNodeKind::MapRange => "Map Range",
            DriverNodeKind::Clamp => "Clamp",
            DriverNodeKind::Add => "Add",
            DriverNodeKind::Mul => "Mul",
            DriverNodeKind::CombineVec3 => "Combine XYZ",
            DriverNodeKind::SplitVec3 => "Split XYZ",
            DriverNodeKind::Vec3Add => "Vec3 Add",
            DriverNodeKind::Vec3Scale => "Vec3 Scale",
            DriverNodeKind::Vec3Length => "Length",
            DriverNodeKind::Vec3Normalize => "Normalize",
            DriverNodeKind::QuatMul => "Quat Mul",
            DriverNodeKind::QuatRotateVec => "Quat × Vec",
            DriverNodeKind::QuatInvert => "Quat Invert",
            DriverNodeKind::QuatScale => "Quat Scale",
            DriverNodeKind::QuatAngle => "Quat Angle",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        let mut k = s.trim().to_ascii_lowercase();
        k = k.replace('→', "to");
        k = k.replace('×', "x");
        k = k.replace([' ', '-'], "_");
        while k.contains("__") {
            k = k.replace("__", "_");
        }
        match k.as_str() {
            "bone_get" | "get_bone" => Some(Self::BoneGet),
            "bone_set" | "set_bone" => Some(Self::BoneSet),
            "morph_set" | "set_morph" => Some(Self::MorphSet),
            "float" => Some(Self::Float),
            "vec3" => Some(Self::Vec3),
            "quat_euler" | "quateuler" => Some(Self::QuatEuler),
            "quat_to_euler" | "quat_euler_out" => Some(Self::QuatToEuler),
            "remap" => Some(Self::Remap),
            "map_range" | "maprange" => Some(Self::MapRange),
            "clamp" => Some(Self::Clamp),
            "add" => Some(Self::Add),
            "mul" | "multiply" => Some(Self::Mul),
            "combine_vec3" | "combine_xyz" | "combinexyz" => Some(Self::CombineVec3),
            "split_vec3" | "split_xyz" | "splitxyz" => Some(Self::SplitVec3),
            "vec3_add" => Some(Self::Vec3Add),
            "vec3_scale" => Some(Self::Vec3Scale),
            "vec3_length" | "length" => Some(Self::Vec3Length),
            "vec3_normalize" | "normalize" => Some(Self::Vec3Normalize),
            "quat_mul" => Some(Self::QuatMul),
            "quat_rotate_vec" | "quat_x_vec" => Some(Self::QuatRotateVec),
            "quat_invert" => Some(Self::QuatInvert),
            "quat_scale" => Some(Self::QuatScale),
            "quat_angle" => Some(Self::QuatAngle),
            _ => None,
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::BoneGet => "bone_get",
            Self::BoneSet => "bone_set",
            Self::MorphSet => "morph_set",
            Self::Float => "float",
            Self::Vec3 => "vec3",
            Self::QuatEuler => "quat_euler",
            Self::QuatToEuler => "quat_to_euler",
            Self::Remap => "remap",
            Self::MapRange => "map_range",
            Self::Clamp => "clamp",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::CombineVec3 => "combine_vec3",
            Self::SplitVec3 => "split_vec3",
            Self::Vec3Add => "vec3_add",
            Self::Vec3Scale => "vec3_scale",
            Self::Vec3Length => "vec3_length",
            Self::Vec3Normalize => "vec3_normalize",
            Self::QuatMul => "quat_mul",
            Self::QuatRotateVec => "quat_rotate_vec",
            Self::QuatInvert => "quat_invert",
            Self::QuatScale => "quat_scale",
            Self::QuatAngle => "quat_angle",
        }
    }

    pub fn output_port_type(self, port: &str) -> u16 {
        use mega_ui::port_type::{ANY, FLOAT, QUAT, VEC3};
        match (self, port) {
            (Self::BoneGet, "pos" | "scale") => VEC3,
            (Self::BoneGet, "rot") => QUAT,
            (Self::Float, "value") => FLOAT,
            (Self::Vec3, "v") => VEC3,
            (Self::QuatEuler, "q") => QUAT,
            (Self::QuatToEuler, "euler") => VEC3,
            (Self::QuatToEuler, "x" | "y" | "z") => FLOAT,
            (
                Self::Remap | Self::MapRange | Self::Clamp | Self::Add | Self::Mul | Self::Vec3Length
                | Self::QuatAngle,
                "out",
            ) => FLOAT,
            (
                Self::CombineVec3
                | Self::Vec3Add
                | Self::Vec3Scale
                | Self::Vec3Normalize
                | Self::QuatRotateVec,
                "out",
            ) => VEC3,
            (Self::SplitVec3, "x" | "y" | "z") => FLOAT,
            (Self::QuatMul | Self::QuatInvert | Self::QuatScale, "out") => QUAT,
            _ => ANY,
        }
    }
}

#[derive(Clone)]
pub struct DriverNode {
    pub id: String,
    pub kind: DriverNodeKind,
    pub title: String,
    pub pos: Vec2,
    pub bone: Option<BoneId>,
    pub space: DriverSpace,
    pub mesh: Option<Handle<Mesh>>,
    pub shape: usize,
    /// Scalar / vec3 / euler params.
    pub floats: [f32; 4],
    pub preview: String,
}

impl DriverNode {
    pub fn new(id: String, kind: DriverNodeKind, pos: Vec2) -> Self {
        let mut floats = [0.0; 4];
        match kind {
            DriverNodeKind::Remap => {
                floats[0] = 0.0;
                floats[1] = 90.0;
            }
            DriverNodeKind::MapRange => {
                floats[0] = 0.0;
                floats[1] = 90.0;
                floats[2] = 0.0;
                floats[3] = 45.0;
            }
            DriverNodeKind::Clamp => {
                floats[0] = 0.0;
                floats[1] = 1.0;
            }
            DriverNodeKind::Vec3 => {
                floats[0] = 0.0;
                floats[1] = 0.0;
                floats[2] = 0.0;
            }
            DriverNodeKind::QuatEuler => {}
            DriverNodeKind::QuatScale => {
                floats[0] = 0.35;
            }
            _ => {}
        }
        Self {
            id,
            kind,
            title: kind.title().into(),
            pos,
            bone: None,
            space: DriverSpace::BindOffset,
            mesh: None,
            shape: 0,
            floats,
            preview: String::new(),
        }
    }
}

#[derive(Clone)]
pub enum DriverVal {
    F(f32),
    V3(Vec3),
    Q(Quat),
}

impl DriverVal {
    pub fn as_f(&self) -> f32 {
        match self {
            DriverVal::F(x) => *x,
            DriverVal::V3(v) => v.x,
            DriverVal::Q(_) => 0.0,
        }
    }

    pub fn as_v3(&self) -> Vec3 {
        match self {
            DriverVal::V3(v) => *v,
            DriverVal::F(x) => Vec3::splat(*x),
            DriverVal::Q(q) => Vec3::new(q.x, q.y, q.z),
        }
    }

    pub fn as_q(&self) -> Quat {
        match self {
            DriverVal::Q(q) => *q,
            _ => Quat::IDENTITY,
        }
    }

    pub fn format(&self) -> String {
        match self {
            DriverVal::F(x) => format!("{x:.3}"),
            DriverVal::V3(v) => format!("({:.2}, {:.2}, {:.2})", v.x, v.y, v.z),
            DriverVal::Q(q) => format!("q({:.2}, {:.2}, {:.2}, {:.2})", q.x, q.y, q.z, q.w),
        }
    }
}

#[derive(Clone)]
pub struct Driver {
    pub id: u32,
    pub name: String,
    pub enabled: bool,
    pub nodes: Vec<DriverNode>,
    pub space: NodeSpace,
    pub next_node_serial: u64,
}

impl Driver {
    pub fn new(id: u32, name: String) -> Self {
        let mut space = NodeSpace::new();
        space.pan = Vec2::new(20.0, 20.0);
        Self {
            id,
            name,
            enabled: true,
            nodes: Vec::new(),
            space,
            next_node_serial: 1,
        }
    }

    pub fn spawn_node(&mut self, kind: DriverNodeKind, pos: Vec2) -> String {
        let id = format!("n{}", self.next_node_serial);
        self.next_node_serial += 1;
        self.nodes.push(DriverNode::new(id.clone(), kind, pos));
        id
    }

    pub fn node_mut(&mut self, id: &str) -> Option<&mut DriverNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    pub fn connect(
        &mut self,
        from: &str,
        from_port: &str,
        to: &str,
        to_port: &str,
    ) -> Result<(), String> {
        let from_kind = self
            .nodes
            .iter()
            .find(|n| n.id == from)
            .map(|n| n.kind)
            .ok_or_else(|| format!("unknown node '{from}'"))?;
        if !self.nodes.iter().any(|n| n.id == to) {
            return Err(format!("unknown node '{to}'"));
        }
        self.space
            .links
            .retain(|l| !(l.to_node == to && l.to_port == to_port));
        let id = self.space.next_link_id;
        self.space.next_link_id += 1;
        self.space.links.push(NodeLink {
            id,
            from_node: from.to_string(),
            from_port: from_port.to_string(),
            to_node: to.to_string(),
            to_port: to_port.to_string(),
            ty: from_kind.output_port_type(from_port),
        });
        Ok(())
    }

    pub fn apply_deletes(&mut self) {
        for id in self.space.take_delete_nodes() {
            self.space.detach_node(&id);
            self.nodes.retain(|n| n.id != id);
        }
    }

    pub fn apply_clones(&mut self) {
        let ids = self.space.take_clone_nodes();
        if ids.is_empty() {
            return;
        }
        let offset = self.space.clone_offset();
        let mut id_map = HashMap::new();
        let mut new_sel = Vec::new();
        for old_id in &ids {
            let Some(src) = self.nodes.iter().find(|n| n.id == *old_id).cloned() else {
                continue;
            };
            let new_id = format!("n{}", self.next_node_serial);
            self.next_node_serial += 1;
            let mut clone = src;
            clone.id = new_id.clone();
            clone.pos += offset;
            clone.preview.clear();
            id_map.insert(old_id.clone(), new_id.clone());
            new_sel.push(new_id);
            self.nodes.push(clone);
        }
        self.space.duplicate_links(&id_map);
        self.space.selected_nodes = new_sel;
        self.space.selected_link = None;
    }

    pub fn drives_morph(&self, mesh: Handle<Mesh>, shape: usize) -> bool {
        self.enabled
            && self.nodes.iter().any(|n| {
                n.kind == DriverNodeKind::MorphSet
                    && n.mesh.is_some_and(|m| m.key() == mesh.key())
                    && n.shape == shape
            })
    }

    pub fn clear_bone_refs(&mut self, kill: &std::collections::HashSet<(u32, u32)>) {
        for n in &mut self.nodes {
            if n.bone.is_some_and(|b| kill.contains(&b.node.key())) {
                n.bone = None;
            }
        }
    }

    pub fn on_shape_removed(&mut self, mesh: Handle<Mesh>, index: usize) {
        for n in &mut self.nodes {
            if n.kind != DriverNodeKind::MorphSet {
                continue;
            }
            let Some(m) = n.mesh else {
                continue;
            };
            if m.key() != mesh.key() {
                continue;
            }
            if n.shape == index {
                n.mesh = None;
                n.shape = 0;
            } else if n.shape > index {
                n.shape -= 1;
            }
        }
    }
}

fn bind_local(rig: &RigDocument, bone: BoneId) -> Transform {
    rig.bind_locals
        .get(&bone.node.key())
        .copied()
        .unwrap_or(Transform::default())
}

fn mat_trs(m: Mat4) -> Transform {
    let (scale, rotation, translation) = m.to_scale_rotation_translation();
    Transform {
        translation,
        rotation,
        scale,
    }
}

fn bone_local(scene: &Scene, bone: BoneId) -> Option<Transform> {
    Some(scene.nodes.get(bone.node)?.local)
}

fn bone_world(scene: &Scene, bone: BoneId) -> Option<Transform> {
    scene.nodes.get(bone.node)?;
    Some(mat_trs(scene.world_matrix(bone.node)))
}

fn parent_world(scene: &Scene, bone: BoneId) -> Mat4 {
    scene
        .nodes
        .get(bone.node)
        .and_then(|n| n.parent)
        .map(|p| scene.world_matrix(p))
        .unwrap_or(Mat4::IDENTITY)
}

fn get_bone_trs(
    scene: &Scene,
    rig: &RigDocument,
    bone: BoneId,
    space: DriverSpace,
) -> Option<Transform> {
    match space {
        DriverSpace::Local => bone_local(scene, bone),
        DriverSpace::World => bone_world(scene, bone),
        DriverSpace::BindOffset => {
            let local = bone_local(scene, bone)?;
            let bind = bind_local(rig, bone);
            let scale = Vec3::new(
                div_axis(local.scale.x, bind.scale.x),
                div_axis(local.scale.y, bind.scale.y),
                div_axis(local.scale.z, bind.scale.z),
            );
            Some(Transform {
                translation: local.translation - bind.translation,
                rotation: (bind.rotation.inverse() * local.rotation).normalize(),
                scale,
            })
        }
    }
}

fn div_axis(a: f32, b: f32) -> f32 {
    if b.abs() < 1e-8 {
        a
    } else {
        a / b
    }
}

/// `blend` 1 = hard set; lower values slerp/lerp from current (stabilizes Pre-IK).
fn apply_bone_set(
    scene: &mut Scene,
    rig: &RigDocument,
    bone: BoneId,
    space: DriverSpace,
    pos: Option<Vec3>,
    rot: Option<Quat>,
    scale: Option<Vec3>,
    blend: f32,
) {
    if pos.is_none() && rot.is_none() && scale.is_none() {
        return;
    }
    let Some(cur_local) = bone_local(scene, bone) else {
        return;
    };
    let blend = blend.clamp(0.0, 1.0);

    let new_local = match space {
        DriverSpace::Local => Transform {
            translation: pos.unwrap_or(cur_local.translation),
            rotation: rot.unwrap_or(cur_local.rotation).normalize(),
            scale: scale.unwrap_or(cur_local.scale),
        },
        DriverSpace::BindOffset => {
            let bind = bind_local(rig, bone);
            Transform {
                translation: bind.translation + pos.unwrap_or(Vec3::ZERO),
                rotation: match rot {
                    Some(r) => (bind.rotation * r).normalize(),
                    None => cur_local.rotation,
                },
                scale: match scale {
                    Some(s) => bind.scale * s,
                    None => cur_local.scale,
                },
            }
        }
        DriverSpace::World => {
            let parent = parent_world(scene, bone);
            let cur_world = mat_trs(parent * cur_local.matrix());
            let world = Transform {
                translation: pos.unwrap_or(cur_world.translation),
                rotation: rot.unwrap_or(cur_world.rotation).normalize(),
                scale: scale.unwrap_or(cur_world.scale),
            };
            let local_m = parent.inverse() * world.matrix();
            mat_trs(local_m)
        }
    };

    if let Some(n) = scene.nodes.get_mut(bone.node) {
        if pos.is_some() {
            n.local.translation = cur_local.translation.lerp(new_local.translation, blend);
        }
        if rot.is_some() {
            n.local.rotation = cur_local
                .rotation
                .slerp(new_local.rotation, blend)
                .normalize();
        }
        if scale.is_some() {
            n.local.scale = cur_local.scale.lerp(new_local.scale, blend);
        }
    }
}

fn remap(raw: f32, from: f32, to: f32) -> f32 {
    let span = to - from;
    if span.abs() < 1e-5 {
        return if raw >= to { 1.0 } else { 0.0 };
    }
    ((raw - from) / span).clamp(0.0, 1.0)
}

fn map_range(raw: f32, in_from: f32, in_to: f32, out_from: f32, out_to: f32) -> f32 {
    let t = remap(raw, in_from, in_to);
    out_from + (out_to - out_from) * t
}

fn input_val(
    links: &[NodeLink],
    resolved: &HashMap<String, DriverVal>,
    node: &str,
    port: &str,
) -> Option<DriverVal> {
    for link in links {
        if link.to_node == node && link.to_port == port {
            let key = format!("{}:{}", link.from_node, link.from_port);
            return resolved.get(&key).cloned();
        }
    }
    None
}

fn as_f(v: Option<DriverVal>) -> f32 {
    v.map(|v| v.as_f()).unwrap_or(0.0)
}

fn as_v3(v: Option<DriverVal>) -> Vec3 {
    v.map(|v| v.as_v3()).unwrap_or(Vec3::ZERO)
}

fn as_q(v: Option<DriverVal>) -> Quat {
    v.map(|v| v.as_q()).unwrap_or(Quat::IDENTITY)
}

struct BoneSetOp {
    bone: BoneId,
    space: DriverSpace,
    pos: Option<Vec3>,
    rot: Option<Quat>,
    scale: Option<Vec3>,
}

fn eval_graph(
    scene: &Scene,
    rig: &RigDocument,
    driver: &mut Driver,
) -> (Vec<BoneSetOp>, Vec<(Handle<Mesh>, usize, f32)>) {
    let links = driver.space.links.clone();
    let mut resolved: HashMap<String, DriverVal> = HashMap::new();

    for _ in 0..16 {
        for n in &driver.nodes {
            match n.kind {
                DriverNodeKind::Float => {
                    resolved.insert(format!("{}:value", n.id), DriverVal::F(n.floats[0]));
                }
                DriverNodeKind::Vec3 => {
                    resolved.insert(
                        format!("{}:v", n.id),
                        DriverVal::V3(Vec3::new(n.floats[0], n.floats[1], n.floats[2])),
                    );
                }
                DriverNodeKind::QuatEuler => {
                    let q = Quat::from_euler(
                        EulerRot::YXZ,
                        n.floats[1].to_radians(),
                        n.floats[0].to_radians(),
                        n.floats[2].to_radians(),
                    );
                    resolved.insert(format!("{}:q", n.id), DriverVal::Q(q.normalize()));
                }
                DriverNodeKind::QuatToEuler => {
                    let q = as_q(input_val(&links, &resolved, &n.id, "q")).normalize();
                    let (y, x, z) = q.to_euler(EulerRot::YXZ);
                    let euler = Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees());
                    resolved.insert(format!("{}:euler", n.id), DriverVal::V3(euler));
                    resolved.insert(format!("{}:x", n.id), DriverVal::F(euler.x));
                    resolved.insert(format!("{}:y", n.id), DriverVal::F(euler.y));
                    resolved.insert(format!("{}:z", n.id), DriverVal::F(euler.z));
                }
                DriverNodeKind::BoneGet => {
                    let Some(bone) = n.bone.filter(|b| rig.bone(*b).is_some()) else {
                        continue;
                    };
                    let Some(trs) = get_bone_trs(scene, rig, bone, n.space) else {
                        continue;
                    };
                    resolved.insert(format!("{}:pos", n.id), DriverVal::V3(trs.translation));
                    resolved.insert(format!("{}:rot", n.id), DriverVal::Q(trs.rotation));
                    resolved.insert(format!("{}:scale", n.id), DriverVal::V3(trs.scale));
                }
                DriverNodeKind::Remap => {
                    let raw = as_f(input_val(&links, &resolved, &n.id, "in"));
                    resolved.insert(
                        format!("{}:out", n.id),
                        DriverVal::F(remap(raw, n.floats[0], n.floats[1])),
                    );
                }
                DriverNodeKind::MapRange => {
                    let raw = as_f(input_val(&links, &resolved, &n.id, "in"));
                    resolved.insert(
                        format!("{}:out", n.id),
                        DriverVal::F(map_range(
                            raw,
                            n.floats[0],
                            n.floats[1],
                            n.floats[2],
                            n.floats[3],
                        )),
                    );
                }
                DriverNodeKind::Clamp => {
                    let raw = as_f(input_val(&links, &resolved, &n.id, "in"));
                    let lo = n.floats[0].min(n.floats[1]);
                    let hi = n.floats[0].max(n.floats[1]);
                    resolved.insert(format!("{}:out", n.id), DriverVal::F(raw.clamp(lo, hi)));
                }
                DriverNodeKind::Add => {
                    let a = as_f(input_val(&links, &resolved, &n.id, "a"));
                    let b = as_f(input_val(&links, &resolved, &n.id, "b"));
                    resolved.insert(format!("{}:out", n.id), DriverVal::F(a + b));
                }
                DriverNodeKind::Mul => {
                    let a = as_f(input_val(&links, &resolved, &n.id, "a"));
                    let b = as_f(input_val(&links, &resolved, &n.id, "b"));
                    resolved.insert(format!("{}:out", n.id), DriverVal::F(a * b));
                }
                DriverNodeKind::CombineVec3 => {
                    let x = as_f(input_val(&links, &resolved, &n.id, "x"));
                    let y = as_f(input_val(&links, &resolved, &n.id, "y"));
                    let z = as_f(input_val(&links, &resolved, &n.id, "z"));
                    resolved.insert(format!("{}:out", n.id), DriverVal::V3(Vec3::new(x, y, z)));
                }
                DriverNodeKind::SplitVec3 => {
                    let v = as_v3(input_val(&links, &resolved, &n.id, "v"));
                    resolved.insert(format!("{}:x", n.id), DriverVal::F(v.x));
                    resolved.insert(format!("{}:y", n.id), DriverVal::F(v.y));
                    resolved.insert(format!("{}:z", n.id), DriverVal::F(v.z));
                }
                DriverNodeKind::Vec3Add => {
                    let a = as_v3(input_val(&links, &resolved, &n.id, "a"));
                    let b = as_v3(input_val(&links, &resolved, &n.id, "b"));
                    resolved.insert(format!("{}:out", n.id), DriverVal::V3(a + b));
                }
                DriverNodeKind::Vec3Scale => {
                    let v = as_v3(input_val(&links, &resolved, &n.id, "v"));
                    let s = as_f(input_val(&links, &resolved, &n.id, "s"));
                    resolved.insert(format!("{}:out", n.id), DriverVal::V3(v * s));
                }
                DriverNodeKind::Vec3Length => {
                    let v = as_v3(input_val(&links, &resolved, &n.id, "v"));
                    resolved.insert(format!("{}:out", n.id), DriverVal::F(v.length()));
                }
                DriverNodeKind::Vec3Normalize => {
                    let v = as_v3(input_val(&links, &resolved, &n.id, "v"));
                    resolved.insert(
                        format!("{}:out", n.id),
                        DriverVal::V3(v.try_normalize().unwrap_or(Vec3::ZERO)),
                    );
                }
                DriverNodeKind::QuatMul => {
                    let a = as_q(input_val(&links, &resolved, &n.id, "a"));
                    let b = as_q(input_val(&links, &resolved, &n.id, "b"));
                    resolved.insert(
                        format!("{}:out", n.id),
                        DriverVal::Q((a * b).normalize()),
                    );
                }
                DriverNodeKind::QuatRotateVec => {
                    let q = as_q(input_val(&links, &resolved, &n.id, "q"));
                    let v = as_v3(input_val(&links, &resolved, &n.id, "v"));
                    resolved.insert(format!("{}:out", n.id), DriverVal::V3(q * v));
                }
                DriverNodeKind::QuatInvert => {
                    let q = as_q(input_val(&links, &resolved, &n.id, "q"));
                    resolved.insert(
                        format!("{}:out", n.id),
                        DriverVal::Q(q.inverse().normalize()),
                    );
                }
                DriverNodeKind::QuatScale => {
                    let q = as_q(input_val(&links, &resolved, &n.id, "q"));
                    // Unconnected weight falls back to node float (default 0.35).
                    let t = input_val(&links, &resolved, &n.id, "t")
                        .map(|v| v.as_f())
                        .unwrap_or(n.floats[0])
                        .clamp(0.0, 1.0);
                    let out = Quat::IDENTITY.slerp(q.normalize(), t).normalize();
                    resolved.insert(format!("{}:out", n.id), DriverVal::Q(out));
                }
                DriverNodeKind::QuatAngle => {
                    let a = as_q(input_val(&links, &resolved, &n.id, "a")).normalize();
                    let b = as_q(input_val(&links, &resolved, &n.id, "b")).normalize();
                    let deg = a.angle_between(b).to_degrees();
                    resolved.insert(format!("{}:out", n.id), DriverVal::F(deg));
                }
                DriverNodeKind::BoneSet | DriverNodeKind::MorphSet => {
                    for port in ["pos", "rot", "scale", "in"] {
                        if let Some(v) = input_val(&links, &resolved, &n.id, port) {
                            resolved.insert(format!("{}:{port}", n.id), v);
                        }
                    }
                }
            }
        }
    }

    let mut bone_ops: Vec<BoneSetOp> = Vec::new();
    let mut morph_writes: Vec<(Handle<Mesh>, usize, f32)> = Vec::new();

    for n in &mut driver.nodes {
        let preview_key = match n.kind {
            DriverNodeKind::Float => format!("{}:value", n.id),
            DriverNodeKind::Vec3 => format!("{}:v", n.id),
            DriverNodeKind::QuatEuler => format!("{}:q", n.id),
            DriverNodeKind::QuatToEuler => format!("{}:euler", n.id),
            DriverNodeKind::BoneGet => format!("{}:rot", n.id),
            DriverNodeKind::BoneSet => format!("{}:rot", n.id),
            DriverNodeKind::MorphSet => format!("{}:in", n.id),
            DriverNodeKind::SplitVec3 => {
                let x = resolved
                    .get(&format!("{}:x", n.id))
                    .map(|v| v.as_f())
                    .unwrap_or(0.0);
                let y = resolved
                    .get(&format!("{}:y", n.id))
                    .map(|v| v.as_f())
                    .unwrap_or(0.0);
                let z = resolved
                    .get(&format!("{}:z", n.id))
                    .map(|v| v.as_f())
                    .unwrap_or(0.0);
                n.preview = format!("({x:.2}, {y:.2}, {z:.2})");
                continue;
            }
            _ => format!("{}:out", n.id),
        };
        n.preview = resolved
            .get(&preview_key)
            .map(|v| v.format())
            .unwrap_or_default();

        match n.kind {
            DriverNodeKind::BoneSet => {
                let Some(bone) = n.bone.filter(|b| rig.bone(*b).is_some()) else {
                    continue;
                };
                let pos = resolved
                    .get(&format!("{}:pos", n.id))
                    .map(|v| v.as_v3());
                let rot = resolved.get(&format!("{}:rot", n.id)).map(|v| v.as_q());
                let scale = resolved
                    .get(&format!("{}:scale", n.id))
                    .map(|v| v.as_v3());
                if pos.is_some() || rot.is_some() || scale.is_some() {
                    bone_ops.push(BoneSetOp {
                        bone,
                        space: n.space,
                        pos,
                        rot,
                        scale,
                    });
                }
            }
            DriverNodeKind::MorphSet => {
                let Some(mesh) = n.mesh else {
                    continue;
                };
                let Some(v) = resolved.get(&format!("{}:in", n.id)) else {
                    continue;
                };
                morph_writes.push((mesh, n.shape, v.as_f().clamp(0.0, 1.0)));
            }
            _ => {}
        }
    }

    (bone_ops, morph_writes)
}

/// When a driver runs relative to IK.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverPass {
    /// Before IK: writes to parents of IK chains (clavicle, etc.) from last frame's pose.
    PreIk,
    /// After IK / soft: morphs + bones that don't re-parent an IK chain.
    PostIk,
}

fn is_strict_ancestor(rig: &RigDocument, ancestor: BoneId, descendant: BoneId) -> bool {
    let mut cur = rig.bone(descendant).and_then(|b| b.parent);
    while let Some(p) = cur {
        if p == ancestor {
            return true;
        }
        cur = rig.bone(p).and_then(|b| b.parent);
    }
    false
}

/// True when writing `bone` changes the parent space of an enabled IK chain.
pub fn bone_supports_ik(rig: &RigDocument, bone: BoneId) -> bool {
    for chain in rig.ik_chains.iter().filter(|c| c.enabled) {
        for &b in chain.bones.iter().chain(std::iter::once(&chain.tip)) {
            if is_strict_ancestor(rig, bone, b) {
                return true;
            }
        }
    }
    false
}

/// Whether this driver has a Set Bone that runs in the Pre-IK pass.
pub fn driver_needs_pre_ik(rig: &RigDocument, driver: &Driver) -> bool {
    driver.nodes.iter().any(|n| {
        n.kind == DriverNodeKind::BoneSet && n.bone.is_some_and(|b| bone_supports_ik(rig, b))
    })
}

/// Evaluate drivers for one pass. Host order: PreIk → IK → soft → PostIk.
pub fn evaluate_drivers(scene: &mut Scene, rig: &mut RigDocument, pass: DriverPass) {
    if rig.mode != AppMode::Pose {
        return;
    }

    let mut all_bone: Vec<BoneSetOp> = Vec::new();
    let mut all_morph: Vec<(Handle<Mesh>, usize, f32)> = Vec::new();

    let mut drivers = std::mem::take(&mut rig.drivers);
    for driver in &mut drivers {
        if !driver.enabled {
            // Still refresh previews in PostIk so the editor stays live.
            if pass == DriverPass::PostIk {
                let _ = eval_graph(scene, rig, driver);
            }
            continue;
        }
        let (bone_ops, morph_w) = eval_graph(scene, rig, driver);
        match pass {
            DriverPass::PreIk => {
                for op in bone_ops {
                    if bone_supports_ik(rig, op.bone) {
                        all_bone.push(op);
                    }
                }
            }
            DriverPass::PostIk => {
                for op in bone_ops {
                    if !bone_supports_ik(rig, op.bone) {
                        all_bone.push(op);
                    }
                }
                all_morph.extend(morph_w);
            }
        }
    }
    rig.drivers = drivers;

    // Pre-IK writes to IK parents must not hard-snap 1:1 from the child pose
    // (period-2 oscillation). Soft blend + user QuatScale keeps it stable.
    let blend = match pass {
        DriverPass::PreIk => 0.55,
        DriverPass::PostIk => 1.0,
    };
    for op in all_bone {
        apply_bone_set(
            scene,
            rig,
            op.bone,
            op.space,
            op.pos,
            op.rot,
            op.scale,
            blend,
        );
    }

    if pass != DriverPass::PostIk || all_morph.is_empty() {
        return;
    }

    let mut meshes: Vec<Handle<Mesh>> = Vec::new();
    for (m, _, _) in &all_morph {
        if !meshes.iter().any(|h| h.key() == m.key()) {
            meshes.push(*m);
        }
    }

    for mesh_h in meshes {
        let Some(mesh) = scene.meshes.get_mut(mesh_h) else {
            continue;
        };
        while mesh.morph_weights.len() < mesh.morph_targets.len() {
            mesh.morph_weights.push(0.0);
        }
        for (m, shape, w) in &all_morph {
            if m.key() != mesh_h.key() {
                continue;
            }
            if let Some(slot) = mesh.morph_weights.get_mut(*shape) {
                *slot = (*w).clamp(0.0, 1.0);
            }
        }
        mesh.apply_morphs();
    }
}
