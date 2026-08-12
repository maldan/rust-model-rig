//! Application state, UI, and viewport tools.

use glam::{Vec2, Vec3};
use mega_render::{GizmoAxis, Light, Scene, SkinningMode, Visualizer, WgpuVisualizer};
use mega_ui::{
    port_type, DockNode, DockState, NodePortSide, ScrollAxes, TextStyle, Ui, Window,
};

use crate::driver::{DriverNodeKind, DriverSpace};
use crate::framework::{Demo, UiCtx, SCENE_TEX};
use crate::gizmo::{self, RotateDrag, TranslateDrag};
use crate::ik::{self, IkPullDrag};
use crate::pick;
use crate::rig::{
    empty_scene, AppMode, BoneId, BrushKind, IkControlKind, MoveMode, RigDocument, Tool,
    TransformSpace,
};
use crate::sculpt::{self, SculptDrag};
use crate::soft_chain::{self, SoftGrabDrag};
use crate::verlet::{self, VerletDrag};

pub struct AppState {
    pub scene: Scene,
    pub rig: RigDocument,
    pub status: String,
    pub rotate_drag: Option<RotateDrag>,
    pub translate_drag: Option<TranslateDrag>,
    pub ik_drag: Option<IkPullDrag>,
    pub verlet_drag: Option<VerletDrag>,
    pub sculpt_drag: Option<SculptDrag>,
    pub soft_grab_drag: Option<SoftGrabDrag>,
    pub marquee: Option<MarqueeDrag>,
    pub gizmo_hover: Option<GizmoAxis>,
    /// Euler degrees shown in inspector (synced from selection).
    pub edit_euler_deg: Vec3,
    pub edit_translation: Vec3,
    pub edit_bone: Option<BoneId>,
    /// Host should re-read orbit cam from `scene.camera`.
    pub resync_camera: bool,
    /// Host should drop visualizer mesh/texture GPU caches (after Scene replace).
    pub clear_gpu_cache: bool,
    /// Bones that rotate when creating IK (tip's ancestors). Default 2 = arm/leg.
    pub ik_create_length: usize,
    /// Open floating driver node editor for this driver id.
    pub editing_driver: Option<u32>,
    /// Drill-down page for driver spawn context menu (0 = root).
    pub driver_spawn_page: u8,
}

#[derive(Clone, Copy)]
pub struct MarqueeDrag {
    pub start: Vec2,
    pub current: Vec2,
    pub additive: bool,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            scene: empty_scene(),
            rig: RigDocument::default(),
            status: "File → New Skeleton · Tab switches Edit/Pose".into(),
            rotate_drag: None,
            translate_drag: None,
            ik_drag: None,
            verlet_drag: None,
            sculpt_drag: None,
            soft_grab_drag: None,
            marquee: None,
            gizmo_hover: None,
            edit_euler_deg: Vec3::ZERO,
            edit_translation: Vec3::ZERO,
            edit_bone: None,
            resync_camera: false,
            clear_gpu_cache: false,
            ik_create_length: 2,
            editing_driver: None,
            driver_spawn_page: 0,
        }
    }

    pub fn has_drag(&self) -> bool {
        self.rotate_drag.is_some()
            || self.translate_drag.is_some()
            || self.ik_drag.is_some()
            || self.verlet_drag.is_some()
            || self.sculpt_drag.is_some()
            || self.soft_grab_drag.is_some()
            || self.marquee.is_some()
    }

    pub fn clear_drags(&mut self) {
        self.rotate_drag = None;
        self.translate_drag = None;
        self.ik_drag = None;
        self.verlet_drag = None;
        self.sculpt_drag = None;
        self.soft_grab_drag = None;
        self.marquee = None;
    }

    /// Wipe interaction / inspector state after replacing the document or scene.
    pub fn reset_session_state(&mut self) {
        self.clear_drags();
        self.gizmo_hover = None;
        self.edit_bone = None;
        self.edit_euler_deg = Vec3::ZERO;
        self.edit_translation = Vec3::ZERO;
        self.ik_create_length = 2;
        self.editing_driver = None;
        self.driver_spawn_page = 0;
        self.resync_camera = true;
        self.clear_gpu_cache = true;
    }

    fn status_from_selection(&mut self) {
        let n = self.rig.selected.len();
        self.status = match n {
            0 => "Selection cleared.".into(),
            1 => {
                let id = self.rig.selection.unwrap();
                let name = self
                    .rig
                    .bone(id)
                    .map(|b| b.name.clone())
                    .unwrap_or_else(|| "?".into());
                format!("Selected: {name}")
            }
            _ => format!("Selected: {n} bones"),
        };
    }

    pub fn open_dialog(&mut self) {
        let path = rfd::FileDialog::new()
            .add_filter("glTF", &["gltf", "glb"])
            .pick_file();
        let Some(path) = path else {
            return;
        };
        match self.rig.load_path(&mut self.scene, &path) {
            Ok(()) => {
                // Pose editor: strip clips so nothing overwrites FK.
                self.scene.animators.clear();
                self.reset_session_state();
                let with_parent = self.rig.bones.iter().filter(|b| b.parent.is_some()).count();
                self.status = format!(
                    "Loaded {} · {} bones ({} with parent) · Pose mode",
                    path.file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("model"),
                    self.rig.bones.len(),
                    with_parent
                );
            }
            Err(e) => {
                self.status = format!("Load failed: {e}");
            }
        }
    }

    pub fn new_skeleton(&mut self) {
        self.rig.new_skeleton(&mut self.scene);
        self.reset_session_state();
        self.status = "New skeleton · Edit mode · Extrude (E) / Add / Translate".into();
    }

    pub fn set_mode(&mut self, mode: AppMode) {
        self.rig.set_mode(&mut self.scene, mode);
        self.clear_drags();
        self.edit_bone = None;
        self.status = match mode {
            AppMode::Edit => "Edit mode — rest pose / hierarchy".into(),
            AppMode::Pose => "Pose mode — FK rotate".into(),
            AppMode::Shape => "Shape mode — Grab / Inflate / Smooth on shape keys".into(),
        };
    }

    pub fn sync_inspector_from_selection(&mut self) {
        let sel = self.rig.selection;
        if self.edit_bone != sel {
            self.edit_bone = sel;
            if let Some(id) = sel {
                if let Some(n) = self.scene.nodes.get(id.node) {
                    let (y, x, z) = n.local.rotation.to_euler(glam::EulerRot::YXZ);
                    self.edit_euler_deg =
                        Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees());
                    self.edit_translation = n.local.translation;
                }
            }
        }
    }

    pub fn apply_inspector_euler(&mut self) {
        let Some(id) = self.edit_bone else {
            return;
        };
        let rad = Vec3::new(
            self.edit_euler_deg.x.to_radians(),
            self.edit_euler_deg.y.to_radians(),
            self.edit_euler_deg.z.to_radians(),
        );
        // glam YXZ → args are (y, x, z)
        let q = glam::Quat::from_euler(glam::EulerRot::YXZ, rad.y, rad.x, rad.z);
        if let Some(n) = self.scene.nodes.get_mut(id.node) {
            n.local.rotation = q;
        }
        self.rig.invalidate_weight_overlay();
        if self.rig.mode == AppMode::Edit {
            self.rig.write_bind_for(&self.scene, id);
        }
    }

    pub fn apply_inspector_translation(&mut self) {
        let Some(id) = self.edit_bone else {
            return;
        };
        if let Some(n) = self.scene.nodes.get_mut(id.node) {
            n.local.translation = self.edit_translation;
        }
        self.rig.invalidate_weight_overlay();
        if self.rig.mode == AppMode::Edit {
            self.rig.write_bind_for(&self.scene, id);
        }
    }

    pub fn gizmo_radius(&self) -> f32 {
        let Some(pivot) = self.rig.selection_pivot(&self.scene) else {
            return 0.15;
        };
        gizmo::gizmo_radius_at(&self.scene, pivot, self.rig.viewport_rect.height())
    }

    pub fn gizmo_pivot(&self) -> Option<Vec3> {
        self.rig.selection_pivot(&self.scene)
    }

    fn sync_inspector_live(&mut self, bone: BoneId) {
        if let Some(n) = self.scene.nodes.get(bone.node) {
            let (y, x, z) = n.local.rotation.to_euler(glam::EulerRot::YXZ);
            self.edit_euler_deg = Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees());
            self.edit_translation = n.local.translation;
            self.edit_bone = Some(bone);
        }
    }
}

pub struct PointerFrame {
    pub pos: Vec2,
    pub pressed: bool,
    pub down: bool,
    pub released: bool,
    pub ctrl: bool,
}

const MARQUEE_CLICK_PX: f32 = 5.0;

/// Viewport select / gizmo after UI laid out `viewport_rect`.
pub fn handle_tools(state: &mut AppState, pointer: &PointerFrame, ui_wants_mouse: bool) {
    let rect = state.rig.viewport_rect;
    let over = rect.width() > 1.0 && rect.height() > 1.0 && rect.contains(pointer.pos);

    if pointer.released {
        if let Some(m) = state.marquee.take() {
            finish_marquee(state, rect, m);
        }
        // Commit bind on edit transforms when drag ends.
        if state.rig.mode == AppMode::Edit {
            if let Some(d) = state.rotate_drag.take() {
                for &(bone, _, _, _) in &d.bones {
                    state.rig.write_bind_for(&state.scene, bone);
                }
            }
            if let Some(d) = state.translate_drag.take() {
                for &(bone, _, _) in &d.bones {
                    state.rig.write_bind_for(&state.scene, bone);
                }
            }
        }
        if let Some(grab) = state.soft_grab_drag.take() {
            soft_chain::release_soft_grab(&mut state.rig, &grab);
        }
        state.clear_drags();
    }

    // Update marquee while dragging.
    if let Some(ref mut m) = state.marquee {
        if pointer.down {
            m.current = pointer.pos;
        }
        return;
    }

    // Shape sculpt stroke.
    if state.sculpt_drag.is_some() {
        if pointer.down {
            let strength = state.rig.brush_strength;
            let radius = state.rig.brush_radius;
            let invert = pointer.ctrl;
            if let Some(ref mut drag) = state.sculpt_drag {
                sculpt::apply_stroke(
                    &mut state.scene,
                    drag,
                    rect,
                    pointer.pos,
                    radius,
                    strength,
                    invert,
                );
            }
            state.rig.invalidate_weight_overlay();
        }
        return;
    }

    // Hover gizmo handles (bone tools only).
    state.gizmo_hover = None;
    if state.rig.mode != AppMode::Shape
        && over
        && state.rig.selection.is_some()
        && !state.has_drag()
    {
        if let (Some(sel), Some(pivot)) = (state.rig.selection, state.gizmo_pivot()) {
            let radius = state.gizmo_radius();
            state.gizmo_hover = match state.rig.tool {
                Tool::Rotate => {
                    gizmo::hover_axis(
                        &state.scene,
                        sel,
                        pivot,
                        state.rig.transform_space,
                        rect,
                        pointer.pos,
                        radius,
                    )
                }
                Tool::Translate => gizmo::hover_translate_axis(
                    &state.scene,
                    sel,
                    pivot,
                    state.rig.transform_space,
                    rect,
                    pointer.pos,
                    radius,
                ),
                _ => None,
            };
        }
    } else if let Some(ref drag) = state.rotate_drag {
        state.gizmo_hover = Some(drag.axis);
    } else if let Some(ref drag) = state.translate_drag {
        state.gizmo_hover = Some(drag.axis);
    } else if let Some(ref drag) = state.ik_drag {
        state.gizmo_hover = Some(drag.axis);
    } else if let Some(ref drag) = state.verlet_drag {
        state.gizmo_hover = Some(drag.axis);
    }

    if state.rotate_drag.is_some() {
        if pointer.down {
            let bone = {
                let drag = state.rotate_drag.as_mut().unwrap();
                gizmo::apply_rotate(&mut state.scene, drag, rect, pointer.pos);
                drag.bone
            };
            state.rig.invalidate_weight_overlay();
            state.sync_inspector_live(bone);
        }
        return;
    }

    if state.translate_drag.is_some() {
        if pointer.down {
            let bone = {
                let drag = state.translate_drag.as_mut().unwrap();
                gizmo::apply_translate(&mut state.scene, drag, rect, pointer.pos);
                drag.bone
            };
            state.rig.invalidate_weight_overlay();
            state.sync_inspector_live(bone);
        }
        return;
    }

    if state.ik_drag.is_some() {
        if pointer.down {
            let bone = {
                let drag = state.ik_drag.as_ref().unwrap();
                ik::apply_ik_pull(&mut state.scene, drag, rect, pointer.pos);
                drag.bone
            };
            state.rig.invalidate_weight_overlay();
            state.sync_inspector_live(bone);
        }
        return;
    }

    if state.verlet_drag.is_some() {
        if pointer.down {
            let bone = {
                let drag = state.verlet_drag.as_ref().unwrap();
                verlet::apply_verlet_pull(&mut state.scene, drag, rect, pointer.pos);
                drag.bone
            };
            state.rig.invalidate_weight_overlay();
            state.sync_inspector_live(bone);
        }
        return;
    }

    if state.soft_grab_drag.is_some() {
        if pointer.down {
            if let Some(ref mut drag) = state.soft_grab_drag {
                soft_chain::update_soft_grab(&state.scene, drag, rect, pointer.pos);
            }
        }
        return;
    }

    if ui_wants_mouse || !over || !pointer.pressed {
        return;
    }

    // Shape mode: grab brush / mesh select.
    if state.rig.mode == AppMode::Shape {
        if state.rig.tool == Tool::Brush {
            if state.rig.active_shape.is_none() {
                state.status = "Create / select a shape key first".into();
                return;
            }
            // Ensure active weight visible while sculpting.
            if let (Some(mesh_h), Some(idx)) = (state.rig.active_mesh, state.rig.active_shape) {
                if let Some(mesh) = state.scene.meshes.get_mut(mesh_h) {
                    let w = mesh.morph_weights.get(idx).copied().unwrap_or(0.0);
                    if w < 0.05 {
                        mesh.set_morph_weight(idx, 1.0);
                    }
                }
            }
            if let Some(drag) =
                sculpt::begin_stroke(&state.scene, &state.rig, rect, pointer.pos)
            {
                state.rig.active_mesh = Some(drag.mesh);
                state.sculpt_drag = Some(drag);
                state.status = match state.rig.brush_kind {
                    BrushKind::Grab => "Grab…".into(),
                    BrushKind::Inflate => {
                        if pointer.ctrl {
                            "Deflate…".into()
                        } else {
                            "Inflate…".into()
                        }
                    }
                    BrushKind::Smooth => "Smooth…".into(),
                };
            }
            return;
        }
        // Select tool: pick mesh under cursor.
        if let Some(ray) = pick::ray_from_viewport(&state.scene, rect, pointer.pos) {
            if let Some(hit) = sculpt::pick_mesh(&state.scene, &ray) {
                state.rig.active_mesh = Some(hit.mesh);
                state.status = format!(
                    "Mesh {}:{}",
                    hit.mesh.index, hit.mesh.generation
                );
            }
        }
        return;
    }

    // Soft Grab: pick soft-chain particle and soft-pin to cursor.
    if state.rig.mode == AppMode::Pose && state.rig.tool == Tool::SoftGrab {
        if let Some(drag) =
            soft_chain::begin_soft_grab(&state.scene, &state.rig, rect, pointer.pos)
        {
            state.rig.set_selection(drag.bone);
            state.edit_bone = None;
            state.status = "Soft Grab…".into();
            state.soft_grab_drag = Some(drag);
        } else {
            state.status = "Soft Grab: click a soft joint".into();
        }
        return;
    }

    let roots = state.rig.selection_roots();

    // Gizmo grab first — only selection roots transform; children follow FK.
    if let (Some(sel), Some(pivot)) = (state.rig.selection, state.gizmo_pivot()) {
        let radius = state.gizmo_radius();
        match state.rig.tool {
            Tool::Rotate => {
                if let Some(drag) = gizmo::begin_rotate(
                    &state.scene,
                    sel,
                    &roots,
                    pivot,
                    state.rig.transform_space,
                    rect,
                    pointer.pos,
                    radius,
                ) {
                    state.rotate_drag = Some(drag);
                    return;
                }
            }
            Tool::Translate => {
                let pose_move = state.rig.mode == AppMode::Pose;
                let space = state.rig.transform_space;
                if pose_move && state.rig.move_mode == MoveMode::AutoIk {
                    if let Some(drag) = ik::begin_ik_pull(
                        &state.scene,
                        &state.rig,
                        sel,
                        pivot,
                        space,
                        rect,
                        pointer.pos,
                        radius,
                    ) {
                        state.ik_drag = Some(drag);
                        return;
                    }
                }
                if pose_move && state.rig.move_mode == MoveMode::Verlet {
                    if let Some(drag) = verlet::begin_verlet_pull(
                        &state.scene,
                        &state.rig,
                        sel,
                        pivot,
                        space,
                        rect,
                        pointer.pos,
                        radius,
                    ) {
                        state.verlet_drag = Some(drag);
                        return;
                    }
                }
                // FK (default), or soft-mode fallback when chain too short.
                if let Some(drag) = gizmo::begin_translate(
                    &state.scene,
                    sel,
                    &roots,
                    pivot,
                    space,
                    rect,
                    pointer.pos,
                    radius,
                ) {
                    state.translate_drag = Some(drag);
                    return;
                }
            }
            Tool::AddBone => {
                if let Some(parent) = state.rig.selection {
                    if state.rig.extrude_bone(&mut state.scene, parent).is_some() {
                        state.edit_bone = None;
                        state.status = "Extruded bone".into();
                    }
                } else if let Some(hit) = gizmo::ray_ground_hit(&state.scene, rect, pointer.pos) {
                    state.rig.add_root_at(&mut state.scene, hit);
                    state.edit_bone = None;
                    state.status = "Added root bone".into();
                }
                return;
            }
            Tool::Select | Tool::Brush | Tool::SoftGrab => {}
        }
    } else if state.rig.tool == Tool::AddBone {
        if let Some(hit) = gizmo::ray_ground_hit(&state.scene, rect, pointer.pos) {
            state.rig.add_root_at(&mut state.scene, hit);
            state.edit_bone = None;
            state.status = "Added root bone".into();
        }
        return;
    }

    // Ctrl+drag: marquee (additive). Plain click: pick / clear.
    if pointer.ctrl && state.rig.tool != Tool::AddBone {
        state.marquee = Some(MarqueeDrag {
            start: pointer.pos,
            current: pointer.pos,
            additive: true,
        });
        return;
    }

    let Some(ray) = pick::ray_from_viewport(&state.scene, rect, pointer.pos) else {
        return;
    };
    if let Some(id) = pick::pick_bone(&state.scene, &state.rig, &ray) {
        state.rig.set_selection(id);
        state.edit_bone = None;
        state.status_from_selection();
    } else if state.gizmo_hover.is_none() {
        state.rig.clear_selection();
        state.edit_bone = None;
        state.status_from_selection();
    }
}

fn finish_marquee(state: &mut AppState, viewport: mega_ui::Rect, m: MarqueeDrag) {
    let drag_len = (m.current - m.start).length();
    state.edit_bone = None;

    if drag_len < MARQUEE_CLICK_PX {
        if let Some(ray) = pick::ray_from_viewport(&state.scene, viewport, m.start) {
            if let Some(id) = pick::pick_bone(&state.scene, &state.rig, &ray) {
                state.rig.toggle_selection(id);
                state.status_from_selection();
            }
        }
        return;
    }

    let ids = pick::bones_in_screen_rect(&state.scene, &state.rig, viewport, m.start, m.current);
    state.rig.select_bones(&ids, m.additive);
    state.status_from_selection();
}

pub fn default_dock() -> DockState {
    DockState::new(DockNode::split_h(
        0.68,
        DockNode::leaf(&["Viewport"]),
        DockNode::split_v(
            0.58,
            DockNode::leaf(&["Bones", "Shapes", "Drivers", "Lights"]),
            DockNode::leaf(&["Inspector"]),
        ),
    ))
}

pub struct RigApp;

impl Demo for RigApp {
    fn title() -> &'static str {
        "model-rig"
    }

    fn build_state() -> AppState {
        AppState::new()
    }

    fn configure(visualizer: &mut WgpuVisualizer) {
        let shadow = visualizer.shadow_settings();
        shadow.map_size = 2048;

        let post = visualizer.post_process();
        post.tonemap.enabled = true;
        post.tonemap.exposure = 1.1;
        post.fxaa.enabled = true;
        post.bloom.enabled = true;
        post.bloom.intensity = 0.12;
        post.ao.enabled = true;
        post.ao.intensity = 0.7;
    }

    fn update(state: &mut AppState, _dt: f32) -> bool {
        if state.has_drag() {
            return true;
        }
        // Keep ticking while soft / drivers need live evaluation in Pose.
        state.rig.mode == AppMode::Pose
            && (state.rig.soft_chains.iter().any(|c| c.enabled)
                || state.rig.drivers.iter().any(|d| d.enabled)
                || state.editing_driver.is_some())
    }

    fn build_ui(ui: &mut Ui, ctx: &mut UiCtx<'_>) -> bool {
        // Match mega_ui theme::MENU_BAR_H / STATUS_BAR_H. Reserve one root
        // spacing gap between menu and dock — otherwise the status label was a
        // plain root widget pushed past the framebuffer (clipped in half).
        let menu_h = 26.0 * ui.scale();
        let status_h = 24.0 * ui.scale();
        // Root layout inserts default spacing (6px) between menu and dock.
        let gap = 6.0 * ui.scale();
        let dock_h = (ctx.window_size.y - menu_h - status_h - gap).max(1.0);
        let dock_size = Vec2::new(ctx.window_size.x, dock_h);

        ui.menu_bar(|ui| {
            ui.menu("File", |ui| {
                if ui.menu_item("New Skeleton").clicked() {
                    ctx.state.new_skeleton();
                }
                if ui.menu_item_icon("folder_open", "Open…").clicked() {
                    ctx.state.open_dialog();
                }
                if ui.menu_item("Reset pose").clicked() {
                    ctx.state.rig.reset_pose(&mut ctx.state.scene);
                    ctx.state.edit_bone = None;
                    ctx.state.status = "Pose reset to bind.".into();
                }
            });
            ui.menu("Mode", |ui| {
                if ui.menu_item("Edit skeleton").clicked() {
                    ctx.state.set_mode(AppMode::Edit);
                }
                if ui.menu_item("Pose").clicked() {
                    ctx.state.set_mode(AppMode::Pose);
                }
                if ui.menu_item("Shape keys").clicked() {
                    ctx.state.set_mode(AppMode::Shape);
                }
            });
            ui.menu("View", |ui| {
                let skel = ctx.state.rig.show_skeleton;
                let mesh = ctx.state.rig.show_mesh;
                let weights = ctx.state.rig.show_weights;
                let colliders = ctx.state.rig.show_colliders;
                if ui
                    .menu_item(if skel {
                        "Hide Skeleton"
                    } else {
                        "Show Skeleton"
                    })
                    .clicked()
                {
                    ctx.state.rig.show_skeleton = !skel;
                }
                if ui
                    .menu_item(if mesh { "Hide Mesh" } else { "Show Mesh" })
                    .clicked()
                {
                    ctx.state.rig.show_mesh = !mesh;
                    ctx.state.rig.apply_mesh_visibility(&mut ctx.state.scene);
                }
                if ui
                    .menu_item(if weights {
                        "Hide Weights"
                    } else {
                        "Show Weights"
                    })
                    .clicked()
                {
                    ctx.state.rig.show_weights = !weights;
                    ctx.state.rig.invalidate_weight_overlay();
                }
                let skin_mode = ctx.state.scene.skinning_mode;
                let skin_label = format!(
                    "Skinning: {} → {}",
                    skin_mode.label(),
                    skin_mode.next().label()
                );
                if ui.menu_item(&skin_label).clicked() {
                    ctx.state.scene.skinning_mode = skin_mode.next();
                    ctx.state.rig.invalidate_weight_overlay();
                    ctx.state.status = format!(
                        "Skinning: {} ({})",
                        ctx.state.scene.skinning_mode.label(),
                        match ctx.state.scene.skinning_mode {
                            SkinningMode::LinearBlend => "matrix blend",
                            SkinningMode::DualQuat => "dual quaternion",
                        }
                    );
                }
                if ui
                    .menu_item(if colliders {
                        "Hide Colliders"
                    } else {
                        "Show Colliders"
                    })
                    .clicked()
                {
                    ctx.state.rig.show_colliders = !colliders;
                }
            });
        });

        let mut keep = false;
        let UiCtx {
            state,
            dock,
            viewport_size,
            fps,
            frame_ms,
            ..
        } = ctx;

        ui.dock_space("main", dock_size, dock, |ui, tab| match tab {
            "Viewport" => {
                viewport_toolbar(ui, state);
                ui.separator();
                let size = ui.available_size();
                **viewport_size = size;
                let rect = ui.texture(SCENE_TEX, size);
                state.rig.viewport_rect = rect;
            }
            "Bones" => {
                bone_panel(ui, state);
            }
            "Shapes" => {
                shapes_panel(ui, state);
            }
            "Drivers" => {
                drivers_panel(ui, state);
            }
            "Lights" => {
                lights_panel(ui, state);
            }
            "Inspector" => {
                inspector_panel(ui, state);
                keep |= state.edit_bone.is_some();
            }
            _ => {}
        });

        driver_editor_window(ui, state);

        ui.status_bar(|ui| {
            ui.label_styled(
                &format!("{:.0} fps · {:.1} ms · {}", fps, frame_ms, state.status),
                TextStyle {
                    color: [0.65, 0.68, 0.72, 1.0],
                    size: 12.0,
                },
            );
        });

        keep || state.has_drag() || state.editing_driver.is_some()
    }
}

fn viewport_toolbar(ui: &mut Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        let mut mode_i = match state.rig.mode {
            AppMode::Edit => 0,
            AppMode::Pose => 1,
            AppMode::Shape => 2,
        };
        if ui
            .toggle("app_mode", &mut mode_i, &["Edit", "Pose", "Shape"])
            .changed()
        {
            state.set_mode(match mode_i {
                0 => AppMode::Edit,
                2 => AppMode::Shape,
                _ => AppMode::Pose,
            });
        }

        match state.rig.mode {
            AppMode::Edit => {
                let mut tool_i = match state.rig.tool {
                    Tool::Select => 0,
                    Tool::Translate => 1,
                    Tool::Rotate => 2,
                    Tool::AddBone => 3,
                    Tool::Brush | Tool::SoftGrab => 0,
                };
                if ui
                    .toggle("tool_edit", &mut tool_i, &["Select", "Move", "Rotate", "Add"])
                    .changed()
                {
                    state.rig.tool = match tool_i {
                        1 => Tool::Translate,
                        2 => Tool::Rotate,
                        3 => Tool::AddBone,
                        _ => Tool::Select,
                    };
                    state.clear_drags();
                }
            }
            AppMode::Pose => {
                let mut tool_i = match state.rig.tool {
                    Tool::Translate => 1,
                    Tool::Rotate => 2,
                    Tool::SoftGrab => 3,
                    _ => 0,
                };
                if ui
                    .toggle(
                        "tool_pose",
                        &mut tool_i,
                        &["Select", "Move", "Rotate", "Grab"],
                    )
                    .changed()
                {
                    state.rig.tool = match tool_i {
                        1 => Tool::Translate,
                        2 => Tool::Rotate,
                        3 => Tool::SoftGrab,
                        _ => Tool::Select,
                    };
                    state.clear_drags();
                    if state.rig.tool == Tool::SoftGrab {
                        state.status = "Tool: Soft Grab — drag soft joints".into();
                    }
                }
                if state.rig.tool == Tool::Translate {
                    let mut move_i = match state.rig.move_mode {
                        MoveMode::Fk => 0,
                        MoveMode::AutoIk => 1,
                        MoveMode::Verlet => 2,
                    };
                    if ui
                        .toggle("move_mode", &mut move_i, &["FK", "IK", "Soft"])
                        .changed()
                    {
                        state.rig.move_mode = match move_i {
                            1 => MoveMode::AutoIk,
                            2 => MoveMode::Verlet,
                            _ => MoveMode::Fk,
                        };
                        state.clear_drags();
                        state.status = match state.rig.move_mode {
                            MoveMode::Fk => "Move · FK".into(),
                            MoveMode::AutoIk => "Move · IK (CCD, short chain)".into(),
                            MoveMode::Verlet => "Move · Soft (CCD + falloff)".into(),
                        };
                    }
                }
            }
            AppMode::Shape => {
                let mut tool_i = match (state.rig.tool, state.rig.brush_kind) {
                    (Tool::Brush, BrushKind::Grab) => 1,
                    (Tool::Brush, BrushKind::Inflate) => 2,
                    (Tool::Brush, BrushKind::Smooth) => 3,
                    _ => 0,
                };
                if ui
                    .toggle(
                        "tool_shape",
                        &mut tool_i,
                        &["Select", "Grab", "Inflate", "Smooth"],
                    )
                    .changed()
                {
                    match tool_i {
                        1 => {
                            state.rig.tool = Tool::Brush;
                            state.rig.brush_kind = BrushKind::Grab;
                        }
                        2 => {
                            state.rig.tool = Tool::Brush;
                            state.rig.brush_kind = BrushKind::Inflate;
                        }
                        3 => {
                            state.rig.tool = Tool::Brush;
                            state.rig.brush_kind = BrushKind::Smooth;
                        }
                        _ => state.rig.tool = Tool::Select,
                    }
                    state.clear_drags();
                }
            }
        }

        if matches!(state.rig.tool, Tool::Translate | Tool::Rotate)
            && state.rig.mode != AppMode::Shape
        {
            let mut space_i = match state.rig.transform_space {
                TransformSpace::Local => 0,
                TransformSpace::World => 1,
            };
            if ui
                .toggle("xform_space", &mut space_i, &["Local", "World"])
                .changed()
            {
                state.rig.transform_space = match space_i {
                    1 => TransformSpace::World,
                    _ => TransformSpace::Local,
                };
                state.clear_drags();
            }
        }
    });
}

fn lights_panel(ui: &mut Ui, state: &mut AppState) {
    ui.label_styled(
        "Lights",
        TextStyle {
            color: [0.9, 0.9, 0.9, 1.0],
            size: 14.0,
        },
    );
    ui.separator();

    let avail = ui.available_size();
    ui.scroll_area("lights_scroll", avail, ScrollAxes::Vertical, |ui| {
        ui.label("Ambient");
        let _ = ui.slider("amb_r", &mut state.scene.ambient[0], 0.0..=1.0);
        let _ = ui.slider("amb_g", &mut state.scene.ambient[1], 0.0..=1.0);
        let _ = ui.slider("amb_b", &mut state.scene.ambient[2], 0.0..=1.0);
        ui.separator();

        for (i, light) in state.scene.lights.iter_mut().enumerate() {
            match light {
                Light::Directional(d) => {
                    ui.label(&format!("Directional #{i}"));
                    let _ = ui.checkbox(&format!("Enabled##dir_en_{i}"), &mut d.enabled);
                    let _ = ui.checkbox(&format!("Shadows##dir_sh_{i}"), &mut d.cast_shadows);
                    ui.label("Direction X / Y / Z");
                    let _ = ui.slider(&format!("dir_x_{i}"), &mut d.direction.x, -1.0..=1.0);
                    let _ = ui.slider(&format!("dir_y_{i}"), &mut d.direction.y, -1.0..=1.0);
                    let _ = ui.slider(&format!("dir_z_{i}"), &mut d.direction.z, -1.0..=1.0);
                    d.direction = d.direction.normalize_or_zero();
                    if d.direction.length_squared() < 1e-8 {
                        d.direction = Vec3::new(0.35, -0.55, 0.75).normalize();
                    }
                    ui.label("Intensity");
                    let _ = ui.slider(&format!("dir_int_{i}"), &mut d.intensity, 0.0..=10.0);
                    ui.label("Color R / G / B");
                    let _ = ui.slider(&format!("dir_cr_{i}"), &mut d.color[0], 0.0..=1.0);
                    let _ = ui.slider(&format!("dir_cg_{i}"), &mut d.color[1], 0.0..=1.0);
                    let _ = ui.slider(&format!("dir_cb_{i}"), &mut d.color[2], 0.0..=1.0);
                    ui.separator();
                }
                Light::Point(p) => {
                    ui.label(&format!("Point #{i}"));
                    let _ = ui.checkbox(&format!("Enabled##pt_en_{i}"), &mut p.enabled);
                    ui.label("Position X / Y / Z");
                    let _ = ui.slider(&format!("pt_x_{i}"), &mut p.position.x, -10.0..=10.0);
                    let _ = ui.slider(&format!("pt_y_{i}"), &mut p.position.y, -10.0..=10.0);
                    let _ = ui.slider(&format!("pt_z_{i}"), &mut p.position.z, -10.0..=10.0);
                    ui.label("Intensity");
                    let _ = ui.slider(&format!("pt_int_{i}"), &mut p.intensity, 0.0..=20.0);
                    ui.label("Range");
                    let _ = ui.slider(&format!("pt_range_{i}"), &mut p.range, 0.1..=30.0);
                    ui.label("Color R / G / B");
                    let _ = ui.slider(&format!("pt_cr_{i}"), &mut p.color[0], 0.0..=1.0);
                    let _ = ui.slider(&format!("pt_cg_{i}"), &mut p.color[1], 0.0..=1.0);
                    let _ = ui.slider(&format!("pt_cb_{i}"), &mut p.color[2], 0.0..=1.0);
                    ui.separator();
                }
            }
        }
    });
}

fn shapes_panel(ui: &mut Ui, state: &mut AppState) {
    ui.label_styled(
        "Shape Keys",
        TextStyle {
            color: [0.9, 0.9, 0.9, 1.0],
            size: 14.0,
        },
    );

    if state.rig.mode != AppMode::Shape {
        ui.horizontal(|ui| {
            ui.label("Not in Shape mode.");
            if ui.button("Shape").clicked() {
                state.set_mode(AppMode::Shape);
            }
        });
        ui.separator();
    }

    // Unique meshes (one entry per mesh handle).
    let mut meshes: Vec<(mega_render::Handle<mega_render::Mesh>, String)> = Vec::new();
    for (_, n) in state.scene.nodes.iter() {
        let Some(h) = n.mesh else {
            continue;
        };
        if meshes.iter().any(|(mh, _)| mh.key() == h.key()) {
            continue;
        }
        if state.scene.meshes.get(h).is_none() {
            continue;
        }
        let label = if n.name.is_empty() {
            format!("Mesh {}", h.index)
        } else {
            n.name.clone()
        };
        meshes.push((h, label));
    }

    if meshes.is_empty() {
        ui.label("No meshes — open a glTF model.");
        return;
    }

    if state
        .rig
        .active_mesh
        .is_none_or(|a| !meshes.iter().any(|(h, _)| h.key() == a.key()))
    {
        state.rig.active_mesh = Some(meshes[0].0);
        state.rig.active_shape = None;
    }

    let mut mesh_i = state
        .rig
        .active_mesh
        .and_then(|a| meshes.iter().position(|(h, _)| h.key() == a.key()))
        .unwrap_or(0);
    let mesh_labels: Vec<String> = meshes.iter().map(|(_, n)| n.clone()).collect();
    let mesh_opts: Vec<&str> = mesh_labels.iter().map(|s| s.as_str()).collect();
    if ui.select("shape_mesh", &mut mesh_i, &mesh_opts).changed() {
        state.rig.active_mesh = Some(meshes[mesh_i].0);
        state.rig.active_shape = None;
        if state.rig.mode != AppMode::Shape {
            state.set_mode(AppMode::Shape);
        }
    }

    ui.horizontal(|ui| {
        if ui.button("+ Shape").clicked() {
            if state.rig.mode != AppMode::Shape {
                state.set_mode(AppMode::Shape);
            }
            if let Some(idx) = state.rig.create_shape_key(&mut state.scene) {
                state.status = format!("Created shape key #{idx}");
            } else {
                state.status = "No mesh for shape key".into();
            }
        }
        if ui.button("Delete").clicked() {
            state.rig.delete_active_shape(&mut state.scene);
            state.status = "Deleted shape key".into();
        }
    });

    ui.separator();
    ui.label("Brush · radius / strength");
    ui.label(match state.rig.brush_kind {
        BrushKind::Grab => "Grab — drag surface",
        BrushKind::Inflate => "Inflate — hold/drag (Ctrl = deflate)",
        BrushKind::Smooth => "Smooth — hold/drag to relax",
    });
    let _ = ui.slider("brush_radius", &mut state.rig.brush_radius, 0.005..=0.5);
    let _ = ui.slider("brush_strength", &mut state.rig.brush_strength, 0.05..=1.0);

    ui.separator();
    let Some(mesh_h) = state.rig.active_mesh else {
        return;
    };

    let keys: Vec<(usize, String, f32)> = {
        let Some(mesh) = state.scene.meshes.get(mesh_h) else {
            ui.label("Mesh missing.");
            return;
        };
        mesh.morph_targets
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let w = mesh.morph_weights.get(i).copied().unwrap_or(0.0);
                (i, t.name.clone(), w)
            })
            .collect()
    };

    ui.label_styled(
        &format!("Keys ({})", keys.len()),
        TextStyle {
            color: [0.9, 0.9, 0.9, 1.0],
            size: 13.0,
        },
    );

    if keys.is_empty() {
        ui.label("No shape keys — press + Shape.");
        return;
    }

    let avail = ui.available_size();
    let active = state.rig.active_shape;
    let mut clicked: Option<usize> = None;
    let mut weight_edits: Vec<(usize, f32)> = Vec::new();

    ui.scroll_area("shapes_scroll", avail, ScrollAxes::Vertical, |ui| {
        for (i, name, mut w) in keys {
            let selected = active == Some(i);
            let driven = state.rig.active_mesh.is_some_and(|mesh| {
                state
                    .rig
                    .drivers
                    .iter()
                    .any(|d| d.drives_morph(mesh, i))
            });
            if ui
                .selectable(&format!("shape_{i}"), selected, |ui| {
                    if driven {
                        ui.label(&format!("{name} (driven)"));
                    } else {
                        ui.label(&name);
                    }
                })
                .clicked()
            {
                clicked = Some(i);
            }
            if ui.slider(&format!("shape_w_{i}"), &mut w, 0.0..=1.0).changed() {
                weight_edits.push((i, w));
                clicked = Some(i);
            }
        }
    });

    for (i, w) in weight_edits {
        if let Some(mesh) = state.scene.meshes.get_mut(mesh_h) {
            mesh.set_morph_weight(i, w);
        }
        state.rig.invalidate_weight_overlay();
    }

    if let Some(i) = clicked {
        state.rig.active_shape = Some(i);
        if state.rig.mode != AppMode::Shape {
            state.set_mode(AppMode::Shape);
        }
    }
}

fn drivers_panel(ui: &mut Ui, state: &mut AppState) {
    ui.label_styled(
        &format!("Drivers ({})", state.rig.drivers.len()),
        TextStyle {
            color: [0.9, 0.9, 0.9, 1.0],
            size: 14.0,
        },
    );
    ui.label("Named graphs · Edit opens floating node editor");
    ui.label("Set on IK parent (e.g. clavicle) → pre-IK; morphs / rest → post-IK");
    ui.separator();

    if ui.button("+ Driver").clicked() {
        let id = state.rig.create_driver();
        state.editing_driver = Some(id);
        if state.rig.mode != AppMode::Pose {
            state.set_mode(AppMode::Pose);
        }
        state.status = format!("Created Driver {id} — edit graph in floating window");
    }

    if state.rig.mode != AppMode::Pose {
        ui.horizontal(|ui| {
            ui.label("Drivers run in Pose.");
            if ui.button("Pose").clicked() {
                state.set_mode(AppMode::Pose);
            }
        });
    }

    ui.separator();
    if state.rig.drivers.is_empty() {
        ui.label("No drivers yet. + Driver to create one.");
        return;
    }

    let ids: Vec<u32> = state.rig.drivers.iter().map(|d| d.id).collect();
    let mut remove: Option<u32> = None;
    let mut open_edit: Option<u32> = None;
    let avail = ui.available_size();

    ui.scroll_area("drivers_scroll", avail, ScrollAxes::Vertical, |ui| {
        for id in ids {
            ui.separator();
            let Some(idx) = state.rig.drivers.iter().position(|d| d.id == id) else {
                continue;
            };
            let node_count = state.rig.drivers[idx].nodes.len();
            let link_count = state.rig.drivers[idx].space.links.len();
            let pre_ik = crate::driver::driver_needs_pre_ik(&state.rig, &state.rig.drivers[idx]);
            let editing = state.editing_driver == Some(id);

            ui.horizontal(|ui| {
                let mut enabled = state.rig.drivers[idx].enabled;
                if ui
                    .checkbox(&format!("##drv_en_{id}"), &mut enabled)
                    .changed()
                {
                    state.rig.drivers[idx].enabled = enabled;
                }
                ui.text_input(
                    &format!("drv_name_{id}"),
                    &mut state.rig.drivers[idx].name,
                );
            });
            let pass = if pre_ik { "pre-IK" } else { "post-IK" };
            ui.label(&format!("{node_count} nodes · {link_count} links · {pass}"));
            ui.horizontal(|ui| {
                let edit_label = if editing {
                    format!("Editing…##drv_ed_{id}")
                } else {
                    format!("Edit##drv_ed_{id}")
                };
                if ui.button(&edit_label).clicked() {
                    open_edit = Some(id);
                }
                if ui.button(&format!("Del##drv_del_{id}")).clicked() {
                    remove = Some(id);
                }
            });
        }
    });

    if let Some(id) = open_edit {
        state.editing_driver = Some(id);
        if state.rig.mode != AppMode::Pose {
            state.set_mode(AppMode::Pose);
        }
        state.status = format!("Editing {}", state.rig.drivers.iter().find(|d| d.id == id).map(|d| d.name.as_str()).unwrap_or("driver"));
    }
    if let Some(id) = remove {
        if state.editing_driver == Some(id) {
            state.editing_driver = None;
        }
        state.rig.remove_driver(id);
        state.status = format!("Removed driver #{id}");
    }
}

fn driver_spawn_menu(ui: &mut Ui, page: &mut u8) -> Option<DriverNodeKind> {
    let mut kind = None;
    match *page {
        1 => {
            if ui.menu_item_keep_open("← Back").clicked() {
                *page = 0;
            }
            ui.separator();
            ui.menu_section("Constants");
            if ui.menu_item("Float").clicked() {
                kind = Some(DriverNodeKind::Float);
            }
            if ui.menu_item("Vec3").clicked() {
                kind = Some(DriverNodeKind::Vec3);
            }
            if ui.menu_item("Quat Euler").clicked() {
                kind = Some(DriverNodeKind::QuatEuler);
            }
        }
        2 => {
            if ui.menu_item_keep_open("← Back").clicked() {
                *page = 0;
            }
            ui.separator();
            ui.menu_section("Math");
            if ui.menu_item("Remap 0–1").clicked() {
                kind = Some(DriverNodeKind::Remap);
            }
            if ui.menu_item("Map Range").clicked() {
                kind = Some(DriverNodeKind::MapRange);
            }
            if ui.menu_item("Clamp").clicked() {
                kind = Some(DriverNodeKind::Clamp);
            }
            if ui.menu_item("Add").clicked() {
                kind = Some(DriverNodeKind::Add);
            }
            if ui.menu_item("Mul").clicked() {
                kind = Some(DriverNodeKind::Mul);
            }
        }
        3 => {
            if ui.menu_item_keep_open("← Back").clicked() {
                *page = 0;
            }
            ui.separator();
            ui.menu_section("Vector");
            if ui.menu_item("Combine XYZ").clicked() {
                kind = Some(DriverNodeKind::CombineVec3);
            }
            if ui.menu_item("Split XYZ").clicked() {
                kind = Some(DriverNodeKind::SplitVec3);
            }
            if ui.menu_item("Vec3 Add").clicked() {
                kind = Some(DriverNodeKind::Vec3Add);
            }
            if ui.menu_item("Vec3 Scale").clicked() {
                kind = Some(DriverNodeKind::Vec3Scale);
            }
            if ui.menu_item("Length").clicked() {
                kind = Some(DriverNodeKind::Vec3Length);
            }
            if ui.menu_item("Normalize").clicked() {
                kind = Some(DriverNodeKind::Vec3Normalize);
            }
        }
        4 => {
            if ui.menu_item_keep_open("← Back").clicked() {
                *page = 0;
            }
            ui.separator();
            ui.menu_section("Quaternion");
            if ui.menu_item("Quat Mul").clicked() {
                kind = Some(DriverNodeKind::QuatMul);
            }
            if ui.menu_item("Quat × Vec").clicked() {
                kind = Some(DriverNodeKind::QuatRotateVec);
            }
            if ui.menu_item("Quat Invert").clicked() {
                kind = Some(DriverNodeKind::QuatInvert);
            }
            if ui.menu_item("Quat Scale").clicked() {
                kind = Some(DriverNodeKind::QuatScale);
            }
            if ui.menu_item("Quat Angle °").clicked() {
                kind = Some(DriverNodeKind::QuatAngle);
            }
            if ui.menu_item("Quat → Euler").clicked() {
                kind = Some(DriverNodeKind::QuatToEuler);
            }
        }
        _ => {
            if ui.menu_item("Get Bone").clicked() {
                kind = Some(DriverNodeKind::BoneGet);
            }
            if ui.menu_item("Set Bone").clicked() {
                kind = Some(DriverNodeKind::BoneSet);
            }
            if ui.menu_item("Set Morph").clicked() {
                kind = Some(DriverNodeKind::MorphSet);
            }
            ui.separator();
            if ui.menu_item_submenu("Constants").clicked() {
                *page = 1;
            }
            if ui.menu_item_submenu("Math").clicked() {
                *page = 2;
            }
            if ui.menu_item_submenu("Vector").clicked() {
                *page = 3;
            }
            if ui.menu_item_submenu("Quaternion").clicked() {
                *page = 4;
            }
        }
    }
    kind
}

fn bone_select_index(bones: &[(BoneId, String)], bone: Option<BoneId>) -> usize {
    match bone {
        Some(b) => bones
            .iter()
            .position(|(id, _)| *id == b)
            .map(|i| i + 1)
            .unwrap_or(0),
        None => 0,
    }
}

fn bone_picker_row(
    ui: &mut Ui,
    node_id: &str,
    bone: &mut Option<BoneId>,
    space: &mut DriverSpace,
    bone_list: &[(BoneId, String)],
    bone_labels: &[String],
    selection: Option<BoneId>,
    status_msg: &mut Option<String>,
) {
    let bone_opts: Vec<&str> = bone_labels.iter().map(|s| s.as_str()).collect();
    let space_opts = ["Local", "World", "Offset"];

    ui.horizontal(|ui| {
        let mut bi = bone_select_index(bone_list, *bone);
        if ui
            .select(&format!("bone_{node_id}"), &mut bi, &bone_opts)
            .changed()
        {
            *bone = if bi == 0 {
                None
            } else {
                bone_list.get(bi - 1).map(|(id, _)| *id)
            };
        }
        if ui.icon_button(&format!("pick_{node_id}"), "edit", selection.is_some()) {
            if let Some(sel) = selection {
                *bone = Some(sel);
                *status_msg = Some("Bone ← selection".into());
            } else {
                *status_msg = Some("Select a bone in the viewport first".into());
            }
        }
    });
    let mut si = space.index();
    if ui
        .select(&format!("space_{node_id}"), &mut si, &space_opts)
        .changed()
    {
        *space = DriverSpace::ALL[si.min(2)];
    }
}

fn driver_editor_window(ui: &mut Ui, state: &mut AppState) {
    let Some(drv_id) = state.editing_driver else {
        return;
    };
    if !state.rig.drivers.iter().any(|d| d.id == drv_id) {
        state.editing_driver = None;
        return;
    }

    let title = state
        .rig
        .drivers
        .iter()
        .find(|d| d.id == drv_id)
        .map(|d| format!("Driver — {}", d.name))
        .unwrap_or_else(|| "Driver".into());

    let mut bone_list: Vec<(BoneId, String)> = state
        .rig
        .bones
        .iter()
        .map(|b| (b.id, b.name.clone()))
        .collect();
    bone_list.sort_by(|a, b| a.1.to_lowercase().cmp(&b.1.to_lowercase()));
    let bone_labels: Vec<String> = std::iter::once("(none)".into())
        .chain(bone_list.iter().map(|(_, n)| n.clone()))
        .collect();
    let selection = state.rig.selection;
    let active_mesh = state.rig.active_mesh;
    let active_shape = state.rig.active_shape;
    let active_shape_name = match (active_mesh, active_shape) {
        (Some(m), Some(i)) => state
            .scene
            .meshes
            .get(m)
            .and_then(|mesh| mesh.morph_targets.get(i))
            .map(|t| t.name.clone()),
        _ => None,
    };
    let ik_parent_bones: std::collections::HashSet<(u32, u32)> = state
        .rig
        .bones
        .iter()
        .filter(|b| crate::driver::bone_supports_ik(&state.rig, b.id))
        .map(|b| b.id.node.key())
        .collect();

    let mut open = true;
    let mut status_msg: Option<String> = None;
    let mut spawn_page = state.driver_spawn_page;

    ui.window(
        Window::new(&title)
            .open(&mut open)
            .resizable(true)
            .pos(Vec2::new(64.0, 48.0))
            .size(Vec2::new(780.0, 520.0)),
        |ui| {
            let Some(driver) = state.rig.drivers.iter_mut().find(|d| d.id == drv_id) else {
                return;
            };
            driver.apply_deletes();
            driver.apply_clones();

            ui.horizontal(|ui| {
                ui.label("RMB empty → spawn");
                let sel_n = driver.space.selected_nodes.len();
                if ui.icon_button("drv_clone", "copy", sel_n > 0) {
                    if sel_n > 0 {
                        driver.space.request_clone_nodes = driver.space.selected_nodes.clone();
                        status_msg = Some(format!("Clone {sel_n} node(s)"));
                    }
                }
                if ui.icon_button("drv_del", "delete", sel_n > 0) {
                    if sel_n > 0 {
                        let ids = driver.space.selected_nodes.clone();
                        for id in &ids {
                            driver.space.detach_node(id);
                        }
                        driver.space.request_delete_nodes.extend(ids);
                        status_msg = Some(format!("Delete {sel_n} node(s)"));
                    }
                }
            });
            ui.separator();

            let size = ui.available_size();
            let size = Vec2::new(size.x, (size.y - 4.0).max(160.0));

            let mut spawn_at: Option<(DriverNodeKind, Vec2)> = None;
            {
                let space = &mut driver.space;
                let nodes = &mut driver.nodes;

                ui.node_space(&format!("drv_graph_{drv_id}"), size, space, |ui| {
                    for n in nodes.iter_mut() {
                        let id = n.id.clone();
                        let title = n.title.clone();
                        let kind = n.kind;
                        let preview = n.preview.clone();
                        ui.node(&id, &title, &mut n.pos, |ui| {
                            draw_driver_node_body(
                                ui,
                                &id,
                                kind,
                                &mut n.bone,
                                &mut n.space,
                                &mut n.mesh,
                                &mut n.shape,
                                &mut n.floats,
                                &bone_list,
                                &bone_labels,
                                selection,
                                active_mesh,
                                active_shape,
                                active_shape_name.as_deref(),
                                &preview,
                                &mut status_msg,
                                &ik_parent_bones,
                            );
                        });
                    }
                });

                let bg = space.background_hovered;
                let world = space.context_world.unwrap_or(Vec2::new(80.0, 80.0));
                if space.context_menu_request {
                    spawn_page = 0;
                }
                let mut spawn_kind: Option<DriverNodeKind> = None;
                ui.context_menu(&format!("drv_spawn_{drv_id}"), bg, |ui| {
                    spawn_kind = driver_spawn_menu(ui, &mut spawn_page);
                });
                if let Some(kind) = spawn_kind {
                    spawn_at = Some((kind, world));
                    spawn_page = 0;
                }
            }
            if let Some((kind, world)) = spawn_at {
                driver.spawn_node(kind, world);
                status_msg = Some(format!("Spawned {}", kind.title()));
            }
        },
    );

    state.driver_spawn_page = spawn_page;

    if !open {
        state.editing_driver = None;
        state.driver_spawn_page = 0;
    }
    if let Some(msg) = status_msg {
        state.status = msg;
    }
}

fn draw_driver_node_body(
    ui: &mut Ui,
    node_id: &str,
    kind: DriverNodeKind,
    bone: &mut Option<BoneId>,
    space: &mut DriverSpace,
    mesh: &mut Option<mega_render::Handle<mega_render::Mesh>>,
    shape: &mut usize,
    floats: &mut [f32; 4],
    bone_list: &[(BoneId, String)],
    bone_labels: &[String],
    selection: Option<BoneId>,
    active_mesh: Option<mega_render::Handle<mega_render::Mesh>>,
    active_shape: Option<usize>,
    active_shape_name: Option<&str>,
    preview: &str,
    status_msg: &mut Option<String>,
    ik_parent_bones: &std::collections::HashSet<(u32, u32)>,
) {
    match kind {
        DriverNodeKind::BoneGet => {
            ui.node_port(NodePortSide::Output, "pos", port_type::VEC3);
            ui.node_port(NodePortSide::Output, "rot", port_type::QUAT);
            ui.node_port(NodePortSide::Output, "scale", port_type::VEC3);
            bone_picker_row(
                ui,
                node_id,
                bone,
                space,
                bone_list,
                bone_labels,
                selection,
                status_msg,
            );
            if !preview.is_empty() {
                ui.label(preview);
            }
        }
        DriverNodeKind::BoneSet => {
            ui.node_port(NodePortSide::Input, "pos", port_type::VEC3);
            ui.node_port(NodePortSide::Input, "rot", port_type::QUAT);
            ui.node_port(NodePortSide::Input, "scale", port_type::VEC3);
            bone_picker_row(
                ui,
                node_id,
                bone,
                space,
                bone_list,
                bone_labels,
                selection,
                status_msg,
            );
            if bone.is_some_and(|b| ik_parent_bones.contains(&b.node.key())) {
                ui.label("pre-IK (parent of IK chain)");
            }
            if !preview.is_empty() {
                ui.label(preview);
            }
        }
        DriverNodeKind::MorphSet => {
            ui.node_port(NodePortSide::Input, "in", port_type::FLOAT);
            let label = if mesh.is_some() {
                format!("shape #{shape}")
            } else {
                "(no morph)".into()
            };
            ui.label(&label);
            ui.horizontal(|ui| {
                if ui.icon_button(&format!("morph_{node_id}"), "edit", active_shape.is_some()) {
                    match (active_mesh, active_shape) {
                        (Some(m), Some(s)) => {
                            *mesh = Some(m);
                            *shape = s;
                            let name = active_shape_name.unwrap_or("shape");
                            *status_msg = Some(format!("Morph ← {name}"));
                        }
                        _ => *status_msg = Some("Select an active shape key first".into()),
                    }
                }
                ui.label("active shape");
            });
            if !preview.is_empty() {
                ui.label(&format!("w={preview}"));
            }
        }
        DriverNodeKind::Float => {
            ui.node_port(NodePortSide::Output, "value", port_type::FLOAT);
            ui.drag_float(&format!("c_{node_id}"), &mut floats[0], 0.1);
        }
        DriverNodeKind::Vec3 => {
            ui.node_port(NodePortSide::Output, "v", port_type::VEC3);
            let mut v = Vec3::new(floats[0], floats[1], floats[2]);
            if ui
                .vec3(&format!("v3_{node_id}"), &mut v, 0.1, Vec3::ZERO)
                .changed()
            {
                floats[0] = v.x;
                floats[1] = v.y;
                floats[2] = v.z;
            }
        }
        DriverNodeKind::QuatEuler => {
            ui.node_port(NodePortSide::Output, "q", port_type::QUAT);
            ui.label("X / Y / Z °");
            let mut v = Vec3::new(floats[0], floats[1], floats[2]);
            if ui
                .vec3(&format!("qe_{node_id}"), &mut v, 1.0, Vec3::ZERO)
                .changed()
            {
                floats[0] = v.x;
                floats[1] = v.y;
                floats[2] = v.z;
            }
        }
        DriverNodeKind::QuatToEuler => {
            ui.node_port(NodePortSide::Input, "q", port_type::QUAT);
            ui.label("X / Y / Z °");
            if !preview.is_empty() {
                ui.label(preview);
            }
            ui.node_port(NodePortSide::Output, "euler", port_type::VEC3);
            ui.node_port(NodePortSide::Output, "x", port_type::FLOAT);
            ui.node_port(NodePortSide::Output, "y", port_type::FLOAT);
            ui.node_port(NodePortSide::Output, "z", port_type::FLOAT);
        }
        DriverNodeKind::Remap => {
            ui.node_port(NodePortSide::Input, "in", port_type::FLOAT);
            ui.horizontal(|ui| {
                ui.label("From");
                ui.drag_float(&format!("from_{node_id}"), &mut floats[0], 0.5);
            });
            ui.horizontal(|ui| {
                ui.label("To");
                ui.drag_float(&format!("to_{node_id}"), &mut floats[1], 0.5);
            });
            if !preview.is_empty() {
                ui.label(&format!("= {preview}"));
            }
            ui.node_port(NodePortSide::Output, "out", port_type::FLOAT);
        }
        DriverNodeKind::MapRange => {
            ui.node_port(NodePortSide::Input, "in", port_type::FLOAT);
            ui.label("In");
            ui.horizontal(|ui| {
                ui.drag_float(&format!("if_{node_id}"), &mut floats[0], 0.5);
                ui.drag_float(&format!("it_{node_id}"), &mut floats[1], 0.5);
            });
            ui.label("Out");
            ui.horizontal(|ui| {
                ui.drag_float(&format!("of_{node_id}"), &mut floats[2], 0.5);
                ui.drag_float(&format!("ot_{node_id}"), &mut floats[3], 0.5);
            });
            if !preview.is_empty() {
                ui.label(&format!("= {preview}"));
            }
            ui.node_port(NodePortSide::Output, "out", port_type::FLOAT);
        }
        DriverNodeKind::Clamp => {
            ui.node_port(NodePortSide::Input, "in", port_type::FLOAT);
            ui.horizontal(|ui| {
                ui.label("Min");
                ui.drag_float(&format!("lo_{node_id}"), &mut floats[0], 0.05);
            });
            ui.horizontal(|ui| {
                ui.label("Max");
                ui.drag_float(&format!("hi_{node_id}"), &mut floats[1], 0.05);
            });
            if !preview.is_empty() {
                ui.label(&format!("= {preview}"));
            }
            ui.node_port(NodePortSide::Output, "out", port_type::FLOAT);
        }
        DriverNodeKind::Add | DriverNodeKind::Mul => {
            ui.node_port(NodePortSide::Input, "a", port_type::FLOAT);
            ui.node_port(NodePortSide::Input, "b", port_type::FLOAT);
            if !preview.is_empty() {
                ui.label(&format!("= {preview}"));
            }
            ui.node_port(NodePortSide::Output, "out", port_type::FLOAT);
        }
        DriverNodeKind::CombineVec3 => {
            ui.node_port(NodePortSide::Input, "x", port_type::FLOAT);
            ui.node_port(NodePortSide::Input, "y", port_type::FLOAT);
            ui.node_port(NodePortSide::Input, "z", port_type::FLOAT);
            if !preview.is_empty() {
                ui.label(preview);
            }
            ui.node_port(NodePortSide::Output, "out", port_type::VEC3);
        }
        DriverNodeKind::SplitVec3 => {
            ui.node_port(NodePortSide::Input, "v", port_type::VEC3);
            if !preview.is_empty() {
                ui.label(preview);
            }
            ui.node_port(NodePortSide::Output, "x", port_type::FLOAT);
            ui.node_port(NodePortSide::Output, "y", port_type::FLOAT);
            ui.node_port(NodePortSide::Output, "z", port_type::FLOAT);
        }
        DriverNodeKind::Vec3Add => {
            ui.node_port(NodePortSide::Input, "a", port_type::VEC3);
            ui.node_port(NodePortSide::Input, "b", port_type::VEC3);
            if !preview.is_empty() {
                ui.label(preview);
            }
            ui.node_port(NodePortSide::Output, "out", port_type::VEC3);
        }
        DriverNodeKind::Vec3Scale => {
            ui.node_port(NodePortSide::Input, "v", port_type::VEC3);
            ui.node_port(NodePortSide::Input, "s", port_type::FLOAT);
            if !preview.is_empty() {
                ui.label(preview);
            }
            ui.node_port(NodePortSide::Output, "out", port_type::VEC3);
        }
        DriverNodeKind::Vec3Length => {
            ui.node_port(NodePortSide::Input, "v", port_type::VEC3);
            if !preview.is_empty() {
                ui.label(preview);
            }
            ui.node_port(NodePortSide::Output, "out", port_type::FLOAT);
        }
        DriverNodeKind::Vec3Normalize => {
            ui.node_port(NodePortSide::Input, "v", port_type::VEC3);
            if !preview.is_empty() {
                ui.label(preview);
            }
            ui.node_port(NodePortSide::Output, "out", port_type::VEC3);
        }
        DriverNodeKind::QuatMul => {
            ui.node_port(NodePortSide::Input, "a", port_type::QUAT);
            ui.node_port(NodePortSide::Input, "b", port_type::QUAT);
            if !preview.is_empty() {
                ui.label(preview);
            }
            ui.node_port(NodePortSide::Output, "out", port_type::QUAT);
        }
        DriverNodeKind::QuatRotateVec => {
            ui.node_port(NodePortSide::Input, "q", port_type::QUAT);
            ui.node_port(NodePortSide::Input, "v", port_type::VEC3);
            if !preview.is_empty() {
                ui.label(preview);
            }
            ui.node_port(NodePortSide::Output, "out", port_type::VEC3);
        }
        DriverNodeKind::QuatInvert => {
            ui.node_port(NodePortSide::Input, "q", port_type::QUAT);
            if !preview.is_empty() {
                ui.label(preview);
            }
            ui.node_port(NodePortSide::Output, "out", port_type::QUAT);
        }
        DriverNodeKind::QuatScale => {
            ui.node_port(NodePortSide::Input, "q", port_type::QUAT);
            ui.node_port(NodePortSide::Input, "t", port_type::FLOAT);
            ui.horizontal(|ui| {
                ui.label("Weight");
                ui.drag_float(&format!("qs_{node_id}"), &mut floats[0], 0.01);
            });
            if !preview.is_empty() {
                ui.label(preview);
            }
            ui.node_port(NodePortSide::Output, "out", port_type::QUAT);
        }
        DriverNodeKind::QuatAngle => {
            ui.node_port(NodePortSide::Input, "a", port_type::QUAT);
            ui.node_port(NodePortSide::Input, "b", port_type::QUAT);
            if !preview.is_empty() {
                ui.label(&format!("{preview}°"));
            }
            ui.node_port(NodePortSide::Output, "out", port_type::FLOAT);
        }
    }
}

fn bone_panel(ui: &mut Ui, state: &mut AppState) {
    ui.label_styled(
        &format!("Bones ({})", state.rig.bones.len()),
        TextStyle {
            color: [0.9, 0.9, 0.9, 1.0],
            size: 14.0,
        },
    );

    if state.rig.mode == AppMode::Edit {
        ui.horizontal(|ui| {
            if ui.button("Extrude").clicked() {
                if let Some(sel) = state.rig.selection {
                    if state.rig.extrude_bone(&mut state.scene, sel).is_some() {
                        state.edit_bone = None;
                        state.status = "Extruded bone".into();
                    }
                } else {
                    state.status = "Select a bone to extrude".into();
                }
            }
            if ui.button("Delete").clicked() {
                if let Some(sel) = state.rig.selection {
                    state.rig.delete_bone_subtree(&mut state.scene, sel);
                    state.clear_drags();
                    state.edit_bone = None;
                    state.status = "Deleted bone subtree".into();
                }
            }
        });
    }

    ui.separator();
    ik_chains_section(ui, state);
    ui.separator();
    soft_chains_section(ui, state);
    ui.separator();

    let avail = ui.available_size();
    let roots = state.rig.roots();
    let prev = state.rig.selection;
    let mut tree_sel = prev.map(bone_tree_id);
    ui.scroll_area("bones_scroll", avail, ScrollAxes::Vertical, |ui| {
        if roots.is_empty() {
            ui.label("No bones yet.");
            if ui.button("New Skeleton").clicked() {
                state.new_skeleton();
            }
            return;
        }
        ui.tree_scope(&mut tree_sel, |ui| {
            for root in &roots {
                draw_bone_tree(ui, state, *root);
            }
        });
    });

    let new_sel = tree_sel.as_ref().and_then(|s| {
        state
            .rig
            .bones
            .iter()
            .find(|b| bone_tree_id(b.id) == *s)
            .map(|b| b.id)
    });
    if new_sel != prev {
        if let Some(id) = new_sel {
            state.rig.set_selection(id);
            state.edit_bone = None;
            state.status_from_selection();
        } else {
            state.rig.clear_selection();
            state.edit_bone = None;
            state.status_from_selection();
        }
    }
}

fn ik_chains_section(ui: &mut Ui, state: &mut AppState) {
    ui.label_styled(
        &format!("IK Chains ({})", state.rig.ik_chains.len()),
        TextStyle {
            color: [0.9, 0.9, 0.9, 1.0],
            size: 13.0,
        },
    );

    if state.rig.ik_chains.is_empty() {
        ui.label("Select tip (hand/foot) → Create IK");
        return;
    }

    let chains: Vec<(u32, String, bool)> = state
        .rig
        .ik_chains
        .iter()
        .map(|c| (c.id, c.name.clone(), c.enabled))
        .collect();

    for (id, name, enabled) in chains {
        ui.horizontal(|ui| {
            let label = if enabled {
                format!("● {name}")
            } else {
                format!("○ {name}")
            };
            if ui.button(&label).clicked() {
                state.rig.set_ik_enabled(id, !enabled);
                state.status = if !enabled {
                    format!("{name} enabled")
                } else {
                    format!("{name} muted")
                };
            }
            if ui.button("Sel").clicked() {
                if let Some(c) = state.rig.ik_chains.iter().find(|c| c.id == id) {
                    let t = c.target;
                    state.rig.set_selection(t);
                    state.edit_bone = None;
                    state.rig.tool = Tool::Translate;
                    state.status_from_selection();
                }
            }
            if ui.button("X").clicked() {
                state.rig.remove_ik_chain(&mut state.scene, id);
                state.clear_drags();
                state.edit_bone = None;
                state.status = format!("Removed {name}");
            }
        });
    }
}

fn soft_chains_section(ui: &mut Ui, state: &mut AppState) {
    ui.label_styled(
        &format!("Soft Chains ({})", state.rig.soft_chains.len()),
        TextStyle {
            color: [0.9, 0.9, 0.9, 1.0],
            size: 13.0,
        },
    );

    if state.rig.soft_chains.is_empty() {
        ui.label("Select bone → Create Soft (auto parent/children)");
        return;
    }

    let chains: Vec<(u32, String, bool)> = state
        .rig
        .soft_chains
        .iter()
        .map(|c| (c.id, c.name.clone(), c.enabled))
        .collect();

    for (id, name, enabled) in chains {
        ui.horizontal(|ui| {
            let label = if enabled {
                format!("● {name}")
            } else {
                format!("○ {name}")
            };
            if ui.button(&label).clicked() {
                state.rig.set_soft_enabled(id, !enabled);
                state.status = if !enabled {
                    format!("{name} enabled")
                } else {
                    format!("{name} muted")
                };
            }
            if ui.button("Sel").clicked() {
                if let Some(c) = state.rig.soft_chains.iter().find(|c| c.id == id) {
                    if let Some(&b) = c.bones.first() {
                        state.rig.set_selection(b);
                        state.edit_bone = None;
                        state.status_from_selection();
                    }
                }
            }
            if ui.button("X").clicked() {
                state.rig.remove_soft_chain(id);
                state.clear_drags();
                state.edit_bone = None;
                state.status = format!("Removed {name}");
            }
        });
    }
}

fn bone_tree_id(id: BoneId) -> String {
    format!("bone_{}_{}", id.node.index, id.node.generation)
}

fn draw_bone_tree(ui: &mut Ui, state: &AppState, id: BoneId) {
    let Some(b) = state.rig.bone(id) else {
        return;
    };
    let name = b.name.clone();
    let children = b.children.clone();
    let id_str = bone_tree_id(id);

    if children.is_empty() {
        ui.tree_leaf(&id_str, &name);
    } else {
        ui.tree_node(&id_str, &name, |ui| {
            for child in children {
                draw_bone_tree(ui, state, child);
            }
        });
    }
}

fn inspector_panel(ui: &mut Ui, state: &mut AppState) {
    state.sync_inspector_from_selection();
    ui.label_styled(
        "Inspector",
        TextStyle {
            color: [0.9, 0.9, 0.9, 1.0],
            size: 14.0,
        },
    );
    ui.separator();

    let avail = ui.available_size();
    ui.scroll_area("inspector_scroll", avail, ScrollAxes::Vertical, |ui| {
        inspector_panel_body(ui, state);
    });
}

fn inspector_panel_body(ui: &mut Ui, state: &mut AppState) {
    let Some(sel) = state.rig.selection else {
        ui.label("Nothing selected.");
        return;
    };
    let n_sel = state.rig.selected.len();
    if n_sel > 1 {
        ui.label(&format!("{n_sel} bones selected · editing active"));
    }
    let (name, deform, parent_name) = {
        let Some(b) = state.rig.bone(sel) else {
            ui.label("Invalid selection.");
            return;
        };
        let parent_name = b
            .parent
            .and_then(|p| state.rig.bone(p).map(|pb| pb.name.clone()));
        (b.name.clone(), b.deform, parent_name)
    };

    ui.label(&format!("Name: {name}"));
    ui.label(&format!("Deform: {deform}"));
    if let Some(kind) = state.rig.ik_control_kind(sel) {
        ui.label(match kind {
            IkControlKind::Target => "Role: IK Target (orange)",
            IkControlKind::Pole => "Role: IK Pole (purple)",
        });
    }
    if let Some(p) = parent_name {
        ui.label(&format!("Parent: {p}"));
    } else {
        ui.label("Parent: —");
    }
    ui.separator();

    // IK setup / manage
    {
        let existing = state.rig.ik_chain_for_tip(sel).map(|c| c.id);
        let is_control = state.rig.ik_control_kind(sel).is_some();
        if let Some(cid) = existing {
            ui.label("IK on this tip");
            if ui.button("Select IK target").clicked() {
                if let Some(c) = state.rig.ik_chains.iter().find(|c| c.id == cid) {
                    let t = c.target;
                    state.rig.set_selection(t);
                    state.edit_bone = None;
                    state.rig.tool = Tool::Translate;
                }
            }
            if ui.button("Remove IK").clicked() {
                state.rig.remove_ik_chain(&mut state.scene, cid);
                state.clear_drags();
                state.edit_bone = None;
                state.status = "IK removed".into();
            }
            ui.separator();
        } else if !is_control {
            ui.label("IK chain length (rotating bones)");
            ui.horizontal(|ui| {
                if ui.button("−").clicked() {
                    state.ik_create_length = state.ik_create_length.saturating_sub(1).max(1);
                }
                ui.label(&format!("{}", state.ik_create_length));
                if ui.button("+").clicked() {
                    state.ik_create_length = (state.ik_create_length + 1).min(32);
                }
            });
            if ui.button("Create IK").clicked() {
                let len = state.ik_create_length;
                match state.rig.create_ik_from_tip(&mut state.scene, sel, len) {
                    Ok(_) => {
                        state.edit_bone = None;
                        state.rig.tool = Tool::Translate;
                        if state.rig.mode != AppMode::Pose {
                            state.set_mode(AppMode::Pose);
                        }
                        state.status = format!(
                            "IK ×{len} · Move target, Rotate target = hand, Pole = elbow"
                        );
                    }
                    Err(e) => {
                        state.status = e.into();
                    }
                }
            }
            ui.separator();
        }
    }

    // Soft setup / params
    {
        let soft_id = state.rig.soft_chain_containing(sel).map(|c| c.id);
        let is_control = state.rig.ik_control_kind(sel).is_some();
        if let Some(cid) = soft_id {
            ui.label("Soft chain on this bone");
            if let Some(c) = state.rig.soft_chains.iter_mut().find(|c| c.id == cid) {
                ui.label("Gravity (m/s²)");
                if ui
                    .slider(&format!("soft_g_{cid}"), &mut c.gravity, 0.0..=40.0)
                    .changed()
                {
                    c.initialized = false;
                }
                ui.label("Stiffness (1/s²)");
                let _ = ui.slider(&format!("soft_k_{cid}"), &mut c.stiffness, 0.0..=200.0);
                ui.label("Damping (1/s)");
                let _ = ui.slider(&format!("soft_d_{cid}"), &mut c.damping, 0.0..=40.0);
                ui.label("Inertia (move lag)");
                let _ = ui.slider(&format!("soft_i_{cid}"), &mut c.inertia, 0.0..=20.0);
                ui.label("Max angle (°)");
                let mut deg = c.max_angle.to_degrees();
                if ui
                    .slider(&format!("soft_a_{cid}"), &mut deg, 5.0..=170.0)
                    .changed()
                {
                    c.max_angle = deg.to_radians();
                }
            }
            if ui.button("Remove Soft").clicked() {
                state.rig.remove_soft_chain(cid);
                state.clear_drags();
                state.edit_bone = None;
                state.status = "Soft removed".into();
            }
            ui.separator();
        } else if !is_control {
            ui.label("Soft: select any bone in a linear chain");
            if ui.button("Create Soft").clicked() {
                match state.rig.create_soft_from_bone(&state.scene, sel) {
                    Ok(_) => {
                        state.edit_bone = None;
                        if state.rig.mode != AppMode::Pose {
                            state.set_mode(AppMode::Pose);
                        }
                        state.status =
                            "Soft created · Pose mode: gravity + support plane".into();
                    }
                    Err(e) => state.status = e.into(),
                }
            }
            ui.separator();
        }
    }

    // Capsule collider on selected bone
    {
        let is_control = state.rig.ik_control_kind(sel).is_some();
        if !is_control {
            let col_id = state.rig.collider_on_bone(sel).map(|c| c.id);
            if let Some(cid) = col_id {
                ui.label("Capsule collider");
                if let Some(c) = state.rig.colliders.iter_mut().find(|c| c.id == cid) {
                    let _ = ui.checkbox(&format!("Enabled##col_en_{cid}"), &mut c.enabled);
                    ui.label("Radius");
                    if ui
                        .drag_float(&format!("col_r_{cid}"), &mut c.radius, 0.001)
                        .changed()
                    {
                        c.radius = c.radius.clamp(0.001, 0.5);
                    }
                    ui.label("Length");
                    let mut len = c.length();
                    let mut off = c.axis_offset();
                    if ui
                        .drag_float(&format!("col_l_{cid}"), &mut len, 0.001)
                        .changed()
                    {
                        c.set_length_offset(len.clamp(0.001, 1.0), off.clamp(-0.5, 0.5));
                    }
                    ui.label("Offset along axis");
                    len = c.length();
                    off = c.axis_offset();
                    if ui
                        .slider(&format!("col_o_{cid}"), &mut off, -0.5..=0.5)
                        .changed()
                    {
                        c.set_length_offset(len.max(0.001), off);
                    }
                    ui.label("Softness (0=jelly, 1=firmer)");
                    let _ = ui.slider(&format!("col_s_{cid}"), &mut c.softness, 0.0..=1.0);
                }
                if ui.button("Remove Collider").clicked() {
                    state.rig.remove_collider(cid);
                    state.status = "Collider removed".into();
                }
            } else {
                ui.label("Collider: capsule volume on this bone");
                if ui.button("Add Capsule Collider").clicked() {
                    match state.rig.create_capsule_collider(&state.scene, sel) {
                        Ok(_) => state.status = "Capsule collider added".into(),
                        Err(e) => state.status = e.into(),
                    }
                }
            }
            ui.separator();
        }
    }

    if state.rig.mode == AppMode::Edit
        || state.rig.ik_control_kind(sel).is_some()
    {
        ui.label("Local translation");
        if ui
            .vec3("translation", &mut state.edit_translation, 0.01, Vec3::ZERO)
            .changed()
        {
            state.apply_inspector_translation();
        }
        ui.separator();
    }

    ui.label("Local rotation (YXZ °)");
    if ui
        .vec3("euler", &mut state.edit_euler_deg, 0.5, Vec3::ZERO)
        .changed()
    {
        state.apply_inspector_euler();
    }
    if state.rig.mode == AppMode::Pose {
        if let Some(n) = state.scene.nodes.get(sel.node) {
            let t = n.local.translation;
            ui.label(&format!(
                "Translation: {:.3}, {:.3}, {:.3}",
                t.x, t.y, t.z
            ));
        }
    }
    ui.separator();
    if ui.button("Reset bone to bind").clicked() {
        state.rig.reset_selected(&mut state.scene);
        state.edit_bone = None;
    }
}
