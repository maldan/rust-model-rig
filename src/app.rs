//! Application state, UI, and viewport tools.

use glam::{Vec2, Vec3};
use mega_render::{GizmoAxis, Scene, Visualizer, WgpuVisualizer};
use mega_ui::{DockNode, DockState, ScrollAxes, TextStyle, Ui};

use crate::framework::{Demo, UiCtx, SCENE_TEX};
use crate::gizmo::{self, RotateDrag, TranslateDrag};
use crate::pick;
use crate::rig::{empty_scene, AppMode, BoneId, RigDocument, Tool};

pub struct AppState {
    pub scene: Scene,
    pub rig: RigDocument,
    pub status: String,
    pub rotate_drag: Option<RotateDrag>,
    pub translate_drag: Option<TranslateDrag>,
    pub marquee: Option<MarqueeDrag>,
    pub gizmo_hover: Option<GizmoAxis>,
    /// Euler degrees shown in inspector (synced from selection).
    pub edit_euler_deg: Vec3,
    pub edit_translation: Vec3,
    pub edit_bone: Option<BoneId>,
    /// Host should re-read orbit cam from `scene.camera`.
    pub resync_camera: bool,
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
            marquee: None,
            gizmo_hover: None,
            edit_euler_deg: Vec3::ZERO,
            edit_translation: Vec3::ZERO,
            edit_bone: None,
            resync_camera: false,
        }
    }

    pub fn has_drag(&self) -> bool {
        self.rotate_drag.is_some() || self.translate_drag.is_some() || self.marquee.is_some()
    }

    pub fn clear_drags(&mut self) {
        self.rotate_drag = None;
        self.translate_drag = None;
        self.marquee = None;
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
                self.clear_drags();
                self.gizmo_hover = None;
                self.edit_bone = None;
                self.resync_camera = true;
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
        self.clear_drags();
        self.gizmo_hover = None;
        self.edit_bone = None;
        self.resync_camera = true;
        self.status = "New skeleton · Edit mode · Extrude (E) / Add / Translate".into();
    }

    pub fn set_mode(&mut self, mode: AppMode) {
        self.rig.set_mode(&mut self.scene, mode);
        self.clear_drags();
        self.edit_bone = None;
        self.status = match mode {
            AppMode::Edit => "Edit mode — rest pose / hierarchy".into(),
            AppMode::Pose => "Pose mode — FK rotate".into(),
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
        state.clear_drags();
    }

    // Update marquee while dragging.
    if let Some(ref mut m) = state.marquee {
        if pointer.down {
            m.current = pointer.pos;
        }
        return;
    }

    // Hover gizmo handles.
    state.gizmo_hover = None;
    if over && state.rig.selection.is_some() && !state.has_drag() {
        if let (Some(sel), Some(pivot)) = (state.rig.selection, state.gizmo_pivot()) {
            let radius = state.gizmo_radius();
            state.gizmo_hover = match state.rig.tool {
                Tool::Rotate => {
                    gizmo::hover_axis(&state.scene, sel, pivot, rect, pointer.pos, radius)
                }
                Tool::Translate => gizmo::hover_translate_axis(
                    &state.scene,
                    sel,
                    pivot,
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
    }

    if state.rotate_drag.is_some() {
        if pointer.down {
            let bone = {
                let drag = state.rotate_drag.as_mut().unwrap();
                gizmo::apply_rotate(&mut state.scene, drag, rect, pointer.pos);
                drag.bone
            };
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
            state.sync_inspector_live(bone);
        }
        return;
    }

    if ui_wants_mouse || !over || !pointer.pressed {
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
                    rect,
                    pointer.pos,
                    radius,
                ) {
                    state.rotate_drag = Some(drag);
                    return;
                }
            }
            Tool::Translate => {
                if let Some(drag) = gizmo::begin_translate(
                    &state.scene,
                    sel,
                    &roots,
                    pivot,
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
            Tool::Select => {}
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
            DockNode::leaf(&["Bones"]),
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
        state.has_drag()
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
            });
            ui.menu("View", |ui| {
                let skel = ctx.state.rig.show_skeleton;
                let mesh = ctx.state.rig.show_mesh;
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
            "Inspector" => {
                inspector_panel(ui, state);
                keep |= state.edit_bone.is_some();
            }
            _ => {}
        });

        ui.status_bar(|ui| {
            ui.label_styled(
                &format!("{:.0} fps · {:.1} ms · {}", fps, frame_ms, state.status),
                TextStyle {
                    color: [0.65, 0.68, 0.72, 1.0],
                    size: 12.0,
                },
            );
        });

        keep || state.has_drag()
    }
}

fn viewport_toolbar(ui: &mut Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        let mut mode_i = match state.rig.mode {
            AppMode::Edit => 0,
            AppMode::Pose => 1,
        };
        if ui.toggle("app_mode", &mut mode_i, &["Edit", "Pose"]).changed() {
            state.set_mode(if mode_i == 0 {
                AppMode::Edit
            } else {
                AppMode::Pose
            });
        }

        match state.rig.mode {
            AppMode::Edit => {
                let mut tool_i = match state.rig.tool {
                    Tool::Select => 0,
                    Tool::Translate => 1,
                    Tool::Rotate => 2,
                    Tool::AddBone => 3,
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
                    _ => 0,
                };
                if ui
                    .toggle("tool_pose", &mut tool_i, &["Select", "Move", "Rotate"])
                    .changed()
                {
                    state.rig.tool = match tool_i {
                        1 => Tool::Translate,
                        2 => Tool::Rotate,
                        _ => Tool::Select,
                    };
                    state.clear_drags();
                }
            }
        }
    });

    let hint = match (state.rig.mode, state.rig.tool) {
        (AppMode::Edit, Tool::AddBone) => "Click bone to extrude · empty click places root",
        (AppMode::Edit, Tool::Translate) => "Drag arrows · Ctrl+drag box select · E extrude",
        (AppMode::Edit, Tool::Rotate) => "Drag rings · Ctrl+drag box select",
        (AppMode::Edit, _) => "Select · Ctrl+drag box · Ctrl+click toggle · Tab → Pose",
        (AppMode::Pose, Tool::Translate) => "Drag arrows · Ctrl+drag box select",
        (AppMode::Pose, Tool::Rotate) => "Drag rings · Ctrl+drag box select",
        (AppMode::Pose, _) => "Select · Ctrl+drag box · Ctrl+click toggle · Tab → Edit",
    };
    ui.label_styled(
        hint,
        TextStyle {
            color: [0.55, 0.58, 0.62, 1.0],
            size: 12.0,
        },
    );
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
    if let Some(p) = parent_name {
        ui.label(&format!("Parent: {p}"));
    } else {
        ui.label("Parent: —");
    }
    ui.separator();

    if state.rig.mode == AppMode::Edit {
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
