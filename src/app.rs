//! Application state, UI, and viewport tools.

use glam::{Vec2, Vec3};
use mega_render::{Scene, Visualizer, WgpuVisualizer};
use mega_ui::{DockNode, DockState, ScrollAxes, TextStyle, Ui};

use crate::framework::{Demo, UiCtx, SCENE_TEX};
use crate::gizmo::{self, Axis, RotateDrag};
use crate::pick;
use crate::rig::{empty_scene, BoneId, RigDocument, Tool};

pub struct AppState {
    pub scene: Scene,
    pub rig: RigDocument,
    pub status: String,
    pub rotate_drag: Option<RotateDrag>,
    pub gizmo_hover: Option<Axis>,
    /// Euler degrees shown in inspector (synced from selection).
    pub edit_euler_deg: Vec3,
    pub edit_bone: Option<BoneId>,
    /// Host should re-read orbit cam from `scene.camera`.
    pub resync_camera: bool,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            scene: empty_scene(),
            rig: RigDocument::default(),
            status: "Open a glTF / GLB · RMB orbit · MMB pan · wheel zoom".into(),
            rotate_drag: None,
            gizmo_hover: None,
            edit_euler_deg: Vec3::ZERO,
            edit_bone: None,
            resync_camera: false,
        }
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
                self.rotate_drag = None;
                self.gizmo_hover = None;
                self.edit_bone = None;
                self.resync_camera = true;
                let with_parent = self.rig.bones.iter().filter(|b| b.parent.is_some()).count();
                self.status = format!(
                    "Loaded {} · {} bones ({} with parent)",
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

    pub fn sync_inspector_from_selection(&mut self) {
        let sel = self.rig.selection;
        if self.edit_bone != sel {
            self.edit_bone = sel;
            if let Some(id) = sel {
                if let Some(n) = self.scene.nodes.get(id.node) {
                    let (y, x, z) = n.local.rotation.to_euler(glam::EulerRot::YXZ);
                    self.edit_euler_deg =
                        Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees());
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
    }

    pub fn gizmo_radius(&self) -> f32 {
        let Some(sel) = self.rig.selection else {
            return 0.15;
        };
        gizmo::gizmo_radius(
            &self.scene,
            sel,
            self.rig.average_bone_length(&self.scene),
        )
    }
}

pub struct PointerFrame {
    pub pos: Vec2,
    pub pressed: bool,
    pub down: bool,
    pub released: bool,
}

/// Viewport select / gizmo after UI laid out `viewport_rect`.
pub fn handle_tools(state: &mut AppState, pointer: &PointerFrame, ui_wants_mouse: bool) {
    let rect = state.rig.viewport_rect;
    let over = rect.width() > 1.0 && rect.height() > 1.0 && rect.contains(pointer.pos);

    if pointer.released {
        state.rotate_drag = None;
    }

    // Hover gizmo rings.
    state.gizmo_hover = None;
    if over && state.rig.tool == Tool::Rotate {
        if let Some(sel) = state.rig.selection {
            let radius = state.gizmo_radius();
            if let Some(ref drag) = state.rotate_drag {
                state.gizmo_hover = Some(drag.axis);
            } else {
                state.gizmo_hover = gizmo::hover_axis(&state.scene, sel, rect, pointer.pos, radius);
            }
        }
    }

    if state.rotate_drag.is_some() {
        if pointer.down {
            // Split borrows: apply using a clone of drag params.
            let drag = state.rotate_drag.clone().unwrap();
            gizmo::apply_rotate(&mut state.scene, &drag, rect, pointer.pos);
            if let Some(n) = state.scene.nodes.get(drag.bone.node) {
                let (y, x, z) = n.local.rotation.to_euler(glam::EulerRot::YXZ);
                state.edit_euler_deg =
                    Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees());
                state.edit_bone = Some(drag.bone);
            }
        }
        return;
    }

    if ui_wants_mouse || !over || !pointer.pressed {
        return;
    }

    // Gizmo ring first.
    if state.rig.tool == Tool::Rotate {
        if let Some(sel) = state.rig.selection {
            let radius = state.gizmo_radius();
            if let Some(drag) =
                gizmo::begin_rotate(&state.scene, sel, rect, pointer.pos, radius)
            {
                state.rotate_drag = Some(drag);
                return;
            }
        }
    }

    // Pick bone.
    let Some(ray) = pick::ray_from_viewport(&state.scene, rect, pointer.pos) else {
        return;
    };
    if let Some(id) = pick::pick_bone(&state.scene, &state.rig, &ray) {
        state.rig.selection = Some(id);
        state.edit_bone = None;
        if let Some(name) = state.rig.bone(id).map(|b| b.name.clone()) {
            state.status = format!("Selected: {name}");
        }
    } else if state.gizmo_hover.is_none() {
        state.rig.selection = None;
        state.edit_bone = None;
    }
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
        state.rotate_drag.is_some()
    }

    fn build_ui(ui: &mut Ui, ctx: &mut UiCtx<'_>) -> bool {
        let status_h = 24.0 * ui.scale();
        let menu_h = 28.0 * ui.scale();
        let dock_h = (ctx.window_size.y - status_h - menu_h).max(1.0);
        let dock_size = Vec2::new(ctx.window_size.x, dock_h);

        ui.menu_bar(|ui| {
            ui.menu("File", |ui| {
                if ui.menu_item_icon("folder_open", "Open…").clicked() {
                    ctx.state.open_dialog();
                }
                if ui.menu_item("Reset pose").clicked() {
                    ctx.state.rig.reset_pose(&mut ctx.state.scene);
                    ctx.state.edit_bone = None;
                    ctx.state.status = "Pose reset to bind.".into();
                }
            });
            ui.menu("View", |ui| {
                let mut skel = ctx.state.rig.show_skeleton;
                let mut mesh = ctx.state.rig.show_mesh;
                if ui.checkbox("Skeleton", &mut skel).changed() {
                    ctx.state.rig.show_skeleton = skel;
                }
                if ui.checkbox("Mesh", &mut mesh).changed() {
                    ctx.state.rig.show_mesh = mesh;
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
                let path = state
                    .rig
                    .source_path
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .and_then(|s| s.to_str())
                    .unwrap_or("(no model)");
                ui.label_styled(
                    &format!("{path}  ·  {:?}", state.rig.tool),
                    TextStyle {
                        color: [0.85, 0.85, 0.85, 1.0],
                        size: 13.0,
                    },
                );
                ui.horizontal(|ui| {
                    if ui.button("Open").clicked() {
                        state.open_dialog();
                    }
                    let sel = state.rig.tool == Tool::Select;
                    let rot = state.rig.tool == Tool::Rotate;
                    if ui.button(if sel { "[Select]" } else { "Select" }).clicked() {
                        state.rig.tool = Tool::Select;
                        state.rotate_drag = None;
                    }
                    if ui.button(if rot { "[Rotate]" } else { "Rotate" }).clicked() {
                        state.rig.tool = Tool::Rotate;
                    }
                    if ui.button("Reset bone").clicked() {
                        state.rig.reset_selected(&mut state.scene);
                        state.edit_bone = None;
                    }
                });
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

        ui.label_styled(
            &format!("{:.0} fps · {:.1} ms · {}", fps, frame_ms, state.status),
            TextStyle {
                color: [0.65, 0.68, 0.72, 1.0],
                size: 12.0,
            },
        );

        keep || state.rotate_drag.is_some()
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
    ui.separator();
    let avail = ui.available_size();
    ui.scroll_area("bones_scroll", avail, ScrollAxes::Vertical, |ui| {
        let roots = state.rig.roots();
        if roots.is_empty() {
            ui.label("No bones — open a skinned glTF.");
            return;
        }
        for root in roots {
            draw_bone_tree(ui, state, root);
        }
    });
}

fn draw_bone_tree(ui: &mut Ui, state: &mut AppState, id: BoneId) {
    let (name, children, selected) = {
        let Some(b) = state.rig.bone(id) else {
            return;
        };
        (
            b.name.clone(),
            b.children.clone(),
            state.rig.selection == Some(id),
        )
    };
    let label = if selected {
        format!("▸ {name}")
    } else {
        name.clone()
    };
    let id_str = format!("bone_{}_{}", id.node.index, id.node.generation);

    if children.is_empty() {
        if ui.button(&label).clicked() {
            state.rig.selection = Some(id);
            state.edit_bone = None;
            state.status = format!("Selected: {name}");
        }
    } else {
        ui.tree_node(&id_str, &label, |ui| {
            if ui.button(&format!("select · {name}")).clicked() {
                state.rig.selection = Some(id);
                state.edit_bone = None;
                state.status = format!("Selected: {name}");
            }
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
    ui.label("Local rotation (YXZ °)");
    if ui
        .vec3("euler", &mut state.edit_euler_deg, 0.5, Vec3::ZERO)
        .changed()
    {
        state.apply_inspector_euler();
    }
    if let Some(n) = state.scene.nodes.get(sel.node) {
        let t = n.local.translation;
        ui.label(&format!(
            "Translation: {:.3}, {:.3}, {:.3}",
            t.x, t.y, t.z
        ));
    }
    ui.separator();
    if ui.button("Reset bone to bind").clicked() {
        state.rig.reset_selected(&mut state.scene);
        state.edit_bone = None;
    }
}
