//! Winit + wgpu host for model-rig (adapted from mega-render examples/framework).

use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{Vec2, Vec3};
use mega_render::{
    view_gizmo, Camera, DebugView, InputFrame, PostProcessSettings, Projection, Scene,
    ShadowSettings, Visualizer, WgpuVisualizer,
};
use mega_ui::wgpu::{DrawStats, UiRenderer};
use mega_ui::{CursorIcon, DockState, Ui, UiInput};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::app::{handle_tools, AppState, PointerFrame};
use crate::gizmo;
use crate::rig::{draw_rig_debug, Tool};

/// Texture slot for the 3D viewport (`ui.texture(SCENE_TEX, …)`).
pub const SCENE_TEX: u32 = 0;

/// Per-frame UI context passed to [`Demo::build_ui`].
#[allow(dead_code)] // host fills extra fields for future panels
pub struct UiCtx<'a> {
    pub state: &'a mut AppState,
    pub post: &'a mut PostProcessSettings,
    pub shadow: &'a mut ShadowSettings,
    pub debug_view: &'a mut DebugView,
    pub dock: &'a mut DockState,
    /// Full window size in pixels (for dock layout).
    pub window_size: Vec2,
    /// Set this to the scene pane size in **pixels** (from `ui.available_size()`).
    pub viewport_size: &'a mut Vec2,
    pub dt: f32,
    /// Frames-per-second averaged over the last ~1 second.
    pub fps: f32,
    /// Mean frame time in milliseconds over the last ~1 second.
    pub frame_ms: f32,
    pub stats: DrawStats,
}

/// App hook for the host loop.
pub trait Demo {
    fn title() -> &'static str;
    fn window_size() -> (f64, f64) {
        (1440.0, 900.0)
    }
    fn build_state() -> AppState;
    fn configure(_visualizer: &mut WgpuVisualizer) {}
    fn init_ui(ui: &mut Ui) {
        ui.load_builtin_icons();
    }
    /// Optional per-frame update. Return `true` to keep animating.
    fn update(_state: &mut AppState, _dt: f32) -> bool {
        false
    }
    /// Build dock / widgets. Return `true` to keep redrawing.
    fn build_ui(ui: &mut Ui, ctx: &mut UiCtx<'_>) -> bool;
}

struct OrbitCam {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    distance: f32,
    orbiting: bool,
    panning: bool,
    /// Smooth snap toward (yaw, pitch).
    snap: Option<OrbitSnap>,
    projection: Projection,
    /// Blender-style: orbit after an axis snap returns to perspective.
    auto_perspective: bool,
    /// Set by view-gizmo snap; cleared when orbiting restores perspective.
    ortho_from_view: bool,
}

struct OrbitSnap {
    from_yaw: f32,
    from_pitch: f32,
    to_yaw: f32,
    to_pitch: f32,
    t: f32,
}

impl OrbitCam {
    fn from_scene(scene: &Scene) -> Self {
        let mut cam = Self {
            target: scene.camera.target,
            yaw: std::f32::consts::PI, // sit on −Z, look along +Z (forward)
            pitch: 0.35,
            distance: 5.0,
            orbiting: false,
            panning: false,
            snap: None,
            projection: scene.camera.projection,
            auto_perspective: true,
            ortho_from_view: false,
        };
        cam.sync_from_camera(&scene.camera);
        cam
    }

    fn sync_from_camera(&mut self, camera: &Camera) {
        self.target = camera.target;
        let offset = camera.eye - camera.target;
        self.distance = offset.length().max(0.05);
        let d = offset.normalize_or_zero();
        if d.length_squared() < 1e-8 {
            return;
        }
        self.pitch = d.y.clamp(-1.0, 1.0).asin();
        self.yaw = d.x.atan2(d.z);
        self.projection = camera.projection;
    }

    fn eye(&self) -> Vec3 {
        let pitch = self.pitch.clamp(-1.55, 1.55);
        self.target
            + Vec3::new(
                self.distance * self.yaw.sin() * pitch.cos(),
                self.distance * pitch.sin(),
                self.distance * self.yaw.cos() * pitch.cos(),
            )
    }

    fn active(&self) -> bool {
        self.orbiting || self.panning || self.snap.is_some()
    }

    fn add_orbit(&mut self, dx: f32, dy: f32) {
        self.snap = None;
        if self.auto_perspective && self.ortho_from_view {
            self.projection = Projection::Perspective;
            self.ortho_from_view = false;
        }
        const SENS: f32 = 0.005;
        self.yaw += dx * SENS;
        // DeviceEvent: +dy = mouse moved down. Drag down → look from below (pitch down).
        self.pitch = (self.pitch + dy * SENS).clamp(-1.55, 1.55);
    }

    fn add_pan(&mut self, dx: f32, dy: f32) {
        self.snap = None;
        let eye = self.eye();
        let forward = (self.target - eye).normalize_or_zero();
        let right = Vec3::Y.cross(forward).normalize_or_zero();
        let up = forward.cross(right).normalize_or_zero();
        let sens = self.distance * 0.0018;
        self.target += right * (-dx * sens) + up * (dy * sens);
    }

    fn add_zoom(&mut self, scroll_y: f32) {
        // scroll_y > 0 = wheel up → zoom in
        let factor = (1.0 - scroll_y * 0.0015).clamp(0.5, 1.5);
        self.distance = (self.distance * factor).clamp(0.05, 500.0);
    }

    /// Place camera on `dir` (world), looking at target. Smooth short tween + ortho.
    fn snap_to_dir(&mut self, dir: Vec3) {
        let d = dir.normalize_or_zero();
        if d.length_squared() < 1e-8 {
            return;
        }
        let to_pitch = d.y.clamp(-1.0, 1.0).asin().clamp(-1.55, 1.55);
        let to_yaw = d.x.atan2(d.z);
        self.snap = Some(OrbitSnap {
            from_yaw: self.yaw,
            from_pitch: self.pitch,
            to_yaw,
            to_pitch,
            t: 0.0,
        });
        self.projection = Projection::Orthographic;
        self.ortho_from_view = true;
    }

    fn tick_snap(&mut self, dt: f32) {
        let Some(snap) = self.snap.as_mut() else {
            return;
        };
        snap.t = (snap.t + dt / 0.22).min(1.0);
        let t = snap.t;
        // Smoothstep.
        let s = t * t * (3.0 - 2.0 * t);
        self.yaw = lerp_angle(snap.from_yaw, snap.to_yaw, s);
        self.pitch = snap.from_pitch + (snap.to_pitch - snap.from_pitch) * s;
        if t >= 1.0 {
            self.yaw = snap.to_yaw;
            self.pitch = snap.to_pitch;
            self.snap = None;
        }
    }

    fn apply(&self, scene: &mut Scene) {
        let eye = self.eye();
        let focus_distance = scene.camera.focus_distance;
        let focus_target = scene.camera.focus_target;
        let focus_smooth = scene.camera.focus_smooth;
        let f_stop = scene.camera.f_stop;
        let fov_y = scene.camera.fov_y;
        let near = scene.camera.near;
        let far = (self.distance * 20.0).max(50.0);
        let near_max = (self.distance * 0.05).max(0.01);
        scene.camera = Camera::look_at(eye, self.target);
        scene.camera.fov_y = fov_y;
        scene.camera.projection = self.projection;
        scene.camera.ortho_size = Camera::ortho_size_from_distance(self.distance, fov_y);
        scene.camera.focus_distance = focus_distance;
        scene.camera.focus_target = focus_target;
        scene.camera.focus_smooth = focus_smooth;
        scene.camera.f_stop = f_stop;
        scene.camera.near = near.clamp(0.01, near_max);
        scene.camera.far = far;
    }
}

fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let mut d = b - a;
    while d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    }
    while d < -std::f32::consts::PI {
        d += std::f32::consts::TAU;
    }
    a + d * t
}

struct SceneTarget {
    _texture: wgpu::Texture,
    render_view: wgpu::TextureView,
    size: (u32, u32),
}

impl SceneTarget {
    fn ensure(device: &wgpu::Device, ui: &mut UiRenderer, size: (u32, u32)) -> Self {
        let (w, h) = (size.0.max(1), size.1.max(1));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("demo scene color"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let render_view = texture.create_view(&Default::default());
        ui.bind_texture_view(device, SCENE_TEX, texture.create_view(&Default::default()));
        Self {
            _texture: texture,
            render_view,
            size: (w, h),
        }
    }

    fn resize(&mut self, device: &wgpu::Device, ui: &mut UiRenderer, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);
        if self.size == (w, h) {
            return;
        }
        *self = Self::ensure(device, ui, (w, h));
    }
}

#[derive(Default)]
struct FrameInput {
    mouse_pos: Vec2,
    mouse_down: bool,
    mouse_pressed: bool,
    mouse_released: bool,
    mouse_right_down: bool,
    mouse_right_pressed: bool,
    mouse_right_released: bool,
    mouse_middle_down: bool,
    mouse_middle_pressed: bool,
    mouse_middle_released: bool,
    scroll_delta: Vec2,
    text: String,
    key_backspace: bool,
    key_enter: bool,
    key_left: bool,
    key_right: bool,
    key_up: bool,
    key_down: bool,
    key_home: bool,
    key_end: bool,
    key_shift: bool,
    key_ctrl: bool,
    key_copy: bool,
    key_paste: bool,
    key_cut: bool,
    key_select_all: bool,
    modifiers: winit::keyboard::ModifiersState,
    clipboard_paste: String,
}

impl FrameInput {
    fn clear_edges(&mut self) {
        self.mouse_pressed = false;
        self.mouse_released = false;
        self.mouse_right_pressed = false;
        self.mouse_right_released = false;
        self.mouse_middle_pressed = false;
        self.mouse_middle_released = false;
        self.scroll_delta = Vec2::ZERO;
        self.text.clear();
        self.key_backspace = false;
        self.key_enter = false;
        self.key_left = false;
        self.key_right = false;
        self.key_up = false;
        self.key_down = false;
        self.key_home = false;
        self.key_end = false;
        self.key_copy = false;
        self.key_paste = false;
        self.key_cut = false;
        self.key_select_all = false;
        self.clipboard_paste.clear();
    }

    fn shortcut_mod(&self) -> bool {
        self.modifiers.control_key() || self.modifiers.super_key()
    }

    fn to_ui(&self, viewport: Vec2, dt: f32) -> UiInput {
        UiInput {
            mouse_pos: self.mouse_pos,
            mouse_down: self.mouse_down,
            mouse_pressed: self.mouse_pressed,
            mouse_released: self.mouse_released,
            mouse_right_down: self.mouse_right_down,
            mouse_right_pressed: self.mouse_right_pressed,
            mouse_right_released: self.mouse_right_released,
            viewport,
            scroll_delta: self.scroll_delta,
            dt,
            text: self.text.clone(),
            key_backspace: self.key_backspace,
            key_enter: self.key_enter,
            key_left: self.key_left,
            key_right: self.key_right,
            key_up: self.key_up,
            key_down: self.key_down,
            key_home: self.key_home,
            key_end: self.key_end,
            key_shift: self.key_shift,
            key_ctrl: self.key_ctrl || self.modifiers.super_key(),
            key_copy: self.key_copy,
            key_paste: self.key_paste,
            key_cut: self.key_cut,
            key_select_all: self.key_select_all,
            clipboard: self.clipboard_paste.clone(),
        }
    }
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    visualizer: WgpuVisualizer,
    ui_renderer: UiRenderer,
    scene_target: SceneTarget,
}

pub struct Host<D: Demo> {
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    state: AppState,
    orbit: OrbitCam,
    ui: Ui,
    dock: DockState,
    input: FrameInput,
    last_frame: Instant,
    fps_accum_dt: f32,
    fps_frames: u32,
    fps: f32,
    frame_ms: f32,
    animating: bool,
    cursor: CursorIcon,
    draw_stats: DrawStats,
    clipboard: Option<arboard::Clipboard>,
    want_capture_mouse: bool,
    want_capture_keyboard: bool,
    viewport_size: Vec2,
    _marker: std::marker::PhantomData<D>,
}

impl<D: Demo> Host<D> {
    fn new(dock: DockState) -> Self {
        let state = D::build_state();
        let orbit = OrbitCam::from_scene(&state.scene);
        let mut ui = Ui::new();
        D::init_ui(&mut ui);
        Self {
            window: None,
            gpu: None,
            state,
            orbit,
            ui,
            dock,
            input: FrameInput::default(),
            last_frame: Instant::now(),
            fps_accum_dt: 0.0,
            fps_frames: 0,
            fps: 0.0,
            frame_ms: 0.0,
            animating: true,
            cursor: CursorIcon::Default,
            draw_stats: DrawStats::default(),
            clipboard: arboard::Clipboard::new().ok(),
            want_capture_mouse: false,
            want_capture_keyboard: false,
            viewport_size: Vec2::new(1280.0, 720.0),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn run(dock: DockState) {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
        let event_loop = EventLoop::new().expect("event loop");
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut host = Self::new(dock);
        event_loop.run_app(&mut host).expect("run app");
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            // Need real hardware limits — buckets pin color-attachment bytes to 32,
            // which is already full for our 4-target G-buffer before velocity.
            apply_limit_buckets: false,
        }))
        .expect("no suitable GPU adapter");

        let mut limits = wgpu::Limits::default();
        // G-buffer is already 2×rgba16float + 2×rgba8unorm = 32 bytes/sample
        // (rgba8 counts as 8 in the WebGPU table). Velocity (rg16float = 4) needs ≥36.
        let adapter_bytes = adapter.limits().max_color_attachment_bytes_per_sample;
        assert!(
            adapter_bytes >= 36,
            "GPU max_color_attachment_bytes_per_sample={adapter_bytes}, need ≥36 for velocity MRT"
        );
        limits.max_color_attachment_bytes_per_sample = adapter_bytes.min(64);

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mega-render demo"),
            required_features: wgpu::Features::empty(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .expect("request_device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| *f == wgpu::TextureFormat::Rgba8UnormSrgb)
            .expect("adapter must support Rgba8UnormSrgb swapchain");

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: caps
                .present_modes
                .iter()
                .copied()
                .find(|m| *m == wgpu::PresentMode::Fifo)
                .unwrap_or(caps.present_modes[0]),
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);

        let mut visualizer = WgpuVisualizer::new(&device, &queue);
        let vp = (
            self.viewport_size.x.max(1.0) as u32,
            self.viewport_size.y.max(1.0) as u32,
        );
        visualizer.ensure_target(vp.0, vp.1);
        D::configure(&mut visualizer);

        let mut ui_renderer = UiRenderer::new(&device, &queue, format, &self.ui);
        ui_renderer.set_viewport(&queue, width as f32, height as f32);
        let scene_target = SceneTarget::ensure(&device, &mut ui_renderer, vp);

        self.gpu = Some(Gpu {
            device,
            queue,
            surface,
            config,
            visualizer,
            ui_renderer,
            scene_target,
        });
        self.window = Some(window);
    }

    fn resize(&mut self, width: u32, height: u32) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        if width == 0 || height == 0 {
            return;
        }
        gpu.config.width = width;
        gpu.config.height = height;
        gpu.surface.configure(&gpu.device, &gpu.config);
        gpu.ui_renderer
            .set_viewport(&gpu.queue, width as f32, height as f32);
    }

    fn begin_paste(&mut self) {
        self.input.key_paste = true;
        if !self.input.clipboard_paste.is_empty() {
            return;
        }
        if let Some(cb) = self.clipboard.as_mut() {
            if let Ok(text) = cb.get_text() {
                self.input.clipboard_paste = text;
            }
        }
    }

    fn apply_cursor(&mut self, window: &Window, cursor: CursorIcon) {
        if cursor == self.cursor {
            return;
        }
        self.cursor = cursor;
        window.set_cursor(map_cursor(cursor));
    }

    fn redraw(&mut self) {
        let Some(window) = self.window.clone() else {
            return;
        };
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.fps_accum_dt += dt;
        self.fps_frames += 1;
        if self.fps_accum_dt >= 1.0 {
            self.fps = self.fps_frames as f32 / self.fps_accum_dt;
            self.frame_ms = (self.fps_accum_dt / self.fps_frames.max(1) as f32) * 1000.0;
            self.fps_accum_dt = 0.0;
            self.fps_frames = 0;
        }

        if self.state.resync_camera {
            self.orbit.sync_from_camera(&self.state.scene.camera);
            self.state.resync_camera = false;
        }

        let scroll = self.input.scroll_delta.y;
        if scroll.abs() > 0.0
            && !self.state.has_drag()
            && self.state.rig.viewport_rect.contains(self.input.mouse_pos)
        {
            self.orbit.add_zoom(scroll);
        }

        self.state.scene.poll_loads();
        let demo_anim = D::update(&mut self.state, dt);
        self.orbit.tick_snap(dt);
        self.orbit.apply(&mut self.state.scene);
        if let Some(gpu) = self.gpu.as_mut() {
            if gpu.visualizer.post_process().dof.auto_focus {
                self.state.scene.camera.autofocus_ground(0.0);
            }
        }
        self.state.scene.camera.tick_focus(dt);

        let viewport = Vec2::new(size.width as f32, size.height as f32);
        let ui_input = if self.orbit.active() {
            let mut starved = self.input.to_ui(viewport, dt);
            starved.mouse_down = false;
            starved.mouse_pressed = false;
            starved.mouse_released = false;
            starved.mouse_right_down = false;
            starved.mouse_right_pressed = false;
            starved.mouse_right_released = false;
            starved.scroll_delta = Vec2::ZERO;
            starved
        } else {
            self.input.to_ui(viewport, dt)
        };

        self.ui.begin_frame(ui_input);

        let cursor = {
            let Some(gpu) = self.gpu.as_mut() else {
                return;
            };

            let mut viewport_size = self.viewport_size;
            let mut debug_view = gpu.visualizer.debug_view();
            let keep_ui = {
                let (post, shadow) = gpu.visualizer.effect_settings();
                let mut ctx = UiCtx {
                    state: &mut self.state,
                    post,
                    shadow,
                    debug_view: &mut debug_view,
                    dock: &mut self.dock,
                    window_size: viewport,
                    viewport_size: &mut viewport_size,
                    dt,
                    fps: self.fps,
                    frame_ms: self.frame_ms,
                    stats: self.draw_stats,
                };
                D::build_ui(&mut self.ui, &mut ctx)
            };
            gpu.visualizer.set_debug_view(debug_view);
            self.viewport_size = viewport_size;

            let out = self.ui.end_frame();
            self.want_capture_mouse = out.want_capture_mouse;
            self.want_capture_keyboard = out.want_capture_keyboard;
            self.animating = demo_anim || self.orbit.active() || keep_ui || out.needs_repaint;

            // Viewport tools use this frame's rect + pointer (before edge clear).
            if !self.orbit.active() {
                let vp_rect = self.state.rig.viewport_rect;
                let mouse = self.input.mouse_pos;
                let vp_size = Vec2::new(vp_rect.width().max(1.0), vp_rect.height().max(1.0));
                let local = mouse - vp_rect.min;
                let over_view_gizmo = vp_rect.contains(mouse)
                    && view_gizmo::contains_cursor(vp_size, local);

                let mut consumed = false;
                if over_view_gizmo && self.input.mouse_pressed {
                    if let Some(axis) =
                        view_gizmo::hit_test(&self.state.scene.camera, vp_size, local)
                    {
                        self.orbit.snap_to_dir(axis.dir());
                        self.state.status = format!("View {} · ortho", axis.label());
                        consumed = true;
                    }
                }

                if !consumed {
                    handle_tools(
                        &mut self.state,
                        &PointerFrame {
                            pos: mouse,
                            pressed: self.input.mouse_pressed && !over_view_gizmo,
                            down: self.input.mouse_down,
                            released: self.input.mouse_released,
                        },
                        out.want_capture_mouse && !vp_rect.contains(mouse),
                    );
                }
            }

            // Debug / gizmo after tools so pose matches this frame.
            self.state.scene.debug.clear();
            self.state.scene.debug.grid(
                Vec3::ZERO,
                2.5,
                0.25,
                [0.22, 0.22, 0.24, 0.10], // minor — very soft
                4,
                [0.28, 0.28, 0.30, 0.28], // major — still muted
            );
            draw_rig_debug(&mut self.state.scene, &self.state.rig);
            if let Some(sel) = self.state.rig.selection {
                let radius = self.state.gizmo_radius();
                match self.state.rig.tool {
                    Tool::Rotate => {
                        gizmo::draw_rotate_gizmo(
                            &mut self.state.scene,
                            sel,
                            radius,
                            self.state.gizmo_hover,
                            self.state.rotate_drag.as_ref(),
                        );
                    }
                    Tool::Translate => {
                        gizmo::draw_translate_gizmo(
                            &mut self.state.scene,
                            sel,
                            radius,
                            self.state.gizmo_hover,
                            self.state.translate_drag.as_ref(),
                        );
                    }
                    _ => {}
                }
            }

            // Orientation gizmo (HUD overlay on the viewport texture).
            {
                let vp_rect = self.state.rig.viewport_rect;
                let vp_size = Vec2::new(
                    self.viewport_size.x.max(1.0),
                    self.viewport_size.y.max(1.0),
                );
                let local = self.input.mouse_pos - vp_rect.min;
                let hud_input = InputFrame {
                    cursor: local,
                    mouse_down: self.input.mouse_down,
                    mouse_pressed: self.input.mouse_pressed,
                    mouse_released: self.input.mouse_released,
                    scroll_delta: Vec2::ZERO,
                    dt,
                };
                self.state.scene.hud.begin(&hud_input, vp_size);
                view_gizmo::draw(
                    &mut self.state.scene.hud,
                    &self.state.scene.camera,
                    vp_size,
                    local,
                );
                let _ = self.state.scene.hud.end();
            }

            if let Some(text) = out.clipboard {
                if let Some(cb) = self.clipboard.as_mut() {
                    let _ = cb.set_text(text);
                }
            }
            let cursor = out.cursor;
            let draw_list = out.draw_list;

            let vp_w = self.viewport_size.x.round().max(1.0) as u32;
            let vp_h = self.viewport_size.y.round().max(1.0) as u32;
            gpu.scene_target
                .resize(&gpu.device, &mut gpu.ui_renderer, vp_w, vp_h);
            gpu.visualizer.ensure_target(vp_w, vp_h);

            gpu.visualizer.sync(&self.state.scene);
            let aspect = vp_w as f32 / vp_h as f32;
            gpu.visualizer
                .render_to(&self.state.scene, aspect, &gpu.scene_target.render_view);

            gpu.ui_renderer
                .sync_atlases(&gpu.device, &gpu.queue, &mut self.ui);

            let frame = match gpu.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    window.request_redraw();
                    return;
                }
                _ => {
                    window.request_redraw();
                    return;
                }
            };

            let view = frame.texture.create_view(&Default::default());
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("demo frame"),
                });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.04,
                                g: 0.04,
                                b: 0.04,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                self.draw_stats = gpu.ui_renderer.draw(&gpu.queue, &mut pass, &draw_list);
            }

            gpu.queue.submit(Some(encoder.finish()));
            window.pre_present_notify();
            gpu.queue.present(frame);
            cursor
        };

        self.input.clear_edges();
        self.apply_cursor(&window, cursor);
    }
}

impl<D: Demo> ApplicationHandler for Host<D> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let (w, h) = D::window_size();
        let attrs = Window::default_attributes()
            .with_title(D::title())
            .with_inner_size(LogicalSize::new(w, h));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.init_gpu(window.clone());
        window.request_redraw();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let playing = self.animating || self.orbit.active() || self.state.has_drag();
        let ms = if playing { 8 } else { 33 };
        event_loop.set_control_flow(ControlFlow::wait_duration(Duration::from_millis(ms)));
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            let dx = delta.0 as f32;
            let dy = delta.1 as f32;
            let mut used = false;
            if self.orbit.orbiting && !self.state.has_drag() {
                self.orbit.add_orbit(dx, dy);
                used = true;
            }
            if self.orbit.panning {
                self.orbit.add_pan(dx, dy);
                used = true;
            }
            if used {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.resize(size.width, size.height);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = self.window.as_ref().map(|w| w.inner_size());
                if let Some(size) = size {
                    self.resize(size.width, size.height);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::CursorMoved { position, .. } => {
                self.input.mouse_pos = Vec2::new(position.x as f32, position.y as f32);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                let over_vp = self.state.rig.viewport_rect.contains(self.input.mouse_pos);
                let vp_size = Vec2::new(
                    self.state.rig.viewport_rect.width().max(1.0),
                    self.state.rig.viewport_rect.height().max(1.0),
                );
                let over_view_gizmo = over_vp
                    && view_gizmo::contains_cursor(
                        vp_size,
                        self.input.mouse_pos - self.state.rig.viewport_rect.min,
                    );
                let nav_ok = !self.state.has_drag()
                    && !over_view_gizmo
                    && (!self.want_capture_mouse || over_vp);
                match button {
                    MouseButton::Left => {
                        if down && !self.input.mouse_down {
                            self.input.mouse_pressed = true;
                        }
                        if !down && self.input.mouse_down {
                            self.input.mouse_released = true;
                        }
                        self.input.mouse_down = down;
                    }
                    MouseButton::Right => {
                        if down && !self.input.mouse_right_down {
                            self.input.mouse_right_pressed = true;
                        }
                        if !down && self.input.mouse_right_down {
                            self.input.mouse_right_released = true;
                        }
                        self.input.mouse_right_down = down;
                        self.orbit.orbiting = down && nav_ok;
                    }
                    MouseButton::Middle => {
                        if down && !self.input.mouse_middle_down {
                            self.input.mouse_middle_pressed = true;
                        }
                        if !down && self.input.mouse_middle_down {
                            self.input.mouse_middle_released = true;
                        }
                        self.input.mouse_middle_down = down;
                        self.orbit.panning = down && nav_ok;
                    }
                    _ => {}
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.input.scroll_delta = match delta {
                    MouseScrollDelta::LineDelta(x, y) => Vec2::new(x * 40.0, y * 40.0),
                    MouseScrollDelta::PixelDelta(p) => Vec2::new(p.x as f32, p.y as f32),
                };
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.input.modifiers = mods.state();
                self.input.key_shift = mods.state().shift_key();
                self.input.key_ctrl = mods.state().control_key();
            }
            WindowEvent::Focused(false) => {
                self.orbit.orbiting = false;
                self.orbit.panning = false;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let down = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    if down && code == KeyCode::Escape {
                        if self.state.rig.selection.is_some() || self.state.has_drag() {
                            self.state.rig.selection = None;
                            self.state.clear_drags();
                            self.state.edit_bone = None;
                            self.state.status = "Selection cleared.".into();
                        } else {
                            event_loop.exit();
                        }
                    }
                    if down && !self.want_capture_keyboard {
                        match code {
                            KeyCode::Tab => {
                                let next = match self.state.rig.mode {
                                    crate::rig::AppMode::Edit => crate::rig::AppMode::Pose,
                                    crate::rig::AppMode::Pose => crate::rig::AppMode::Edit,
                                };
                                self.state.set_mode(next);
                            }
                            KeyCode::KeyR | KeyCode::Digit3 => {
                                self.state.rig.tool = crate::rig::Tool::Rotate;
                                self.state.clear_drags();
                                self.state.status = "Tool: Rotate".into();
                            }
                            KeyCode::KeyG | KeyCode::Digit2 => {
                                if self.state.rig.mode == crate::rig::AppMode::Edit {
                                    self.state.rig.tool = crate::rig::Tool::Translate;
                                    self.state.clear_drags();
                                    self.state.status = "Tool: Move".into();
                                }
                            }
                            KeyCode::KeyA => {
                                if self.state.rig.mode == crate::rig::AppMode::Edit {
                                    self.state.rig.tool = crate::rig::Tool::AddBone;
                                    self.state.clear_drags();
                                    self.state.status =
                                        "Tool: Add — click to extrude / place root".into();
                                }
                            }
                            KeyCode::KeyE => {
                                if self.state.rig.mode == crate::rig::AppMode::Edit {
                                    if let Some(sel) = self.state.rig.selection {
                                        if self
                                            .state
                                            .rig
                                            .extrude_bone(&mut self.state.scene, sel)
                                            .is_some()
                                        {
                                            self.state.edit_bone = None;
                                            self.state.status = "Extruded bone".into();
                                        }
                                    }
                                }
                            }
                            KeyCode::Delete => {
                                if self.state.rig.mode == crate::rig::AppMode::Edit {
                                    if let Some(sel) = self.state.rig.selection {
                                        self.state
                                            .rig
                                            .delete_bone_subtree(&mut self.state.scene, sel);
                                        self.state.clear_drags();
                                        self.state.edit_bone = None;
                                        self.state.status = "Deleted bone subtree".into();
                                    }
                                }
                            }
                            KeyCode::Digit1 => {
                                self.state.rig.tool = crate::rig::Tool::Select;
                                self.state.clear_drags();
                                self.state.status = "Tool: Select".into();
                            }
                            _ => {}
                        }
                    }
                }

                if down {
                    let shortcut = self.input.shortcut_mod();
                    match &event.logical_key {
                        Key::Named(NamedKey::Backspace) => self.input.key_backspace = true,
                        Key::Named(NamedKey::Enter) => self.input.key_enter = true,
                        Key::Named(NamedKey::ArrowLeft) => self.input.key_left = true,
                        Key::Named(NamedKey::ArrowRight) => self.input.key_right = true,
                        Key::Named(NamedKey::ArrowUp) => self.input.key_up = true,
                        Key::Named(NamedKey::ArrowDown) => self.input.key_down = true,
                        Key::Named(NamedKey::Home) => self.input.key_home = true,
                        Key::Named(NamedKey::End) => self.input.key_end = true,
                        Key::Character(c) if shortcut => match c.to_lowercase().as_str() {
                            "c" => self.input.key_copy = true,
                            "v" => self.begin_paste(),
                            "x" => self.input.key_cut = true,
                            "a" => self.input.key_select_all = true,
                            _ => {}
                        },
                        _ => {}
                    }
                    if shortcut {
                        if let PhysicalKey::Code(code) = event.physical_key {
                            match code {
                                KeyCode::KeyC => self.input.key_copy = true,
                                KeyCode::KeyV => self.begin_paste(),
                                KeyCode::KeyX => self.input.key_cut = true,
                                KeyCode::KeyA => self.input.key_select_all = true,
                                _ => {}
                            }
                        }
                    }
                    if !shortcut {
                        if let Some(text) = event.text.as_ref() {
                            for ch in text.chars() {
                                if !ch.is_control() {
                                    self.input.text.push(ch);
                                }
                            }
                        }
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::Ime(winit::event::Ime::Commit(text)) => {
                self.input.text.push_str(&text);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn map_cursor(icon: CursorIcon) -> winit::window::CursorIcon {
    match icon {
        CursorIcon::Default => winit::window::CursorIcon::Default,
        CursorIcon::Pointer => winit::window::CursorIcon::Pointer,
        CursorIcon::Move => winit::window::CursorIcon::Move,
        CursorIcon::ResizeNwse => winit::window::CursorIcon::NwseResize,
        CursorIcon::ResizeEw => winit::window::CursorIcon::EwResize,
        CursorIcon::ResizeNs => winit::window::CursorIcon::NsResize,
        CursorIcon::Text => winit::window::CursorIcon::Text,
    }
}
