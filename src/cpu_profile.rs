//! Cheap per-frame CPU zones. F8 dumps a text report for pasting into chat.

use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const CAP: usize = 600;
const N: usize = 20;

pub const ZONE_NAMES: [&str; N] = [
    "update",
    "ui_build",
    "ui_end",
    "tools",
    "drivers_pre_ik",
    "ik",
    "soft_chains",
    "drivers_post_ik",
    "dbg_grid",
    "dbg_skeleton",
    "dbg_helpers",
    "dbg_gizmo",
    "hud",
    "vp_resize",
    "gpu_sync",
    "gpu_scene",
    "gpu_atlas",
    "swapchain",
    "ui_encode",
    "present",
];

const ZONE_LOOK: [&str; N] = [
    "framework: poll_loads / camera / orbit",
    "app.rs build_ui — dock, panels, widgets",
    "mega-ui Ui::end_frame — layout + tessellate",
    "app.rs handle_tools — gizmo / pick / sculpt drag",
    "driver.rs evaluate_drivers PreIk",
    "ik_chain.rs evaluate_ik_chains",
    "soft_chain.rs evaluate_soft_chains",
    "driver.rs evaluate_drivers PostIk",
    "framework: scene.debug.grid",
    "rig.rs draw_rig_debug — bone lines/points",
    "ik/soft/weight overlays + brush cursor",
    "gizmo.rs rotate/translate draw",
    "view_gizmo + scene.hud",
    "scene_target.resize + visualizer.ensure_target",
    "mega-render WgpuVisualizer::sync",
    "mega-render WgpuVisualizer::render_to (encode 3D)",
    "mega-ui UiRenderer::sync_atlases",
    "wgpu get_current_texture — vsync/GPU wait if large",
    "mega-ui UiRenderer::draw swapchain pass",
    "queue.submit + present",
];

#[derive(Clone, Copy)]
pub enum Zone {
    Update = 0,
    UiBuild = 1,
    UiEnd = 2,
    Tools = 3,
    DriversPreIk = 4,
    Ik = 5,
    Soft = 6,
    DriversPostIk = 7,
    DbgGrid = 8,
    DbgSkeleton = 9,
    DbgHelpers = 10,
    DbgGizmo = 11,
    Hud = 12,
    VpResize = 13,
    GpuSync = 14,
    GpuScene = 15,
    GpuAtlas = 16,
    Swapchain = 17,
    UiEncode = 18,
    Present = 19,
}

pub struct CpuProfile {
    frames: Vec<Frame>,
}

struct Frame {
    total_ms: f32,
    zones: [f32; N],
}

pub struct FrameClock {
    t0: Instant,
    last: Instant,
    zones: [f32; N],
}

impl CpuProfile {
    pub fn new() -> Self {
        Self {
            frames: Vec::with_capacity(CAP),
        }
    }

    pub fn clear(&mut self) {
        self.frames.clear();
    }

    pub fn begin() -> FrameClock {
        let t = Instant::now();
        FrameClock {
            t0: t,
            last: t,
            zones: [0.0; N],
        }
    }

    fn push(&mut self, frame: Frame) {
        if self.frames.len() == CAP {
            self.frames.remove(0);
        }
        self.frames.push(frame);
    }

    pub fn dump(&self, extra: &str) -> Result<(PathBuf, String), String> {
        let report = self.format(extra);
        let path = Path::new("cpu-profile.txt");
        std::fs::write(path, &report).map_err(|e| e.to_string())?;
        Ok((path.to_path_buf(), report))
    }

    fn format(&self, extra: &str) -> String {
        let n = self.frames.len();
        let mut s = String::new();
        let _ = writeln!(s, "model-rig CPU profile");
        let _ = writeln!(s, "frames={n} (ring cap {CAP})");
        if !extra.is_empty() {
            let _ = writeln!(s, "{extra}");
        }
        if n == 0 {
            let _ = writeln!(s, "no frames yet — run the app a few seconds, then F8");
            return s;
        }

        let mut totals: Vec<f32> = self.frames.iter().map(|f| f.total_ms).collect();
        totals.sort_by(|a, b| a.total_cmp(b));
        let avg: f32 = totals.iter().sum::<f32>() / n as f32;
        let _ = writeln!(s);
        let _ = writeln!(s, "frame_ms  avg={:.2}  p50={:.2}  p95={:.2}  max={:.2}", avg, pct(&totals, 0.50), pct(&totals, 0.95), totals[n - 1]);
        let _ = writeln!(s, "fps_est   avg={:.0}  p50={:.0}  p95_slow={:.0}", 1000.0 / avg.max(0.01), 1000.0 / pct(&totals, 0.50).max(0.01), 1000.0 / pct(&totals, 0.95).max(0.01));

        let _ = writeln!(s);
        let _ = writeln!(s, "zones (ms)                    avg    p50    p95    max   %frame");
        let mut rows: Vec<(usize, f32, f32, f32, f32, f32)> = Vec::new();
        for i in 0..N {
            let mut v: Vec<f32> = self.frames.iter().map(|f| f.zones[i]).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            let zavg = v.iter().sum::<f32>() / n as f32;
            rows.push((i, zavg, pct(&v, 0.50), pct(&v, 0.95), v[n - 1], 100.0 * zavg / avg.max(0.01)));
        }
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        for &(i, zavg, p50, p95, max, pct_frame) in &rows {
            if zavg < 0.005 && max < 0.05 {
                continue;
            }
            let _ = writeln!(
                s,
                "  {:22}  {:6.2} {:6.2} {:6.2} {:6.2}  {:5.1}%",
                ZONE_NAMES[i], zavg, p50, p95, max, pct_frame
            );
        }

        let _ = writeln!(s);
        let _ = writeln!(s, "look (share >= 3%)");
        for &(i, _, _, _, _, pct_frame) in &rows {
            if pct_frame < 3.0 {
                continue;
            }
            let _ = writeln!(s, "  {:22}  {}", ZONE_NAMES[i], ZONE_LOOK[i]);
        }

        let mut indexed: Vec<(usize, f32)> = self.frames.iter().enumerate().map(|(i, f)| (i, f.total_ms)).collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let _ = writeln!(s);
        let _ = writeln!(s, "worst frames (ms, top 3 zones)");
        for &(idx, total) in indexed.iter().take(8) {
            let f = &self.frames[idx];
            let mut z: Vec<(usize, f32)> = (0..N).map(|i| (i, f.zones[i])).collect();
            z.sort_by(|a, b| b.1.total_cmp(&a.1));
            let _ = write!(s, "  #{idx:<3} {total:6.2}  ");
            for (j, (zi, ms)) in z.iter().take(3).enumerate() {
                if j > 0 {
                    s.push_str("  ");
                }
                let _ = write!(s, "{}={:.2}", ZONE_NAMES[*zi], ms);
            }
            s.push('\n');
        }

        let _ = writeln!(s);
        let _ = writeln!(s, "read: large `swapchain` = waiting on GPU/vsync, not CPU. `gpu_scene` is CPU encode of the 3D pass (inside mega-render).");
        s
    }
}

impl FrameClock {
    pub fn lap(&mut self, zone: Zone) {
        let now = Instant::now();
        self.zones[zone as usize] = (now - self.last).as_secs_f32() * 1000.0;
        self.last = now;
    }

    pub fn finish(self, profile: &mut CpuProfile) {
        profile.push(Frame {
            total_ms: (Instant::now() - self.t0).as_secs_f32() * 1000.0,
            zones: self.zones,
        });
    }
}

fn pct(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() - 1) as f32 * p).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}
