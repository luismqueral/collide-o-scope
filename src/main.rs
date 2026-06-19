#![allow(deprecated)] // egui 0.34 deprecation warnings for panel API renames
#![allow(dead_code)] // Old egui UI code kept as reference during web UI migration

//! Application entry point and the winit event loop.
//!
//! This file owns the top-level `App` state (layers, master effects, transport,
//! tempo, library, export job, …) and drives everything per frame. The flow:
//! `main()` parses CLI args, starts the web control-panel server, and hands an
//! `App` to winit's event loop. winit then calls back into our
//! `ApplicationHandler` impl: `resumed()` once to build the window + GPU, and
//! `window_event()` for every input/redraw. The real work happens in the
//! `RedrawRequested` arm, which runs the whole per-frame pipeline (drain web
//! actions → advance video → evaluate automation → render → present).
//!
//! `mod foo;` declares a submodule — Rust pulls in `foo.rs` (or `foo/mod.rs`).
//! These lines wire the other source files into this binary's module tree.

mod audio;
mod automation;
mod effects;
mod input;
mod layers;
mod ntsc;
mod patch;
mod render_export;
mod renderer;
mod video;
mod web;

#[cfg(test)]
mod test_support;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui_wgpu::ScreenDescriptor;
use winit::application::ApplicationHandler;
use winit::event::{KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::ModifiersState;
use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

use input::{apply_action, map_key, ControlFlow};
use layers::{is_supported_media, BlendMode, Layer};
use renderer::Renderer;
use web::state::WebState;

const TARGET_FPS: u64 = 30;
const FRAME_DURATION: Duration = Duration::from_millis(1000 / TARGET_FPS);

/// Output pixel dims for a ratio label + quality (= length of the SHORTER side).
/// 16:9/1080 → 1920×1080, 9:16/1080 → 1080×1920, 1:1/1080 → 1080×1080.
/// Result dims are forced even (h264/encoder-friendly for export).
fn output_dims(ratio: &str, quality: u32) -> (u32, u32) {
    let short = quality.max(2);
    let (rw, rh): (f32, f32) = match ratio {
        "4:3" => (4.0, 3.0),
        "1:1" => (1.0, 1.0),
        "9:16" => (9.0, 16.0),
        "21:9" => (21.0, 9.0),
        _ => (16.0, 9.0), // default + "16:9"
    };
    let (w, h) = if rw >= rh {
        (((short as f32) * rw / rh).round() as u32, short) // landscape/square: height = short
    } else {
        (short, ((short as f32) * rh / rw).round() as u32) // portrait: width = short
    };
    (w & !1, h & !1) // even dims
}

/// UV scale for a layer's fit mode given source (sw×sh) + canvas (dw×dh) dims.
/// mode: 0=stretch (1,1), 1=fit/contain (letterbox), 2=fill/cover (crop).
/// Multiplied about-center in the shader; >1 = transparent bars, <1 = crop.
pub fn fit_scale(mode: f32, sw: f32, sh: f32, dw: f32, dh: f32) -> (f32, f32) {
    if sw <= 0.0 || sh <= 0.0 || dw <= 0.0 || dh <= 0.0 {
        return (1.0, 1.0);
    }
    let r = (sw / sh) / (dw / dh); // source aspect / canvas aspect
    match mode.round() as i32 {
        1 => {
            if r > 1.0 {
                (1.0, r)
            } else {
                (1.0 / r, 1.0)
            }
        } // contain
        2 => {
            if r > 1.0 {
                (1.0 / r, 1.0)
            } else {
                (1.0, r)
            }
        } // cover
        _ => (1.0, 1.0), // stretch
    }
}

/// All top-level application state, held for the lifetime of the program.
///
/// Several fields are `Option<...>` because they can't be built until winit
/// hands us a window: `window`, `renderer`, and the three `egui_*` handles all
/// start `None` and are filled in lazily inside `resumed()` (see the
/// `ApplicationHandler` impl below). Everything else is initialised up-front in
/// `App::new`. This struct is effectively the single source of truth the
/// per-frame `RedrawRequested` pipeline reads from and writes to.
struct App {
    initial_video: Option<String>,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    layers: Vec<Layer>,
    selected_layer: Option<usize>,
    next_layer_id: u64,
    master_effects: effects::EffectUniforms,
    // Master param automations: param name → compiled expression
    master_automations: std::collections::HashMap<String, automation::Expr>,
    // Master automation parse errors: param name → error message
    master_automation_errors: std::collections::HashMap<String, String>,
    master_paused: bool,
    // Master audio bus. The engine holds the audible authority (callback
    // atomics); these mirror it for snapshots/patches and survive when audio is
    // disabled (no output device).
    master_volume: f32, // dB, −60..+6 (0 = unity)
    master_limiter: bool,
    last_frame_time: Instant,
    start_time: Instant,
    // Tap-tempo state. `tap_bpm` is the current tempo; `tap_downbeat` is the
    // elapsed-seconds moment of the last tap (beat phase 0); `tap_times` holds
    // recent tap timestamps (elapsed secs) so we can average their spacing.
    tap_bpm: f32,
    tap_downbeat: f32,
    tap_times: Vec<f32>,
    // Master content framerate (frame-hold / stutter). 30 = smooth (default).
    master_fps: f32,
    last_content_advance: Instant,
    content_tick: u64,
    content_time: f32,
    // Master output size: aspect ratio label + quality (shorter-side px).
    output_ratio: String,
    output_quality: u32,
    modifiers: ModifiersState,
    // Library
    library_folder: Option<PathBuf>,
    library_files: Vec<PathBuf>,
    // Saved patch names (file stems), cached; re-scanned on save/delete
    patch_files: Vec<String>,
    // YAML editor
    yaml_editor: patch::editor::EditorState,
    // egui state
    egui_ctx: egui::Context,
    egui_winit: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,
    video_egui_texture_id: Option<egui::TextureId>,
    // NTSC/VHS effects
    ntsc_state: ntsc::NtscState,
    // Web control panel
    web_state: Arc<WebState>,
    // Offline render export
    export_job: Option<render_export::ExportJob>,
    // Audio output (Phase 1: per-layer mute/volume/pan + varispeed, mixed to a
    // master volume/limiter/meter bus). None if no usable output device was
    // found — the app then runs silently.
    audio: Option<audio::AudioEngine>,
}

impl App {
    fn new(
        initial_video: Option<String>,
        library_folder: Option<PathBuf>,
        web_state: Arc<WebState>,
    ) -> Self {
        let library_files = library_folder
            .as_ref()
            .map(|f| scan_folder(f))
            .unwrap_or_default();

        // Generate thumbnails on background thread
        generate_thumbnails(&library_files, web_state.clone());

        let patch_files = scan_patches(&patches_dir(&library_folder));

        // Build the audio engine up-front; if no output device is available we
        // log and keep going silently rather than failing the whole app.
        let audio = match audio::AudioEngine::new() {
            Ok(engine) => Some(engine),
            Err(e) => {
                eprintln!("Audio disabled: {e}");
                None
            }
        };

        Self {
            initial_video,
            window: None,
            renderer: None,
            layers: Vec::new(),
            selected_layer: None,
            next_layer_id: 0,
            master_effects: effects::EffectUniforms::default(),
            master_automations: std::collections::HashMap::new(),
            master_automation_errors: std::collections::HashMap::new(),
            master_paused: false,
            master_volume: 0.0,
            master_limiter: true,
            last_frame_time: Instant::now(),
            start_time: Instant::now(),
            tap_bpm: 120.0,
            tap_downbeat: 0.0,
            tap_times: Vec::new(),
            master_fps: 30.0,
            last_content_advance: Instant::now(),
            content_tick: 0,
            content_time: 0.0,
            output_ratio: "16:9".to_string(),
            output_quality: 1080,
            modifiers: ModifiersState::empty(),
            library_folder,
            library_files,
            patch_files,
            yaml_editor: patch::editor::EditorState::default(),
            egui_ctx: egui::Context::default(),
            egui_winit: None,
            egui_renderer: None,
            video_egui_texture_id: None,
            ntsc_state: ntsc::NtscState::new(),
            web_state,
            export_job: None,
            audio,
        }
    }

    fn add_layer(&mut self, path: &str) {
        let renderer = self.renderer.as_ref().unwrap();
        match Layer::new(path, &renderer.device) {
            Ok(mut layer) => {
                let id = self.next_layer_id;
                layer.id = id;
                self.next_layer_id += 1;
                self.layers.push(layer);
                self.selected_layer = Some(self.layers.len() - 1);
                // Register this layer's audio track with the mixer, keyed by the
                // stable layer id. Files with no audio stream stay silent.
                if let Some(audio) = &self.audio {
                    audio.add_source(id, path);
                }
            }
            Err(e) => {
                eprintln!("Failed to open video: {e}");
            }
        }
    }

    fn set_library_folder(&mut self, folder: PathBuf) {
        self.library_files = scan_folder(&folder);
        self.library_folder = Some(folder);
    }

    /// Handle an action from the web UI.
    fn handle_web_action(&mut self, action: web::state::WebAction) {
        use web::state::WebAction;
        match action {
            WebAction::SetParam { param, value } => {
                let mut snap = web::state::EffectsSnapshot::from_uniforms(&self.master_effects);
                snap.apply_param(&param, &value);
                snap.apply_to_uniforms(&mut self.master_effects);
            }
            WebAction::SetMasterFramerate { value } => {
                self.master_fps = value.clamp(1.0, TARGET_FPS as f32);
            }
            WebAction::SetOutputSize { ratio, quality } => {
                self.output_ratio = ratio;
                self.output_quality = quality.clamp(120, 4320);
                let (w, h) = output_dims(&self.output_ratio, self.output_quality);
                if let Some(r) = self.renderer.as_mut() {
                    r.set_output_size(w, h);
                    self.master_effects.resolution = [w as f32, h as f32];
                    // output_view moved — rebind the egui preview texture.
                    if let (Some(egui_r), Some(id)) =
                        (self.egui_renderer.as_mut(), self.video_egui_texture_id)
                    {
                        egui_r.update_egui_texture_from_wgpu_texture(
                            &r.device,
                            &r.output_view,
                            wgpu::FilterMode::Linear,
                            id,
                        );
                    }
                }
            }
            WebAction::AddLayer { filename } => {
                // Find the full path from the library
                if let Some(path) = self.library_files.iter().find(|p| {
                    p.file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .as_deref()
                        == Some(&filename)
                }) {
                    let path_str = path.to_string_lossy().to_string();
                    self.add_layer(&path_str);
                }
            }
            WebAction::SetLayerClip { index, filename } => {
                // Swap a layer's source video in place, preserving its FX, opacity,
                // blend, transport, and automation. Only the decoder/texture/dims
                // (and the resolution-dependent uniform) come from the new clip.
                if index < self.layers.len() {
                    let path_str = self
                        .library_files
                        .iter()
                        .find(|p| {
                            p.file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .as_deref()
                                == Some(&filename)
                        })
                        .map(|p| p.to_string_lossy().to_string());
                    if let Some(path_str) = path_str {
                        let renderer = self.renderer.as_ref().unwrap();
                        match Layer::new(&path_str, &renderer.device) {
                            Ok(fresh) => {
                                let layer = &mut self.layers[index];
                                layer.decoder = fresh.decoder;
                                layer.audio_only = fresh.audio_only;
                                layer.texture = fresh.texture;
                                layer.texture_view = fresh.texture_view;
                                layer.width = fresh.width;
                                layer.height = fresh.height;
                                layer.filename = fresh.filename;
                                layer.effects.resolution =
                                    [fresh.width as f32, fresh.height as f32];
                                layer.frame_accumulator = 0.0;
                                // Swap the audio source in place too (keeps the
                                // layer's mute/volume/pan/speed).
                                let id = layer.id;
                                if let Some(audio) = &self.audio {
                                    audio.set_source_path(id, &path_str);
                                }
                            }
                            Err(e) => eprintln!("Failed to open video: {e}"),
                        }
                    }
                }
            }
            WebAction::RemoveLayer { index } => {
                if index < self.layers.len() {
                    let id = self.layers[index].id;
                    if let Some(audio) = &self.audio {
                        audio.remove_source(id);
                    }
                    self.layers.remove(index);
                    if self.layers.is_empty() {
                        self.selected_layer = None;
                    } else if let Some(sel) = self.selected_layer {
                        if sel >= self.layers.len() {
                            self.selected_layer = Some(self.layers.len() - 1);
                        }
                    }
                }
            }
            WebAction::MoveLayer { from, to } => {
                let len = self.layers.len();
                if from < len && to < len && from != to {
                    let layer = self.layers.remove(from);
                    self.layers.insert(to, layer);
                    // Keep the egui selection pointing at the same layer.
                    if let Some(sel) = self.selected_layer {
                        self.selected_layer = Some(if sel == from {
                            to
                        } else if from < sel && sel <= to {
                            sel - 1
                        } else if to <= sel && sel < from {
                            sel + 1
                        } else {
                            sel
                        });
                    }
                    log::info!("Layer moved {from} → {to}");
                }
            }
            WebAction::ToggleVisibility { index } => {
                if index < self.layers.len() {
                    self.layers[index].visible = !self.layers[index].visible;
                    log::info!("Layer {index} visibility → {}", self.layers[index].visible);
                }
            }
            WebAction::ToggleLayerPause { index } => {
                if index < self.layers.len() {
                    let paused = !self.layers[index].paused;
                    self.layers[index].paused = paused;
                    let id = self.layers[index].id;
                    if let Some(audio) = &self.audio {
                        audio.set_paused(id, paused);
                    }
                    log::info!("Layer {index} paused → {paused}");
                }
            }
            WebAction::ToggleMasterPause => {
                self.master_paused = !self.master_paused;
                if let Some(audio) = &self.audio {
                    audio.set_master_paused(self.master_paused);
                }
            }
            WebAction::ResetFx => {
                self.master_effects.reset();
            }
            WebAction::ResetGroup { group } => {
                let defaults = crate::effects::EffectUniforms::default();
                match group.as_str() {
                    "digital" => {
                        self.master_effects.pixelate_size = defaults.pixelate_size;
                        self.master_effects.rgb_split = defaults.rgb_split;
                        self.master_effects.hue_shift = defaults.hue_shift;
                        self.master_effects.saturation = defaults.saturation;
                        self.master_effects.brightness = defaults.brightness;
                        self.master_effects.contrast = defaults.contrast;
                        self.master_effects.posterize = defaults.posterize;
                        self.master_effects.invert = defaults.invert;
                    }
                    "analog" => {
                        self.master_effects.grain_intensity = defaults.grain_intensity;
                        self.master_effects.grain_size = defaults.grain_size;
                        self.master_effects.grain_algo = defaults.grain_algo;
                        self.master_effects.color_grain = defaults.color_grain;
                        self.master_effects.vignette = defaults.vignette;
                        self.master_effects.color_drift = defaults.color_drift;
                    }
                    "motion" => {
                        self.master_effects.breathe_scale = defaults.breathe_scale;
                        self.master_effects.breathe_rotation = defaults.breathe_rotation;
                        self.master_effects.breathe_position = defaults.breathe_position;
                    }
                    "vhs" => {
                        self.ntsc_state.params = ntsc::NtscParams::default();
                    }
                    _ => {}
                }
            }
            WebAction::SetLayerParam {
                index,
                param,
                value,
            } => {
                if index < self.layers.len() {
                    let layer = &mut self.layers[index];
                    match param.as_str() {
                        "opacity" => {
                            if let Some(v) = value.as_f64() {
                                layer.opacity = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "speed" => {
                            if let Some(v) = value.as_f64() {
                                layer.speed = (v as f32).clamp(0.25, 4.0);
                            }
                        }
                        "fps" => {
                            if let Some(v) = value.as_f64() {
                                layer.fps = (v as f32).clamp(1.0, 60.0);
                            }
                        }
                        // Loop in/out points (fractions of the clip). Set the one
                        // field, then push the pair through `set_loop` so the
                        // decoder's window updates (and gets clamped/ordered).
                        "loop_start" => {
                            if let Some(v) = value.as_f64() {
                                layer.set_loop((v as f32).clamp(0.0, 1.0), layer.loop_end);
                            }
                        }
                        "loop_end" => {
                            if let Some(v) = value.as_f64() {
                                layer.set_loop(layer.loop_start, (v as f32).clamp(0.0, 1.0));
                            }
                        }
                        "blend_mode" => {
                            if let Some(s) = value.as_str() {
                                layer.blend_mode = match s {
                                    "screen" => crate::layers::BlendMode::Screen,
                                    "multiply" => crate::layers::BlendMode::Multiply,
                                    "difference" => crate::layers::BlendMode::Difference,
                                    _ => crate::layers::BlendMode::Normal,
                                };
                            }
                        }
                        "mute" => {
                            if let Some(b) = value.as_bool() {
                                layer.audio.mute = b;
                            }
                        }
                        "volume" => {
                            if let Some(v) = value.as_f64() {
                                layer.audio.volume = (v as f32).clamp(-60.0, 6.0);
                            }
                        }
                        "pan" => {
                            if let Some(v) = value.as_f64() {
                                layer.audio.pan = (v as f32).clamp(-1.0, 1.0);
                            }
                        }
                        "eq_low" => {
                            if let Some(v) = value.as_f64() {
                                layer.audio.eq_low = (v as f32).clamp(-24.0, 12.0);
                            }
                        }
                        "eq_mid" => {
                            if let Some(v) = value.as_f64() {
                                layer.audio.eq_mid = (v as f32).clamp(-24.0, 12.0);
                            }
                        }
                        "eq_high" => {
                            if let Some(v) = value.as_f64() {
                                layer.audio.eq_high = (v as f32).clamp(-24.0, 12.0);
                            }
                        }
                        "delay_time" => {
                            if let Some(v) = value.as_f64() {
                                layer.audio.delay_time = (v as f32).clamp(0.0, 1000.0);
                            }
                        }
                        "delay_feedback" => {
                            if let Some(v) = value.as_f64() {
                                layer.audio.delay_feedback = (v as f32).clamp(0.0, 0.95);
                            }
                        }
                        "delay_mix" => {
                            if let Some(v) = value.as_f64() {
                                layer.audio.delay_mix = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "hue_shift" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.hue_shift = (v as f32).clamp(-180.0, 180.0);
                            }
                        }
                        "saturation" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.saturation = (v as f32).clamp(-1.0, 1.0);
                            }
                        }
                        "brightness" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.brightness = (v as f32).clamp(-1.0, 1.0);
                            }
                        }
                        "contrast" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.contrast = (v as f32).clamp(-1.0, 1.0);
                            }
                        }
                        "pixelate" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.pixelate_size = (v as f32).clamp(1.0, 32.0);
                            }
                        }
                        "rgb_split" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.rgb_split = (v as f32).clamp(0.0, 30.0);
                            }
                        }
                        "posterize" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.posterize = (v as f32).clamp(0.0, 16.0);
                            }
                        }
                        "invert" => {
                            if let Some(b) = value.as_bool() {
                                layer.effects.invert = if b { 1.0 } else { 0.0 };
                            }
                        }
                        "wave_amp" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.wave_amp = (v as f32).clamp(0.0, 0.1);
                            }
                        }
                        "wave_freq" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.wave_freq = (v as f32).clamp(0.0, 50.0);
                            }
                        }
                        "wave_speed" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.wave_speed = (v as f32).clamp(0.0, 10.0);
                            }
                        }
                        "wave_axis" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.wave_axis = (v as f32).clamp(0.0, 2.0);
                            } else if let Some(s) = value.as_str() {
                                layer.effects.wave_axis =
                                    s.parse::<f32>().unwrap_or(0.0).clamp(0.0, 2.0);
                            }
                        }
                        "swirl_angle" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.swirl_angle = (v as f32).clamp(-720.0, 720.0);
                            }
                        }
                        "swirl_radius" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.swirl_radius = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "bulge_strength" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.bulge_strength = (v as f32).clamp(-1.0, 1.0);
                            }
                        }
                        "bulge_radius" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.bulge_radius = (v as f32).clamp(0.05, 1.0);
                            }
                        }
                        "chroma_enable" => {
                            if let Some(b) = value.as_bool() {
                                layer.effects.chroma_enable = if b { 1.0 } else { 0.0 };
                            }
                        }
                        "chroma_threshold" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.chroma_threshold = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "chroma_smoothness" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.chroma_smoothness = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "chroma_spill" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.chroma_spill = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "chroma_color" => {
                            if let Some(s) = value.as_str() {
                                let (r, g, b) = hex_to_rgb01(s);
                                layer.effects.chroma_color_r = r;
                                layer.effects.chroma_color_g = g;
                                layer.effects.chroma_color_b = b;
                            }
                        }
                        "chroma_bg_enable" => {
                            if let Some(b) = value.as_bool() {
                                layer.effects.chroma_bg_enable = if b { 1.0 } else { 0.0 };
                            }
                        }
                        "chroma_bg_color" => {
                            if let Some(s) = value.as_str() {
                                let (r, g, b) = hex_to_rgb01(s);
                                layer.effects.chroma_bg_r = r;
                                layer.effects.chroma_bg_g = g;
                                layer.effects.chroma_bg_b = b;
                            }
                        }
                        "slice_intensity" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.slice_intensity = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "slice_height" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.slice_height = (v as f32).clamp(1.0, 128.0);
                            }
                        }
                        "slice_prob" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.slice_prob = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "slice_speed" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.slice_speed = (v as f32).clamp(0.0, 30.0);
                            }
                        }
                        "block_size" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.block_size = (v as f32).clamp(4.0, 128.0);
                            }
                        }
                        "block_intensity" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.block_intensity = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "block_prob" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.block_prob = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "block_speed" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.block_speed = (v as f32).clamp(0.0, 30.0);
                            }
                        }
                        "shift_chroma" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.shift_chroma = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "slice_axis" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.slice_axis = (v as f32).clamp(0.0, 2.0);
                            }
                        }
                        "jitter_amount" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.jitter_amount = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "jitter_speed" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.jitter_speed = (v as f32).clamp(0.0, 30.0);
                            }
                        }
                        "datamosh" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.datamosh = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "feedback_persistence" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.feedback_persistence = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "feedback_zoom" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.feedback_zoom = (v as f32).clamp(0.8, 1.2);
                            }
                        }
                        "feedback_rotate" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.feedback_rotate = (v as f32).clamp(-30.0, 30.0);
                            }
                        }
                        "feedback_luma_key" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.feedback_luma_key = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "feedback_chroma" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.feedback_chroma = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "feedback_additive" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.feedback_additive = (v as f32).clamp(0.0, 1.0);
                            }
                        }
                        "layer_x" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.layer_x = (v as f32).clamp(-1.0, 1.0);
                            }
                        }
                        "layer_y" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.layer_y = (v as f32).clamp(-1.0, 1.0);
                            }
                        }
                        "layer_scale" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.layer_scale = (v as f32).clamp(0.1, 4.0);
                            }
                        }
                        "fit_mode" => {
                            if let Some(v) = value.as_f64() {
                                layer.effects.fit_mode = (v as f32).clamp(0.0, 2.0).round();
                            }
                        }
                        _ => {}
                    }
                    // Forward audio-relevant changes to the mixer. Done after the
                    // `&mut layer` borrow above ends (NLL) so we can touch
                    // `self.audio` + re-read the layer.
                    if let Some(audio) = &self.audio {
                        let id = self.layers[index].id;
                        match param.as_str() {
                            "mute" | "volume" | "pan" | "eq_low" | "eq_mid" | "eq_high"
                            | "delay_time" | "delay_feedback" | "delay_mix" => {
                                audio.set_params(id, self.layers[index].audio)
                            }
                            "speed" => audio.set_speed(id, self.layers[index].speed),
                            _ => {}
                        }
                    }
                }
            }
            WebAction::ResetLayerGroup { index, group } => {
                if let Some(layer) = self.layers.get_mut(index) {
                    let d = crate::effects::EffectUniforms::default();
                    match group.as_str() {
                        "source" => {
                            // Transport only (speed/fps + loop window). Clip +
                            // paused are left alone — resetting a group shouldn't
                            // reload media.
                            layer.speed = 1.0;
                            layer.fps = 30.0;
                            // Back to whole-clip loop (forwards into the decoder).
                            layer.set_loop(0.0, 1.0);
                        }
                        "blend" => {
                            // Composite params. visible is left alone (matches the
                            // prior behavior of not toggling visibility on reset).
                            layer.opacity = 1.0;
                            layer.blend_mode = crate::layers::BlendMode::Normal;
                        }
                        "audio" => {
                            // Only the mixer-channel params; Audio FX (eq/delay)
                            // is a separate group ("audiofx") so it survives.
                            layer.audio.mute = false;
                            layer.audio.volume = 0.0;
                            layer.audio.pan = 0.0;
                        }
                        "audiofx" => {
                            layer.audio.eq_low = 0.0;
                            layer.audio.eq_mid = 0.0;
                            layer.audio.eq_high = 0.0;
                            layer.audio.delay_time = 0.0;
                            layer.audio.delay_feedback = 0.0;
                            layer.audio.delay_mix = 0.0;
                        }
                        "color" => {
                            // DIGITAL was dissolved into the layer view: rgb_split,
                            // posterize and invert now live in COLOR (plus the
                            // grade params and shift_chroma).
                            layer.effects.hue_shift = d.hue_shift;
                            layer.effects.saturation = d.saturation;
                            layer.effects.brightness = d.brightness;
                            layer.effects.contrast = d.contrast;
                            layer.effects.invert = d.invert;
                            layer.effects.posterize = d.posterize;
                            layer.effects.shift_chroma = d.shift_chroma;
                            layer.effects.rgb_split = d.rgb_split;
                        }
                        "warp" => {
                            // pixelate (the other former DIGITAL param) folds into
                            // WARP alongside the geometric distortions.
                            layer.effects.wave_amp = d.wave_amp;
                            layer.effects.wave_freq = d.wave_freq;
                            layer.effects.wave_speed = d.wave_speed;
                            layer.effects.wave_axis = d.wave_axis;
                            layer.effects.swirl_angle = d.swirl_angle;
                            layer.effects.swirl_radius = d.swirl_radius;
                            layer.effects.bulge_strength = d.bulge_strength;
                            layer.effects.bulge_radius = d.bulge_radius;
                            layer.effects.pixelate_size = d.pixelate_size;
                        }
                        "colorkey" => {
                            layer.effects.chroma_enable = d.chroma_enable;
                            layer.effects.chroma_threshold = d.chroma_threshold;
                            layer.effects.chroma_smoothness = d.chroma_smoothness;
                            layer.effects.chroma_spill = d.chroma_spill;
                            layer.effects.chroma_color_r = d.chroma_color_r;
                            layer.effects.chroma_color_g = d.chroma_color_g;
                            layer.effects.chroma_color_b = d.chroma_color_b;
                            layer.effects.chroma_bg_enable = d.chroma_bg_enable;
                            layer.effects.chroma_bg_r = d.chroma_bg_r;
                            layer.effects.chroma_bg_g = d.chroma_bg_g;
                            layer.effects.chroma_bg_b = d.chroma_bg_b;
                        }
                        "slice" => {
                            layer.effects.slice_intensity = d.slice_intensity;
                            layer.effects.slice_height = d.slice_height;
                            layer.effects.slice_prob = d.slice_prob;
                            layer.effects.slice_speed = d.slice_speed;
                            layer.effects.slice_axis = d.slice_axis;
                        }
                        "blocks" => {
                            layer.effects.block_size = d.block_size;
                            layer.effects.block_intensity = d.block_intensity;
                            layer.effects.block_prob = d.block_prob;
                            layer.effects.block_speed = d.block_speed;
                        }
                        "glitch" => {
                            layer.effects.jitter_amount = d.jitter_amount;
                            layer.effects.jitter_speed = d.jitter_speed;
                            layer.effects.datamosh = d.datamosh;
                        }
                        "feedback" => {
                            layer.effects.feedback_persistence = d.feedback_persistence;
                            layer.effects.feedback_zoom = d.feedback_zoom;
                            layer.effects.feedback_rotate = d.feedback_rotate;
                            layer.effects.feedback_luma_key = d.feedback_luma_key;
                            layer.effects.feedback_chroma = d.feedback_chroma;
                            layer.effects.feedback_additive = d.feedback_additive;
                        }
                        "transform" => {
                            layer.effects.layer_x = d.layer_x;
                            layer.effects.layer_y = d.layer_y;
                            layer.effects.layer_scale = d.layer_scale;
                            layer.effects.fit_mode = d.fit_mode;
                        }
                        _ => {}
                    }
                    // Resync the mixer for groups that touch audio: "audio"/"audiofx"
                    // (mute/volume/pan/eq/delay) and "source" (resets speed → varispeed).
                    if let Some(audio) = &self.audio {
                        let id = self.layers[index].id;
                        match group.as_str() {
                            "audio" | "audiofx" => audio.set_params(id, self.layers[index].audio),
                            "source" => audio.set_speed(id, self.layers[index].speed),
                            _ => {}
                        }
                    }
                }
            }
            WebAction::SetNtscParam { param, value } => {
                self.ntsc_state.set_param(&param, &value);
            }
            WebAction::SetMasterAudioParam { param, value } => match param.as_str() {
                "master_volume" => {
                    if let Some(v) = value.as_f64() {
                        let db = (v as f32).clamp(-60.0, 6.0);
                        self.master_volume = db;
                        if let Some(audio) = &self.audio {
                            audio.set_master_volume(db);
                        }
                    }
                }
                "limiter" => {
                    if let Some(b) = value.as_bool() {
                        self.master_limiter = b;
                        if let Some(audio) = &self.audio {
                            audio.set_master_limiter(b);
                        }
                    }
                }
                _ => {}
            },
            WebAction::StartExport {
                width,
                height,
                fps,
                duration_secs,
                match_preview,
            } => {
                if self.export_job.is_none() || self.export_job.as_ref().unwrap().is_done() {
                    let patch = patch::PatchState::capture(
                        &self.master_effects,
                        &self.master_automations,
                        &self.layers,
                        &self.ntsc_state.params,
                        self.master_volume,
                        self.master_limiter,
                    );
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let timestamp = now;
                    let output_dir = self
                        .library_folder
                        .as_ref()
                        .map(|f| f.parent().unwrap_or(f).join("renders"))
                        .unwrap_or_else(|| std::path::PathBuf::from("renders"));
                    let output_path = format!(
                        "{}/patch_{}_{width}x{height}.mp4",
                        output_dir.display(),
                        timestamp
                    );
                    let lib_folder = self
                        .library_folder
                        .as_ref()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_else(|| ".".to_string());
                    let config = render_export::ExportConfig {
                        width,
                        height,
                        fps,
                        duration_secs,
                        output_path,
                        bpm: self.tap_bpm,
                        match_preview,
                    };
                    self.export_job =
                        Some(render_export::ExportJob::start(patch, config, &lib_folder));
                    log::info!("Export started");
                }
            }
            WebAction::CancelExport => {
                if let Some(ref job) = self.export_job {
                    job.cancel();
                }
            }
            WebAction::SetAutomation { param, expr } => match automation::Expr::new(&expr) {
                Ok(compiled) => {
                    self.master_automations.insert(param.clone(), compiled);
                    self.master_automation_errors.remove(&param);
                }
                Err(e) => {
                    self.master_automations.remove(&param);
                    self.master_automation_errors.insert(param, e);
                }
            },
            WebAction::ClearAutomation { param } => {
                self.master_automations.remove(&param);
                self.master_automation_errors.remove(&param);
            }
            WebAction::SetLayerAutomation { index, param, expr } => {
                if index < self.layers.len() {
                    let layer = &mut self.layers[index];
                    match automation::Expr::new(&expr) {
                        Ok(compiled) => {
                            layer.automations.insert(param.clone(), compiled);
                            layer.automation_errors.remove(&param);
                        }
                        Err(e) => {
                            layer.automations.remove(&param);
                            layer.automation_errors.insert(param, e);
                        }
                    }
                }
            }
            WebAction::ClearLayerAutomation { index, param } => {
                if index < self.layers.len() {
                    let layer = &mut self.layers[index];
                    layer.automations.remove(&param);
                    layer.automation_errors.remove(&param);
                }
            }
            WebAction::TapTempo => {
                // Stamp the tap against the same clock the formulas use, so the
                // resulting beat phase lines up with `beat` in eval().
                let now = self.start_time.elapsed().as_secs_f32();
                // If it's been a while since the last tap, start a fresh run —
                // the user is tapping a new tempo, not continuing the old one.
                if let Some(&last) = self.tap_times.last() {
                    if now - last > 2.0 {
                        self.tap_times.clear();
                    }
                }
                self.tap_times.push(now);
                // Keep only the most recent handful of taps for the average.
                if self.tap_times.len() > 8 {
                    let start = self.tap_times.len() - 8;
                    self.tap_times.drain(..start);
                }
                // Need at least two taps to measure an interval.
                if self.tap_times.len() >= 2 {
                    let first = self.tap_times[0];
                    let span = now - first;
                    let intervals = (self.tap_times.len() - 1) as f32;
                    let avg = span / intervals;
                    if avg > 0.0 {
                        self.tap_bpm = (60.0 / avg).clamp(30.0, 300.0);
                    }
                }
                // Every tap is a downbeat: reset beat phase to 0 here.
                self.tap_downbeat = now;
            }
            WebAction::SetBpm { value } => {
                // Manual tempo entry — set the tempo but leave beat phase alone.
                self.tap_bpm = value.clamp(30.0, 300.0);
            }
            WebAction::SavePatch { name } => {
                let name = sanitize_patch_name(&name);
                if name.is_empty() {
                    return;
                }
                let dir = patches_dir(&self.library_folder);
                if let Err(e) = std::fs::create_dir_all(&dir) {
                    log::error!("Failed to create patches dir: {e}");
                    return;
                }
                let patch = patch::PatchState::capture(
                    &self.master_effects,
                    &self.master_automations,
                    &self.layers,
                    &self.ntsc_state.params,
                    self.master_volume,
                    self.master_limiter,
                );
                match serde_yaml::to_string(&patch) {
                    Ok(yaml) => {
                        let path = dir.join(format!("{name}.yaml"));
                        if let Err(e) = std::fs::write(&path, &yaml) {
                            log::error!("Failed to write patch {name}: {e}");
                        } else {
                            self.patch_files = scan_patches(&dir);
                            log::info!("Saved patch '{name}'");
                        }
                    }
                    Err(e) => log::error!("Failed to serialize patch: {e}"),
                }
            }
            WebAction::LoadPatch { name } => {
                let name = sanitize_patch_name(&name);
                if name.is_empty() {
                    return;
                }
                let dir = patches_dir(&self.library_folder);
                let path = dir.join(format!("{name}.yaml"));
                let yaml = match std::fs::read_to_string(&path) {
                    Ok(s) => s,
                    Err(e) => {
                        log::error!("Failed to read patch {name}: {e}");
                        return;
                    }
                };
                let patch = match serde_yaml::from_str::<patch::PatchState>(&yaml) {
                    Ok(p) => p,
                    Err(e) => {
                        log::error!("Failed to parse patch {name}: {e}");
                        return;
                    }
                };
                // Rebuild layers from scratch: PatchState::apply only zips against
                // existing layers, so we recreate decoders via add_layer here.
                // Drop every mixer source first — clearing self.layers doesn't send
                // remove commands, so stale sources would otherwise keep playing.
                if let Some(audio) = &self.audio {
                    audio.clear_sources();
                }
                self.layers.clear();
                self.selected_layer = None;
                for cfg in &patch.layers {
                    let resolved = self.library_files.iter().find(|p| {
                        p.file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .as_deref()
                            == Some(&cfg.filename)
                    });
                    match resolved {
                        Some(p) => {
                            let path_str = p.to_string_lossy().to_string();
                            self.add_layer(&path_str);
                            // Apply the saved config, then resync the resulting audio
                            // params/transport to the mixer: add_layer registered the
                            // source with engine defaults, and apply_to_layer only
                            // touched the Layer mirror. Extract Copy values so the
                            // `self.layers` borrow ends before we touch `self.audio`.
                            let resync = self.layers.last_mut().map(|layer| {
                                cfg.apply_to_layer(layer);
                                (layer.id, layer.audio, layer.speed, layer.paused)
                            });
                            if let (Some((id, params, speed, paused)), Some(audio)) =
                                (resync, &self.audio)
                            {
                                audio.set_params(id, params);
                                audio.set_speed(id, speed);
                                audio.set_paused(id, paused);
                            }
                        }
                        None => log::warn!(
                            "Patch '{name}' references missing video '{}', skipping",
                            cfg.filename
                        ),
                    }
                }
                patch.master.apply_to_uniforms(&mut self.master_effects);
                // Restore the master audio bus onto both the App mirror and the
                // live engine (instant, bypassing the mix-ring latency).
                let (mv, ml) = patch.master_audio();
                self.master_volume = mv;
                self.master_limiter = ml;
                if let Some(audio) = &self.audio {
                    audio.set_master_volume(mv);
                    audio.set_master_limiter(ml);
                }
                if let Some(n) = &patch.ntsc {
                    self.ntsc_state.params = n.to_params();
                }
                // Restore saved master automations (replacing whatever was live),
                // so loading a patch reproduces its animated look, not just static
                // values. Per-layer automations aren't stored in patches.
                let (autos, errors) = patch.compile_master_automations();
                self.master_automations = autos;
                self.master_automation_errors = errors;
                log::info!("Loaded patch '{name}' ({} layers)", self.layers.len());
            }
            WebAction::DeletePatch { name } => {
                let name = sanitize_patch_name(&name);
                if name.is_empty() {
                    return;
                }
                let dir = patches_dir(&self.library_folder);
                let path = dir.join(format!("{name}.yaml"));
                if let Err(e) = std::fs::remove_file(&path) {
                    log::error!("Failed to delete patch {name}: {e}");
                }
                self.patch_files = scan_patches(&dir);
            }
            WebAction::FocusWindow => {
                // Bring the native preview (render output) window to the front;
                // un-minimize first in case it was minimized.
                if let Some(window) = &self.window {
                    window.set_minimized(false);
                    window.focus_window();
                }
            }
        }
    }

    /// Push full app state to the web UI via broadcast.
    /// Broadcast a full snapshot of engine state to the browser, once per frame.
    /// This is the "back" half of the browser↔engine contract: the browser sends
    /// `WebAction`s in (drained in the frame loop), and we send this `AppSnapshot`
    /// out so every connected client repaints to match. We build the snapshot by
    /// copying current values out of `self` (layers, master FX, transport, tempo,
    /// export progress), then publish it two ways — store the latest in
    /// `web_state.app` (so newly-connected clients get an immediate paint) and
    /// `tx.send` it to all live WebSocket subscribers.
    fn push_web_state(&self) {
        use web::state::{AppSnapshot, EffectsSnapshot, LayerSnapshot, NtscSnapshot};

        let snapshot = AppSnapshot {
            msg_type: "state".to_string(),
            effects: EffectsSnapshot::from_uniforms(&self.master_effects),
            ntsc: NtscSnapshot::from_params(&self.ntsc_state.params),
            layers: self
                .layers
                .iter()
                .map(|l| LayerSnapshot {
                    id: l.id,
                    filename: l.filename.clone(),
                    visible: l.visible,
                    paused: l.paused,
                    opacity: l.opacity,
                    speed: l.speed,
                    fps: l.fps,
                    loop_start: l.loop_start,
                    loop_end: l.loop_end,
                    blend_mode: l.blend_mode.as_str().to_string(),
                    progress: l.decoder.as_ref().map(|d| d.progress()).unwrap_or(0.0),
                    audio_only: l.audio_only,
                    automations: l
                        .automations
                        .iter()
                        .map(|(k, v)| (k.clone(), v.source.clone()))
                        .collect(),
                    automation_errors: l.automation_errors.clone(),
                    mute: l.audio.mute,
                    volume: l.audio.volume,
                    pan: l.audio.pan,
                    meter: self
                        .audio
                        .as_ref()
                        .map(|a| a.layer_meter(l.id))
                        .unwrap_or(0.0),
                    eq_low: l.audio.eq_low,
                    eq_mid: l.audio.eq_mid,
                    eq_high: l.audio.eq_high,
                    delay_time: l.audio.delay_time,
                    delay_feedback: l.audio.delay_feedback,
                    delay_mix: l.audio.delay_mix,
                    hue_shift: l.effects.hue_shift,
                    saturation: l.effects.saturation,
                    brightness: l.effects.brightness,
                    contrast: l.effects.contrast,
                    pixelate: l.effects.pixelate_size,
                    rgb_split: l.effects.rgb_split,
                    posterize: l.effects.posterize,
                    invert: l.effects.invert > 0.5,
                    wave_amp: l.effects.wave_amp,
                    wave_freq: l.effects.wave_freq,
                    wave_speed: l.effects.wave_speed,
                    wave_axis: l.effects.wave_axis,
                    swirl_angle: l.effects.swirl_angle,
                    swirl_radius: l.effects.swirl_radius,
                    bulge_strength: l.effects.bulge_strength,
                    bulge_radius: l.effects.bulge_radius,
                    chroma_enable: l.effects.chroma_enable > 0.5,
                    chroma_threshold: l.effects.chroma_threshold,
                    chroma_smoothness: l.effects.chroma_smoothness,
                    chroma_spill: l.effects.chroma_spill,
                    chroma_color: rgb01_to_hex(
                        l.effects.chroma_color_r,
                        l.effects.chroma_color_g,
                        l.effects.chroma_color_b,
                    ),
                    chroma_bg_enable: l.effects.chroma_bg_enable > 0.5,
                    chroma_bg_color: rgb01_to_hex(
                        l.effects.chroma_bg_r,
                        l.effects.chroma_bg_g,
                        l.effects.chroma_bg_b,
                    ),
                    slice_intensity: l.effects.slice_intensity,
                    slice_height: l.effects.slice_height,
                    slice_prob: l.effects.slice_prob,
                    slice_speed: l.effects.slice_speed,
                    block_size: l.effects.block_size,
                    block_intensity: l.effects.block_intensity,
                    block_prob: l.effects.block_prob,
                    block_speed: l.effects.block_speed,
                    shift_chroma: l.effects.shift_chroma,
                    slice_axis: l.effects.slice_axis,
                    jitter_amount: l.effects.jitter_amount,
                    jitter_speed: l.effects.jitter_speed,
                    datamosh: l.effects.datamosh,
                    feedback_persistence: l.effects.feedback_persistence,
                    feedback_zoom: l.effects.feedback_zoom,
                    feedback_rotate: l.effects.feedback_rotate,
                    feedback_luma_key: l.effects.feedback_luma_key,
                    feedback_chroma: l.effects.feedback_chroma,
                    feedback_additive: l.effects.feedback_additive,
                    layer_x: l.effects.layer_x,
                    layer_y: l.effects.layer_y,
                    layer_scale: l.effects.layer_scale,
                    fit_mode: l.effects.fit_mode as u32,
                })
                .collect(),
            library: self
                .library_files
                .iter()
                .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                .collect(),
            patches: self.patch_files.clone(),
            paused: self.master_paused,
            framerate: self.master_fps,
            output_ratio: self.output_ratio.clone(),
            output_quality: self.output_quality,
            output_width: self.renderer.as_ref().map(|r| r.output_width).unwrap_or(0),
            output_height: self.renderer.as_ref().map(|r| r.output_height).unwrap_or(0),
            export_progress: self
                .export_job
                .as_ref()
                .map(|j| {
                    if j.is_done() {
                        1.0
                    } else {
                        j.progress.progress_f32()
                    }
                })
                .unwrap_or(0.0),
            export_error: self
                .export_job
                .as_ref()
                .and_then(|j| {
                    if j.is_done() {
                        let err = j.progress.error.lock().unwrap();
                        if err.is_empty() {
                            None
                        } else {
                            Some(err.clone())
                        }
                    } else {
                        None
                    }
                })
                .unwrap_or_default(),
            automations: self
                .master_automations
                .iter()
                .map(|(k, v)| (k.clone(), v.source.clone()))
                .collect(),
            automation_errors: self.master_automation_errors.clone(),
            bpm: self.tap_bpm,
            beat: (self.start_time.elapsed().as_secs_f32() - self.tap_downbeat) * self.tap_bpm
                / 60.0,
            master_volume: self.master_volume,
            master_limiter: self.master_limiter,
            meter: self.audio.as_ref().map(|a| a.meter()).unwrap_or(0.0),
        };

        // Both publishes are best-effort and never block the render loop:
        // `try_write` skips the cached-snapshot update if a reader holds the lock,
        // and `tx.send`'s result is ignored (it errors only when no clients are
        // connected, which is fine). A dropped snapshot just means clients wait
        // one more frame — they're full snapshots, so nothing accumulates.
        if let Ok(mut app) = self.web_state.app.try_write() {
            *app = snapshot.clone();
        }
        let _ = self
            .web_state
            .tx
            .send(serde_json::to_string(&snapshot).unwrap_or_default());
    }
}

/// Parse a `#rrggbb` hex color into sRGB 0..1 components.
fn hex_to_rgb01(hex: &str) -> (f32, f32, f32) {
    let h = hex.trim_start_matches('#');
    if h.len() == 6 {
        if let Ok(n) = u32::from_str_radix(h, 16) {
            let r = ((n >> 16) & 0xff) as f32 / 255.0;
            let g = ((n >> 8) & 0xff) as f32 / 255.0;
            let b = (n & 0xff) as f32 / 255.0;
            return (r, g, b);
        }
    }
    (0.0, 1.0, 0.0) // default green
}

/// Format sRGB 0..1 components into a `#rrggbb` hex string.
fn rgb01_to_hex(r: f32, g: f32, b: f32) -> String {
    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", to_u8(r), to_u8(g), to_u8(b))
}

/// Scan a directory for video files, returning sorted list of paths.

fn scan_folder(folder: &PathBuf) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && is_supported_media(p))
        .collect();
    files.sort();
    files
}

/// Folder where patch YAML files are stored. Mirrors the renders folder:
/// a `patches/` dir next to the library folder (i.e. project root), falling
/// back to `./patches` when no library folder is set.
fn patches_dir(library_folder: &Option<PathBuf>) -> PathBuf {
    library_folder
        .as_ref()
        .map(|f| f.parent().unwrap_or(f).join("patches"))
        .unwrap_or_else(|| PathBuf::from("patches"))
}

/// List saved patch names (file stems, without the `.yaml` extension), sorted.
/// Returns an empty list if the folder does not exist yet.
fn scan_patches(dir: &PathBuf) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().map(|x| x == "yaml").unwrap_or(false))
        .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
        .collect();
    names.sort();
    names
}

/// Sanitize a user-supplied patch name into a safe filename stem. Keeps
/// alphanumerics, dash, underscore and space; replaces anything else with `_`.
/// This prevents path traversal (e.g. `../`) and other unsafe characters.
fn sanitize_patch_name(name: &str) -> String {
    let cleaned: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' {
                c
            } else {
                '_'
            }
        })
        .collect();
    cleaned.trim().chars().take(48).collect()
}

/// Generate thumbnails and preview frames for all library files using ffmpeg CLI.
/// Thumbnails are generated first (fast), then preview frames in a second pass.
fn generate_thumbnails(files: &[PathBuf], web_state: Arc<web::state::WebState>) {
    let paths: Vec<PathBuf> = files.to_vec();
    std::thread::Builder::new()
        .name("thumb-gen".into())
        .spawn(move || {
            use std::process::Command;
            use std::sync::atomic::{AtomicUsize, Ordering};

            let count = Arc::new(AtomicUsize::new(0));
            let total = paths.len();

            // Pass 1: Generate static thumbnails (fast, parallel batches of 8)
            for chunk in paths.chunks(8) {
                let handles: Vec<_> = chunk
                    .iter()
                    .map(|path| {
                        let path = path.clone();
                        let web_state = web_state.clone();
                        let count = count.clone();
                        std::thread::spawn(move || {
                            let filename = match path.file_name() {
                                Some(n) => n.to_string_lossy().to_string(),
                                None => return,
                            };

                            let output = Command::new("ffmpeg")
                                .args([
                                    "-i",
                                    &path.to_string_lossy(),
                                    "-vframes",
                                    "1",
                                    "-vf",
                                    "scale=180:-1",
                                    "-f",
                                    "image2pipe",
                                    "-vcodec",
                                    "mjpeg",
                                    "-q:v",
                                    "8",
                                    "-loglevel",
                                    "error",
                                    "pipe:1",
                                ])
                                .output();

                            match output {
                                Ok(result)
                                    if result.status.success() && !result.stdout.is_empty() =>
                                {
                                    if let Ok(mut cache) = web_state.thumbnails.write() {
                                        cache.insert(filename, result.stdout);
                                    }
                                    count.fetch_add(1, Ordering::Relaxed);
                                }
                                Ok(result) => {
                                    let err = String::from_utf8_lossy(&result.stderr);
                                    log::warn!("Thumb: ffmpeg failed for {filename}: {err}");
                                }
                                Err(e) => {
                                    log::warn!("Thumb: can't run ffmpeg for {filename}: {e}");
                                }
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    let _ = h.join();
                }
            }

            log::info!(
                "Generated {}/{total} thumbnails",
                count.load(Ordering::Relaxed)
            );

            // Pass 2: Generate preview frames (~8 per video, parallel batches of 4)
            let preview_count = Arc::new(AtomicUsize::new(0));
            for chunk in paths.chunks(4) {
                let handles: Vec<_> = chunk
                    .iter()
                    .map(|path| {
                        let path = path.clone();
                        let web_state = web_state.clone();
                        let preview_count = preview_count.clone();
                        std::thread::spawn(move || {
                            let filename = match path.file_name() {
                                Some(n) => n.to_string_lossy().to_string(),
                                None => return,
                            };

                            // Get video duration with ffprobe
                            let duration = Command::new("ffprobe")
                                .args([
                                    "-v",
                                    "error",
                                    "-show_entries",
                                    "format=duration",
                                    "-of",
                                    "csv=p=0",
                                    &path.to_string_lossy(),
                                ])
                                .output()
                                .ok()
                                .and_then(|o| String::from_utf8(o.stdout).ok())
                                .and_then(|s| s.trim().parse::<f64>().ok())
                                .unwrap_or(0.0);

                            if duration < 0.5 {
                                return;
                            }

                            const NUM_FRAMES: usize = 8;
                            let mut frames = Vec::with_capacity(NUM_FRAMES);

                            for i in 0..NUM_FRAMES {
                                let seek = duration * (i as f64) / (NUM_FRAMES as f64);
                                let seek_str = format!("{:.2}", seek);

                                let output = Command::new("ffmpeg")
                                    .args([
                                        "-ss",
                                        &seek_str,
                                        "-i",
                                        &path.to_string_lossy(),
                                        "-vframes",
                                        "1",
                                        "-vf",
                                        "scale=180:-1",
                                        "-f",
                                        "image2pipe",
                                        "-vcodec",
                                        "mjpeg",
                                        "-q:v",
                                        "10",
                                        "-loglevel",
                                        "error",
                                        "pipe:1",
                                    ])
                                    .output();

                                if let Ok(result) = output {
                                    if result.status.success() && !result.stdout.is_empty() {
                                        frames.push(result.stdout);
                                    }
                                }
                            }

                            if !frames.is_empty() {
                                if let Ok(mut cache) = web_state.preview_frames.write() {
                                    cache.insert(filename, frames);
                                }
                                preview_count.fetch_add(1, Ordering::Relaxed);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    let _ = h.join();
                }
            }

            log::info!(
                "Generated {}/{total} preview strips",
                preview_count.load(Ordering::Relaxed)
            );
        })
        .ok();
}

// winit drives the app through this trait: instead of us owning the main loop,
// winit owns it and calls these methods back when things happen. `resumed` fires
// when the app becomes active (once at startup on desktop, but it can fire again
// after suspend on mobile), and `window_event` fires for every window message
// (input, resize, redraw request). This inversion-of-control is why all our
// per-frame work lives inside `window_event`'s `RedrawRequested` arm.
impl ApplicationHandler for App {
    // Build the window + GPU here, NOT in `App::new`. winit only hands us an
    // `ActiveEventLoop` (the thing that can create a window) at this point, and
    // the renderer/egui need a live window to attach to. The early-return guards
    // against re-running this if `resumed` fires more than once.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Output canvas starts at the master output size (deliberate control).
        // Source clips stretch into it (user accepted stretching).
        let (output_width, output_height) = output_dims(&self.output_ratio, self.output_quality);
        self.master_effects.resolution = [output_width as f32, output_height as f32];

        log::info!("Output: {}x{}", output_width, output_height);

        // Preview window defaults to a small square, independent of the output
        // resolution. The egui preview keeps the output aspect ratio via
        // fit_to_area, so non-square output is letterboxed inside the square frame.
        const PREVIEW_WINDOW_SIZE: u32 = 600;

        let window_attrs = WindowAttributes::default()
            .with_title("collide-o-scope")
            .with_inner_size(winit::dpi::LogicalSize::new(
                PREVIEW_WINDOW_SIZE,
                PREVIEW_WINDOW_SIZE,
            ));

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        let renderer = Renderer::new(window.clone(), output_width, output_height);

        configure_fonts(&self.egui_ctx);

        let egui_winit = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let mut egui_renderer = egui_wgpu::Renderer::new(
            &renderer.device,
            renderer.config.format,
            egui_wgpu::RendererOptions::default(),
        );

        let video_egui_texture_id = egui_renderer.register_native_texture(
            &renderer.device,
            &renderer.output_view,
            wgpu::FilterMode::Linear,
        );

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.egui_winit = Some(egui_winit);
        self.egui_renderer = Some(egui_renderer);
        self.video_egui_texture_id = Some(video_egui_texture_id);

        if let Some(path) = self.initial_video.take() {
            self.add_layer(&path);
        }
    }

    // Called for every window message. We give egui first dibs on the event
    // (so clicks/keystrokes over the YAML editor go to the UI, not the app); if
    // egui `consumed` it, we bail early. Otherwise we `match` the event ourselves.
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(egui_winit) = &mut self.egui_winit {
            let response = egui_winit.on_window_event(self.window.as_ref().unwrap(), &event);
            if response.consumed {
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(new_size.width, new_size.height);
                }
            }

            WindowEvent::ScaleFactorChanged { .. } => {
                let size = self.window.as_ref().unwrap().inner_size();
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
            }

            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
            }

            WindowEvent::DroppedFile(path) => {
                if path.is_dir() {
                    self.set_library_folder(path);
                } else if is_supported_media(&path) {
                    if let Some(path_str) = path.to_str() {
                        let path_owned = path_str.to_string();
                        self.add_layer(&path_owned);
                    }
                }
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key,
                        state,
                        ..
                    },
                ..
            } => {
                use winit::keyboard::{KeyCode, PhysicalKey};

                // Ctrl+key shortcuts (editor toggle, save, load)
                if state == winit::event::ElementState::Pressed && self.modifiers.control_key() {
                    match physical_key {
                        PhysicalKey::Code(KeyCode::KeyE) => {
                            self.yaml_editor.active = !self.yaml_editor.active;
                            return;
                        }
                        PhysicalKey::Code(KeyCode::KeyS) => {
                            patch::editor::save_patch(
                                &self.master_effects,
                                &self.master_automations,
                                &self.layers,
                                &self.ntsc_state.params,
                                self.master_volume,
                                self.master_limiter,
                            );
                            return;
                        }
                        PhysicalKey::Code(KeyCode::KeyO) => {
                            if let Some((autos, errors)) = patch::editor::load_patch(
                                &mut self.master_effects,
                                &mut self.layers,
                                &mut self.ntsc_state.params,
                            ) {
                                self.master_automations = autos;
                                self.master_automation_errors = errors;
                            }
                            return;
                        }
                        _ => {}
                    }
                }

                let shift = self.modifiers.shift_key();
                let action = map_key(physical_key, state, shift);

                if let Some(idx) = self.selected_layer {
                    if let Some(layer) = self.layers.get_mut(idx) {
                        match apply_action(action, &mut layer.effects) {
                            ControlFlow::Quit => event_loop.exit(),
                            ControlFlow::TogglePause => layer.paused = !layer.paused,
                            ControlFlow::ToggleFullscreen => {
                                if let Some(window) = &self.window {
                                    let current = window.fullscreen();
                                    if current.is_some() {
                                        window.set_fullscreen(None);
                                    } else {
                                        window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                                    }
                                }
                            }
                            ControlFlow::Continue => {}
                        }
                    }
                } else {
                    let mut dummy = effects::EffectUniforms::default();
                    match apply_action(action, &mut dummy) {
                        ControlFlow::Quit => event_loop.exit(),
                        ControlFlow::ToggleFullscreen => {
                            if let Some(window) = &self.window {
                                let current = window.fullscreen();
                                if current.is_some() {
                                    window.set_fullscreen(None);
                                } else {
                                    window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            // THE per-frame pipeline. winit asks us to redraw; we run the whole
            // chain here: frame-rate gate → advance video → drain web actions →
            // build the egui frame → evaluate automation → render layers + master
            // FX → optional NTSC → composite egui → present → ask for the next
            // redraw. Everything that makes a frame happen lives in this arm.
            WindowEvent::RedrawRequested => {
                // Skip work while the window is minimised (zero-size surface would
                // fail to render); just re-arm a redraw and bail.
                let win_size = self.window.as_ref().unwrap().inner_size();
                if win_size.width == 0 || win_size.height == 0 {
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                    return;
                }

                // Frame-rate gate: redraws can fire faster than 30fps, so we only
                // do real work once at least FRAME_DURATION has passed. `content_tick`
                // counts the frames that pass the gate (wrapping_add avoids overflow
                // panic when it eventually wraps around the u64 max).
                let now = Instant::now();
                if now - self.last_frame_time >= FRAME_DURATION {
                    self.last_frame_time = now;
                    self.content_tick = self.content_tick.wrapping_add(1);

                    // Frame-hold / stutter: content (decode + the animated `time`
                    // uniform) advances every `stride` render ticks. Only 30/k rates
                    // land evenly on the fixed 30fps tick grid, so the UI exposes
                    // those presets; stride = round(30 / master_fps). At 30 this is 1
                    // (advance every tick); web drain/push/present stay below the gate.
                    let stride = (TARGET_FPS as f32 / self.master_fps).round().max(1.0) as u64;
                    let advance_content = self.content_tick % stride == 0;
                    if advance_content {
                        // Real wall-clock time since the last content advance — drives
                        // catch-up decode so footage tracks the same clock as
                        // content_time / the animated `time` uniform (stays in sync
                        // even when a tick runs long, e.g. the blocking VHS readback).
                        let dt = (now - self.last_content_advance).as_secs_f32();
                        self.last_content_advance = now;
                        self.content_time = self.start_time.elapsed().as_secs_f32();

                        // Advance each layer's footage by the real elapsed time.
                        if !self.master_paused {
                            for layer in &mut self.layers {
                                if !layer.paused {
                                    let queue = &self.renderer.as_ref().unwrap().queue;
                                    layer.advance(dt, queue);
                                }
                            }
                        }
                    }

                    // Drain the web UI's action queue. The browser pushes
                    // `WebAction`s onto this shared Mutex<Vec> from the server
                    // thread; we `try_lock` (non-blocking — never stall the render
                    // loop on the lock) and `drain(..)` everything queued since last
                    // frame into a local Vec, releasing the lock before we apply
                    // them. `unwrap_or_default()` = "empty list if the lock is busy".
                    let pending_actions: Vec<_> = self
                        .web_state
                        .actions
                        .try_lock()
                        .map(|mut a| a.drain(..).collect())
                        .unwrap_or_default();
                    for action in pending_actions {
                        self.handle_web_action(action);
                    }

                    // Build minimal egui frame (video display only, no UI panels)
                    let window = self.window.as_ref().unwrap();
                    let egui_winit = self.egui_winit.as_mut().unwrap();
                    let raw_input = egui_winit.take_egui_input(window);

                    let video_egui_texture_id = self.video_egui_texture_id;
                    let output_width = self.renderer.as_ref().unwrap().output_width;
                    let output_height = self.renderer.as_ref().unwrap().output_height;

                    let full_output = self.egui_ctx.run_ui(raw_input, |ctx| {
                        // Full-window video output (no UI panels)
                        egui::CentralPanel::default()
                            .frame(egui::Frame::NONE.fill(egui::Color32::BLACK))
                            .show(ctx, |ui| {
                                if let Some(tex_id) = video_egui_texture_id {
                                    let available = ui.available_size();
                                    let aspect = output_width as f32 / output_height as f32;
                                    let (w, h) = fit_to_area(available.x, available.y, aspect);
                                    ui.centered_and_justified(|ui| {
                                        ui.image(egui::load::SizedTexture::new(
                                            tex_id,
                                            egui::vec2(w, h),
                                        ));
                                    });
                                }
                            });
                    });

                    // Push full state to web UI
                    self.push_web_state();

                    let window = self.window.as_ref().unwrap();
                    let egui_winit = self.egui_winit.as_mut().unwrap();
                    egui_winit.handle_platform_output(window, full_output.platform_output);

                    let tris = self
                        .egui_ctx
                        .tessellate(full_output.shapes, full_output.pixels_per_point);

                    // Set time uniform on all effects (drives animated noise/breathing).
                    // Driven by the HELD content_time so animated effects stutter in
                    // lockstep with held video frames at low master_fps.
                    let elapsed = self.content_time;
                    // Musical time for beat-synced formulas: `beat` counts beats since
                    // the last tap downbeat at the current tempo. It rides the real wall
                    // clock (not content_time) so tempo sync stays locked to the music
                    // even when a low master_fps stutters the content clock, and so its
                    // phase matches the wall-clock `tap_downbeat` anchor set on each tap.
                    let bpm = self.tap_bpm;
                    let beat =
                        (self.start_time.elapsed().as_secs_f32() - self.tap_downbeat) * bpm / 60.0;
                    for layer in &mut self.layers {
                        layer.effects.time = elapsed;
                        // Evaluate any automated params against `t = elapsed`.
                        // speed/opacity/fps are LAYER-level fields (not effect
                        // uniforms), so route them to the layer directly —
                        // otherwise set_by_name drops them and the snapshot keeps
                        // sending the unchanged field, freezing the slider.
                        for (param, expr) in &layer.automations {
                            let v = expr.eval(elapsed, beat, bpm);
                            match param.as_str() {
                                "opacity" => layer.opacity = v.clamp(0.0, 1.0),
                                "speed" => layer.speed = v.clamp(0.25, 4.0),
                                "fps" => layer.fps = v.clamp(1.0, 60.0),
                                _ => layer.effects.set_by_name(param, v),
                            }
                        }
                        // Recompute fit scale each frame so it tracks canvas resize.
                        let (fx, fy) = fit_scale(
                            layer.effects.fit_mode,
                            layer.effects.resolution[0],
                            layer.effects.resolution[1],
                            output_width as f32,
                            output_height as f32,
                        );
                        layer.effects.fit_scale_x = fx;
                        layer.effects.fit_scale_y = fy;
                    }
                    self.master_effects.time = elapsed;
                    // Evaluate master automations (same `t` so live matches export).
                    for (param, expr) in &self.master_automations {
                        self.master_effects
                            .set_by_name(param, expr.eval(elapsed, beat, bpm));
                    }

                    // A command encoder records GPU commands into a buffer; nothing
                    // executes until we `queue.submit(encoder.finish())`. We record
                    // the layer compositing pass and the master-FX pass into one
                    // encoder, then (below) either submit immediately or, if NTSC is
                    // on, submit early so the CPU can read the pixels back.
                    let renderer = self.renderer.as_mut().unwrap();
                    let mut encoder =
                        renderer
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Frame Encoder"),
                            });
                    renderer.render_layers(&mut encoder, &self.layers);
                    renderer.render_master_effects(&mut encoder, &self.master_effects);

                    // NTSC/VHS post-process at half resolution. The GPU does the
                    // down/upscale (bilinear); ntsc-rs runs on the half-res pixels,
                    // so only ~1/4 the data crosses the bus and there are no CPU
                    // resampling loops. Keeps the live path off slow-motion.
                    if self.ntsc_state.params.enabled {
                        // GPU: downscale composite[0] → half texture, then submit so it's ready.
                        renderer.downscale_to_half(&mut encoder);
                        renderer.queue.submit(std::iter::once(encoder.finish()));

                        let half_w = (renderer.output_width / 2).max(1);
                        let half_h = (renderer.output_height / 2).max(1);

                        // CPU: read half-res, run ntsc-rs at half-res, write back.
                        let mut pixels = renderer.readback_half();
                        self.ntsc_state.apply_full_res(&mut pixels, half_w, half_h);
                        renderer.write_half(&pixels);

                        // Fresh encoder: GPU upscale half → composite[0]; egui pass appends after.
                        encoder = renderer.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("Post-NTSC Encoder"),
                            },
                        );
                        renderer.upscale_half_to_composite(&mut encoder);
                    }

                    let egui_renderer = self.egui_renderer.as_mut().unwrap();
                    for (id, image_delta) in &full_output.textures_delta.set {
                        egui_renderer.update_texture(
                            &renderer.device,
                            &renderer.queue,
                            *id,
                            image_delta,
                        );
                    }

                    let screen_desc = ScreenDescriptor {
                        size_in_pixels: [renderer.config.width, renderer.config.height],
                        pixels_per_point: full_output.pixels_per_point,
                    };

                    egui_renderer.update_buffers(
                        &renderer.device,
                        &renderer.queue,
                        &mut encoder,
                        &tris,
                        &screen_desc,
                    );

                    // Acquire the swapchain image we'll draw the final frame into.
                    // This can fail in normal situations: `Outdated`/`Lost` mean the
                    // surface config is stale (usually a resize or GPU reset), so we
                    // reconfigure and bail this frame — the re-armed redraw will retry
                    // with a fresh surface. The catch-all `_` arm (e.g. `Timeout`)
                    // likewise just skips this frame rather than crashing.
                    let surface_texture = match renderer.surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(t)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                        wgpu::CurrentSurfaceTexture::Outdated
                        | wgpu::CurrentSurfaceTexture::Lost => {
                            let size = window.inner_size();
                            let r = self.renderer.as_mut().unwrap();
                            if size.width > 0 && size.height > 0 {
                                r.resize(size.width, size.height);
                            } else {
                                r.reconfigure_surface();
                            }
                            if let Some(w) = &self.window {
                                w.request_redraw();
                            }
                            return;
                        }
                        _ => {
                            if let Some(w) = &self.window {
                                w.request_redraw();
                            }
                            return;
                        }
                    };
                    let surface_view = surface_texture
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    {
                        let mut render_pass = encoder
                            .begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("egui Pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &surface_view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color {
                                            r: 0.1,
                                            g: 0.1,
                                            b: 0.1,
                                            a: 1.0,
                                        }),
                                        store: wgpu::StoreOp::Store,
                                    },
                                    depth_slice: None,
                                })],
                                depth_stencil_attachment: None,
                                ..Default::default()
                            })
                            .forget_lifetime();

                        egui_renderer.render(&mut render_pass, &tris, &screen_desc);
                    }

                    for id in &full_output.textures_delta.free {
                        egui_renderer.free_texture(id);
                    }

                    // Submit all recorded commands to the GPU, then `present()` hands
                    // the finished image to the compositor for display on screen.
                    renderer.queue.submit(std::iter::once(encoder.finish()));
                    surface_texture.present();
                }

                // Always ask winit for another redraw, even if the frame gate above
                // skipped the work this time — this keeps the render loop running
                // continuously (a "render on every vsync" style loop).
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }
}

/// Returns an optional path to add as a new layer (needs device access from caller).
fn build_ui(
    ctx: &egui::Context,
    layers: &mut Vec<Layer>,
    selected_layer: &mut Option<usize>,
    master_effects: &mut effects::EffectUniforms,
    master_paused: &mut bool,
    yaml_editor: &mut patch::editor::EditorState,
    library_folder: &mut Option<PathBuf>,
    library_files: &mut Vec<PathBuf>,
    video_egui_texture_id: Option<egui::TextureId>,
    output_width: u32,
    output_height: u32,
) -> Option<String> {
    let mut add_layer_path: Option<String> = None;
    let mut remove_layer: Option<usize> = None;
    let mut move_layer: Option<(usize, usize)> = None;
    let mut change_folder = false;

    // LEFT panel: Layers with collapsible per-layer controls
    egui::Panel::left("left_panel")
        .min_size(240.0)
        .default_size(280.0)
        .show(ctx, |ui| {
            // View switcher tabs
            ui.horizontal(|ui| {
                if ui.selectable_label(!yaml_editor.active, "UI").clicked() {
                    yaml_editor.active = false;
                }
                if ui.selectable_label(yaml_editor.active, "Code").clicked() {
                    yaml_editor.active = true;
                }
            });
            ui.separator();

            if yaml_editor.active {
                // Code view
                patch::editor::build_yaml_editor_content(ui, layers, master_effects, yaml_editor);
            } else {
                // UI view: collapsible layers
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let layer_count = layers.len();

                        for i in 0..layer_count {
                            let is_selected = *selected_layer == Some(i);

                            // Layer header row with controls
                            ui.horizontal(|ui| {
                                // Grip handle for reorder
                                let grip = ui.label(egui::RichText::new("⠿").weak());
                                if grip.dragged() {
                                    let delta = grip.drag_delta().y;
                                    if delta < -16.0 && i > 0 {
                                        move_layer = Some((i, i - 1));
                                    } else if delta > 16.0 && i < layer_count - 1 {
                                        move_layer = Some((i, i + 1));
                                    }
                                }

                                // Visibility toggle
                                let eye = if layers[i].visible { "👁" } else { "ꞏ" };
                                if ui.small_button(eye).clicked() {
                                    layers[i].visible = !layers[i].visible;
                                }

                                // Remove button
                                if ui.small_button("×").clicked() {
                                    remove_layer = Some(i);
                                }
                            });

                            // Collapsible header with layer name
                            let header_id = ui.make_persistent_id(format!("layer_col_{i}"));
                            let header = egui::CollapsingHeader::new(
                                egui::RichText::new(&layers[i].filename).strong(),
                            )
                            .id_salt(header_id)
                            .default_open(is_selected);

                            header.show(ui, |ui| {
                                let layer = &mut layers[i];

                                // Transport
                                ui.horizontal(|ui| {
                                    if ui.button(if layer.paused { "▶" } else { "⏸" }).clicked()
                                    {
                                        layer.paused = !layer.paused;
                                    }
                                    if ui.button("Reset FX").clicked() {
                                        layer.effects.reset();
                                    }
                                });

                                labeled_slider(
                                    ui,
                                    "Speed",
                                    egui::Slider::new(&mut layer.speed, 0.25..=4.0)
                                        .logarithmic(true)
                                        .custom_formatter(|v, _| format!("{:.2}×", v)),
                                );
                                labeled_slider(
                                    ui,
                                    "FPS",
                                    egui::Slider::new(&mut layer.fps, 1.0..=60.0)
                                        .step_by(1.0)
                                        .custom_formatter(|v, _| format!("{:.0}", v)),
                                );
                                labeled_slider(
                                    ui,
                                    "Opacity",
                                    egui::Slider::new(&mut layer.opacity, 0.0..=1.0),
                                );
                                ui.horizontal(|ui| {
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(78.0, ui.spacing().interact_size.y),
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                            ui.label("Blend");
                                        },
                                    );
                                    egui::ComboBox::from_id_salt(format!("blend_mode_{i}"))
                                        .selected_text(layer.blend_mode.label())
                                        .show_ui(ui, |ui| {
                                            for mode in BlendMode::ALL {
                                                ui.selectable_value(
                                                    &mut layer.blend_mode,
                                                    *mode,
                                                    mode.label(),
                                                );
                                            }
                                        });
                                });

                                ui.add_space(4.0);
                                effects_sliders(ui, &mut layer.effects, &format!("layer_{i}"));
                            });

                            ui.separator();
                        }

                        if layers.is_empty() {
                            ui.weak("No active layers");
                            ui.weak("Drop files or add from library →");
                        }
                    });
            }
        });

    // RIGHT panel: Master controls + Library
    egui::Panel::right("right_panel")
        .min_size(260.0)
        .default_size(300.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // === MASTER ===
                    ui.heading("Master");
                    ui.separator();

                    // Transport
                    ui.horizontal(|ui| {
                        if ui
                            .button(if *master_paused {
                                "▶ Play All"
                            } else {
                                "⏸ Pause All"
                            })
                            .clicked()
                        {
                            *master_paused = !*master_paused;
                        }
                        if ui.button("Reset FX").clicked() {
                            master_effects.reset();
                        }
                    });

                    ui.add_space(4.0);
                    effects_sliders(ui, master_effects, "master");

                    ui.add_space(12.0);
                    ui.separator();

                    // === LIBRARY ===
                    ui.heading("Library");
                    ui.separator();

                    // Folder path + change button
                    ui.horizontal(|ui| {
                        let folder_label = library_folder
                            .as_ref()
                            .and_then(|f| f.file_name())
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| "No folder".into());
                        ui.label(folder_label);
                        if ui.small_button("…").clicked() {
                            change_folder = true;
                        }
                    });

                    ui.add_space(4.0);
                    ui.separator();

                    // File list
                    for file in library_files.iter() {
                        let name = file
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default();

                        let response = ui.selectable_label(false, &name);
                        if response.double_clicked() {
                            if let Some(path_str) = file.to_str() {
                                add_layer_path = Some(path_str.to_string());
                            }
                        }
                        response.on_hover_text("Double-click to add as layer");
                    }

                    if library_files.is_empty() {
                        ui.weak("No video files");
                        ui.weak("Drop a folder or click …");
                    }
                });
        });

    // Central panel: video output
    egui::CentralPanel::default().show(ctx, |ui| {
        if let Some(tex_id) = video_egui_texture_id {
            let available = ui.available_size();
            let aspect = output_width as f32 / output_height as f32;
            let (w, h) = fit_to_area(available.x, available.y, aspect);
            ui.centered_and_justified(|ui| {
                ui.image(egui::load::SizedTexture::new(tex_id, egui::vec2(w, h)));
            });
        }
    });

    // --- Apply deferred actions ---

    // Change folder (opens native dialog)
    if change_folder {
        if let Some(folder) = rfd::FileDialog::new().pick_folder() {
            *library_files = scan_folder(&folder);
            *library_folder = Some(folder);
        }
    }

    // Move layer
    if let Some((from, to)) = move_layer {
        layers.swap(from, to);
        if *selected_layer == Some(from) {
            *selected_layer = Some(to);
        } else if *selected_layer == Some(to) {
            *selected_layer = Some(from);
        }
    }

    // Remove layer
    if let Some(idx) = remove_layer {
        layers.remove(idx);
        if layers.is_empty() {
            *selected_layer = None;
        } else if let Some(sel) = *selected_layer {
            if sel >= layers.len() {
                *selected_layer = Some(layers.len() - 1);
            }
        }
    }

    add_layer_path
}

/// Inline labeled slider: label on left (fixed width), slider fills remaining space.
fn labeled_slider(ui: &mut egui::Ui, label: &str, slider: egui::Slider<'_>) {
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(78.0, ui.spacing().interact_size.y),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.label(label);
            },
        );
        ui.add(slider);
    });
}

/// Inline labeled checkbox.
fn labeled_checkbox(ui: &mut egui::Ui, label: &str, value: &mut bool) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(78.0, ui.spacing().interact_size.y),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.label(label);
            },
        );
        changed = ui.checkbox(value, "").changed();
    });
    changed
}

/// Shared effects slider UI — used for both master and per-layer effects.
fn effects_sliders(ui: &mut egui::Ui, effects: &mut effects::EffectUniforms, id_prefix: &str) {
    // --- Digital effects ---
    ui.label(egui::RichText::new("Digital").weak().size(11.0));

    labeled_slider(
        ui,
        "Pixelate",
        egui::Slider::new(&mut effects.pixelate_size, 1.0..=32.0)
            .step_by(1.0)
            .custom_formatter(|v, _| format!("{:.0}", v)),
    );
    labeled_slider(
        ui,
        "RGB Split",
        egui::Slider::new(&mut effects.rgb_split, 0.0..=30.0)
            .step_by(1.0)
            .custom_formatter(|v, _| format!("{:.0}", v)),
    );
    labeled_slider(
        ui,
        "Hue",
        egui::Slider::new(&mut effects.hue_shift, -180.0..=180.0)
            .step_by(1.0)
            .suffix("°"),
    );
    labeled_slider(
        ui,
        "Saturation",
        egui::Slider::new(&mut effects.saturation, -1.0..=1.0),
    );
    labeled_slider(
        ui,
        "Brightness",
        egui::Slider::new(&mut effects.brightness, -1.0..=1.0),
    );
    labeled_slider(
        ui,
        "Contrast",
        egui::Slider::new(&mut effects.contrast, -1.0..=1.0),
    );
    labeled_slider(
        ui,
        "Posterize",
        egui::Slider::new(&mut effects.posterize, 0.0..=16.0)
            .step_by(1.0)
            .custom_formatter(|v, _| {
                if v < 2.0 {
                    "Off".to_string()
                } else {
                    format!("{:.0}", v)
                }
            }),
    );

    let mut invert_on = effects.invert > 0.5;
    if labeled_checkbox(ui, "Invert", &mut invert_on) {
        effects.invert = if invert_on { 1.0 } else { 0.0 };
    }

    ui.add_space(4.0);
    ui.separator();

    // --- Analog effects ---
    ui.label(egui::RichText::new("Analog").weak().size(11.0));

    labeled_slider(
        ui,
        "Grain",
        egui::Slider::new(&mut effects.grain_intensity, 0.0..=0.3),
    );

    if effects.grain_intensity > 0.0 {
        labeled_slider(
            ui,
            "  Size",
            egui::Slider::new(&mut effects.grain_size, 1.0..=4.0)
                .step_by(1.0)
                .custom_formatter(|v, _| format!("{:.0}", v)),
        );
        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(
                egui::vec2(78.0, ui.spacing().interact_size.y),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.label("  Algo");
                },
            );
            egui::ComboBox::from_id_salt(format!("grain_algo_{id_prefix}"))
                .width(90.0)
                .selected_text(match effects.grain_algo as i32 {
                    1 => "Perlin",
                    2 => "Salt&Pepper",
                    3 => "Blue",
                    _ => "Gaussian",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut effects.grain_algo, 0.0, "Gaussian");
                    ui.selectable_value(&mut effects.grain_algo, 1.0, "Perlin");
                    ui.selectable_value(&mut effects.grain_algo, 2.0, "Salt&Pepper");
                    ui.selectable_value(&mut effects.grain_algo, 3.0, "Blue");
                });
        });
        let mut color = effects.color_grain > 0.5;
        if labeled_checkbox(ui, "  Color", &mut color) {
            effects.color_grain = if color { 1.0 } else { 0.0 };
        }
    }

    labeled_slider(
        ui,
        "Vignette",
        egui::Slider::new(&mut effects.vignette, 0.0..=1.5),
    );
    labeled_slider(
        ui,
        "Drift",
        egui::Slider::new(&mut effects.color_drift, 0.0..=0.02),
    );

    ui.add_space(4.0);
    ui.separator();

    // --- Motion effects ---
    ui.label(egui::RichText::new("Motion").weak().size(11.0));

    labeled_slider(
        ui,
        "Bth Scale",
        egui::Slider::new(&mut effects.breathe_scale, 0.0..=0.05),
    );
    labeled_slider(
        ui,
        "Bth Rotate",
        egui::Slider::new(&mut effects.breathe_rotation, 0.0..=2.0).suffix("°"),
    );
    labeled_slider(
        ui,
        "Bth Drift",
        egui::Slider::new(&mut effects.breathe_position, 0.0..=0.02),
    );
}

/// Fit a rectangle with given aspect ratio into available width/height.
fn fit_to_area(max_w: f32, max_h: f32, aspect: f32) -> (f32, f32) {
    let w = max_w;
    let h = w / aspect;
    if h <= max_h {
        (w, h)
    } else {
        let h = max_h;
        let w = h * aspect;
        (w, h)
    }
}

fn configure_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    // Fonts are vendored under assets/fonts/ and embedded at compile time, so
    // the build is self-contained (works on CI and any clone, not just a Mac
    // with these installed in ~/Library/Fonts). IBM Plex ships under the SIL
    // Open Font License, which permits redistribution.
    fonts.font_data.insert(
        "IBMPlexSans".to_owned(),
        Arc::new(egui::FontData::from_static(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/fonts/IBMPlexSans-Regular.otf"
        )))),
    );

    fonts.font_data.insert(
        "IBMPlexMono".to_owned(),
        Arc::new(egui::FontData::from_static(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/fonts/IBMPlexMono-Regular.otf"
        )))),
    );

    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "IBMPlexSans".to_owned());

    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "IBMPlexMono".to_owned());

    ctx.set_fonts(fonts);
}

/// Process entry point. Parses CLI args, routes to the headless `render`
/// subcommand if requested, otherwise: starts the web control-panel server,
/// opens the browser panel, builds the `App`, and hands it to winit's event
/// loop (which blocks here until the window closes). The `App`'s window + GPU
/// aren't created yet at this point — that happens later in `resumed()`.
fn main() {
    // env_logger reads the RUST_LOG env var to decide what log levels to print
    // (e.g. `RUST_LOG=info cargo run`). Without it, `log::info!` calls are no-ops.
    env_logger::init();

    // `std::env::args()` yields the program name as [0], then the CLI arguments.
    let args: Vec<String> = std::env::args().collect();

    // Headless subcommands bail out before any window / web server is created.
    if args.get(1).map(|s| s.as_str()) == Some("render") {
        if let Err(e) = run_cli_render(&args[2..]) {
            eprintln!("render failed: {e}");
            std::process::exit(1);
        }
        return;
    }

    // The matrix view is the default panel. `--classic` opens the legacy
    // ("classic") panel instead — kept around for posterity.
    let classic = args.iter().any(|a| a == "--classic");

    // First non-flag argument after the binary name is the video/library path.
    let arg = args.iter().skip(1).find(|a| !a.starts_with("--")).cloned();

    // Detect if arg is a folder (library) or a file (single layer)
    let (initial_video, library_folder) = match arg {
        Some(ref path) => {
            let p = PathBuf::from(path);
            if p.is_dir() {
                (None, Some(p))
            } else {
                // It's a file — also use its parent directory as the library
                let parent = p.parent().map(|p| p.to_path_buf());
                (Some(path.clone()), parent)
            }
        }
        None => {
            // Default: use ./library/ if it exists
            let default_lib = PathBuf::from("library");
            if default_lib.is_dir() {
                (None, Some(default_lib))
            } else {
                (None, None)
            }
        }
    };

    // Start web control panel server
    let web_state = WebState::new();
    let url = web::server::spawn(web_state.clone(), 3030);
    log::info!("Opening control panel: {}", url);
    let open_url = if classic {
        format!("{url}?view=classic")
    } else {
        url.clone()
    };
    open_control_panel(&open_url);

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(initial_video, library_folder, web_state);
    event_loop.run_app(&mut app).unwrap();
}

/// Open the control panel in a chromeless "app mode" window if a Chromium-based
/// browser (Chrome / Edge / Brave) is installed, falling back to the system
/// default browser otherwise. App mode (`--app=<url>`) gives a standalone window
/// with no tabs or address bar, so the panel reads like a native app.
fn open_control_panel(url: &str) {
    // macOS app bundles must be launched via their inner binary to pass flags
    // reliably (`open -a … --args --app=…` is dropped when an instance is up).
    #[cfg(target_os = "macos")]
    let candidates: &[&str] = &[
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ];
    #[cfg(not(target_os = "macos"))]
    let candidates: &[&str] = &["google-chrome", "chromium", "microsoft-edge", "brave"];

    for bin in candidates {
        // On macOS the candidates are absolute paths; skip ones that aren't present.
        #[cfg(target_os = "macos")]
        if !std::path::Path::new(bin).exists() {
            continue;
        }
        if std::process::Command::new(bin)
            .arg(format!("--app={url}"))
            .spawn()
            .is_ok()
        {
            return;
        }
    }

    // No Chromium browser available (or launch failed): use the default browser.
    let _ = open::that(url);
}

/// Parsed CLI arguments for the headless `render` subcommand.
#[derive(Debug, Clone, PartialEq)]
struct RenderArgs {
    patch_path: String,
    library: String,
    out: String,
    width: u32,
    height: u32,
    fps: u32,
    duration: f32,
}

/// Parse the `render` subcommand's flags into a `RenderArgs` (pure: no I/O).
///
/// `--patch` and `--library` are required; everything else has a default.
/// Kept separate from `run_cli_render` so the CLI contract is unit-testable.
fn parse_render_args(args: &[String]) -> Result<RenderArgs, String> {
    let mut patch_path: Option<String> = None;
    let mut library: Option<String> = None;
    let mut out = "experiments/headless-output/out.mp4".to_string();
    let mut width: u32 = 1280;
    let mut height: u32 = 720;
    let mut fps: u32 = 30;
    let mut duration: f32 = 10.0;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--patch" => patch_path = args.get(i + 1).cloned(),
            "--library" => library = args.get(i + 1).cloned(),
            "--out" => {
                if let Some(v) = args.get(i + 1) {
                    out = v.clone();
                }
            }
            "--duration" => {
                duration = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .ok_or("bad --duration")?;
            }
            "--fps" => {
                fps = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .ok_or("bad --fps")?;
            }
            "--res" => {
                let v = args.get(i + 1).ok_or("missing value for --res")?;
                let (w, h) = v.split_once('x').ok_or("--res must look like 1280x720")?;
                width = w.parse().map_err(|_| "bad --res width")?;
                height = h.parse().map_err(|_| "bad --res height")?;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 2;
    }

    Ok(RenderArgs {
        patch_path: patch_path.ok_or("missing --patch <file.yaml>")?,
        library: library.ok_or("missing --library <folder>")?,
        out,
        width,
        height,
        fps,
        duration,
    })
}

/// Headless `render` subcommand: load a patch YAML and render it to an MP4.
///
/// Usage:
///   collide-o-scope render --patch <file.yaml> --library <folder> \
///       [--out <file.mp4>] [--duration <secs>] [--fps <n>] [--res <WxH>]
fn run_cli_render(args: &[String]) -> Result<(), String> {
    let RenderArgs {
        patch_path,
        library,
        out,
        width,
        height,
        fps,
        duration,
    } = parse_render_args(args)?;

    let yaml =
        std::fs::read_to_string(&patch_path).map_err(|e| format!("reading {patch_path}: {e}"))?;
    let patch: patch::PatchState =
        serde_yaml::from_str(&yaml).map_err(|e| format!("parsing {patch_path}: {e}"))?;

    let config = render_export::ExportConfig {
        width,
        height,
        fps,
        duration_secs: duration,
        output_path: out.clone(),
        // Patches don't carry tempo; use the default so beat-synced formulas
        // render deterministically (phase starts at 0 in offline export).
        bpm: 120.0,
        // CLI renders at full-res VHS (highest quality) by default.
        match_preview: false,
    };

    println!(
        "rendering {} layer(s) -> {out} ({width}x{height} @ {fps}fps, {duration}s)",
        patch.layers.len()
    );
    render_export::render_blocking(patch, config, &library)?;
    println!("done: {out}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    /// output_dims maps common aspect-ratio strings to pixel sizes, falls back
    /// to 16:9 for unknown ratios, and always rounds to even dimensions.
    #[test]
    fn output_dims_common_ratios_and_evenness() {
        eprintln!("cli: output_dims resolves aspect ratios to even encoder-friendly sizes");
        assert_eq!(output_dims("16:9", 1080), (1920, 1080));
        assert_eq!(output_dims("4:3", 1080), (1440, 1080));
        assert_eq!(output_dims("1:1", 1080), (1080, 1080));
        assert_eq!(output_dims("9:16", 1080), (1080, 1920));
        assert_eq!(output_dims("21:9", 1080), (2520, 1080));
        // Unknown ratio falls back to 16:9.
        assert_eq!(output_dims("bogus", 720), output_dims("16:9", 720));
        // Dims are always even (encoder-friendly), even for odd quality input.
        let (w, h) = output_dims("16:9", 721);
        assert_eq!(w % 2, 0);
        assert_eq!(h % 2, 0);
    }

    /// fit_scale computes the right per-axis scale for stretch, contain, and
    /// cover fit modes and short-circuits to identity on non-positive dims.
    #[test]
    fn fit_scale_modes() {
        eprintln!("cli: fit_scale handles stretch/contain/cover and degenerate dims");
        // Stretch (mode 0) is always identity.
        assert_eq!(fit_scale(0.0, 1920.0, 1080.0, 1080.0, 1080.0), (1.0, 1.0));
        // Square source on square canvas: any mode is identity.
        assert_eq!(fit_scale(1.0, 100.0, 100.0, 100.0, 100.0), (1.0, 1.0));
        // Contain (mode 1), wide source into square canvas (r = 16/9 > 1):
        // bars top/bottom → scale up Y by r.
        let (x, y) = fit_scale(1.0, 1920.0, 1080.0, 1080.0, 1080.0);
        assert_eq!(x, 1.0);
        assert!(y > 1.0);
        // Cover (mode 2) of the same → crop sides (x < 1, y = 1).
        let (x, y) = fit_scale(2.0, 1920.0, 1080.0, 1080.0, 1080.0);
        assert!(x < 1.0);
        assert_eq!(y, 1.0);
        // Non-positive dims short-circuit to identity.
        assert_eq!(fit_scale(2.0, 0.0, 100.0, 100.0, 100.0), (1.0, 1.0));
        assert_eq!(fit_scale(1.0, 100.0, 100.0, 0.0, 100.0), (1.0, 1.0));
    }

    /// hex_to_rgb01 / rgb01_to_hex parse valid hex (with optional #), fall back
    /// to green on bad input, and round-trip a color while clamping out-of-range.
    #[test]
    fn hex_color_round_trip_and_fallback() {
        eprintln!("cli: hex<->rgb01 conversion round-trips and falls back on bad hex");
        assert_eq!(hex_to_rgb01("#000000"), (0.0, 0.0, 0.0));
        assert_eq!(hex_to_rgb01("#ffffff"), (1.0, 1.0, 1.0));
        // Leading # is optional.
        assert_eq!(hex_to_rgb01("ff0000"), (1.0, 0.0, 0.0));
        // Bad input → default green.
        assert_eq!(hex_to_rgb01("#fff"), (0.0, 1.0, 0.0)); // too short
        assert_eq!(hex_to_rgb01("nothex!"), (0.0, 1.0, 0.0));
        // rgb01_to_hex round-trips and clamps out-of-range.
        assert_eq!(rgb01_to_hex(1.0, 0.0, 0.0), "#ff0000");
        assert_eq!(rgb01_to_hex(2.0, -1.0, 0.5), "#ff0080");
        let (r, g, b) = hex_to_rgb01(&rgb01_to_hex(0.2, 0.4, 0.6));
        assert!((r - 0.2).abs() < 0.01 && (g - 0.4).abs() < 0.01 && (b - 0.6).abs() < 0.01);
    }

    /// sanitize_patch_name keeps safe characters, replaces path-traversal and
    /// symbol characters with underscores, trims ends, and caps length at 48.
    #[test]
    fn sanitize_patch_name_strips_unsafe_and_caps_length() {
        eprintln!("cli: sanitize_patch_name strips unsafe chars and caps length");
        assert_eq!(sanitize_patch_name("my patch-1_v2"), "my patch-1_v2");
        // Path traversal / symbols (incl. dots) become underscores.
        assert_eq!(sanitize_patch_name("../etc/passwd"), "___etc_passwd");
        assert_eq!(sanitize_patch_name("a/b\\c:d"), "a_b_c_d");
        // Trimmed at both ends.
        assert_eq!(sanitize_patch_name("  spaced  "), "spaced");
        // Capped at 48 chars.
        let long = "x".repeat(100);
        assert_eq!(sanitize_patch_name(&long).len(), 48);
    }

    /// fit_to_area fits an aspect ratio inside a bounding box, respecting
    /// whichever of width or height is the limiting dimension.
    #[test]
    fn fit_to_area_respects_both_bounds() {
        eprintln!("cli: fit_to_area respects whichever bound limits the aspect ratio");
        // Width-limited: 16:9 into a wide-but-short box → height drives it.
        let (w, h) = fit_to_area(1600.0, 100.0, 16.0 / 9.0);
        assert!(h <= 100.0 + 1e-3);
        assert!((w / h - 16.0 / 9.0).abs() < 1e-3);
        // Height-limited: tall box → width drives it.
        let (w, h) = fit_to_area(100.0, 1600.0, 16.0 / 9.0);
        assert!(w <= 100.0 + 1e-3);
        assert!((w / h - 16.0 / 9.0).abs() < 1e-3);
    }

    /// parse_render_args accepts just the required --patch/--library flags and
    /// fills in the documented defaults for out path, resolution, fps, duration.
    #[test]
    fn parse_render_args_defaults_with_required_flags() {
        eprintln!("cli: parse_render_args fills defaults when only required flags are given");
        let a = parse_render_args(&args(&["--patch", "p.yaml", "--library", "lib/"])).unwrap();
        assert_eq!(a.patch_path, "p.yaml");
        assert_eq!(a.library, "lib/");
        assert_eq!(a.out, "experiments/headless-output/out.mp4");
        assert_eq!((a.width, a.height), (1280, 720));
        assert_eq!(a.fps, 30);
        assert_eq!(a.duration, 10.0);
    }

    /// parse_render_args applies every optional flag (--out/--res/--fps/
    /// --duration) when supplied, overriding the defaults.
    #[test]
    fn parse_render_args_overrides_all() {
        eprintln!("cli: parse_render_args honors all optional override flags");
        let a = parse_render_args(&args(&[
            "--patch",
            "p.yaml",
            "--library",
            "lib/",
            "--out",
            "o.mp4",
            "--res",
            "640x480",
            "--fps",
            "24",
            "--duration",
            "2.5",
        ]))
        .unwrap();
        assert_eq!(a.out, "o.mp4");
        assert_eq!((a.width, a.height), (640, 480));
        assert_eq!(a.fps, 24);
        assert_eq!(a.duration, 2.5);
    }

    /// parse_render_args errors when either required flag (--patch or
    /// --library) is missing, including the empty-args case.
    #[test]
    fn parse_render_args_requires_patch_and_library() {
        eprintln!("cli: parse_render_args errors when a required flag is missing");
        assert!(parse_render_args(&args(&["--library", "lib/"])).is_err());
        assert!(parse_render_args(&args(&["--patch", "p.yaml"])).is_err());
        assert!(parse_render_args(&[]).is_err());
    }

    /// parse_render_args rejects malformed --res, non-numeric --fps/--duration,
    /// and any unknown flag.
    #[test]
    fn parse_render_args_rejects_bad_values_and_unknown_flags() {
        eprintln!("cli: parse_render_args rejects bad values and unknown flags");
        let base = ["--patch", "p.yaml", "--library", "lib/"];
        // Malformed --res.
        let mut v = base.to_vec();
        v.extend(["--res", "1280"]);
        assert!(parse_render_args(&args(&v)).is_err());
        let mut v = base.to_vec();
        v.extend(["--res", "axb"]);
        assert!(parse_render_args(&args(&v)).is_err());
        // Non-numeric --fps / --duration.
        let mut v = base.to_vec();
        v.extend(["--fps", "soon"]);
        assert!(parse_render_args(&args(&v)).is_err());
        // Unknown flag.
        let mut v = base.to_vec();
        v.extend(["--wat", "1"]);
        assert!(parse_render_args(&args(&v)).is_err());
    }
}
