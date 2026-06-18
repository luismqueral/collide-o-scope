//! Shared state between the web control panel and the render engine.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Mutex, RwLock};

use crate::effects::EffectUniforms;

/// Shared state accessible by both the web server and the render loop.
pub struct WebState {
    /// Full app snapshot (pushed from render loop each frame)
    pub app: RwLock<AppSnapshot>,
    /// Broadcast channel for pushing state to all WebSocket clients
    pub tx: broadcast::Sender<String>,
    /// Actions queue: browser pushes commands, render loop drains them
    pub actions: Mutex<Vec<WebAction>>,
    /// Thumbnail cache: filename → JPEG bytes (generated on library scan)
    pub thumbnails: std::sync::RwLock<HashMap<String, Vec<u8>>>,
    /// Preview frames: filename → vec of JPEG frames (for hover animation)
    pub preview_frames: std::sync::RwLock<HashMap<String, Vec<Vec<u8>>>>,
}

/// Full app state snapshot sent to the browser each frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSnapshot {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub effects: EffectsSnapshot,
    pub ntsc: NtscSnapshot,
    pub layers: Vec<LayerSnapshot>,
    pub library: Vec<String>,
    #[serde(default)]
    pub patches: Vec<String>,
    pub paused: bool,
    /// Master content framerate (frame-hold / stutter). 30 = smooth (default).
    #[serde(default = "default_framerate")]
    pub framerate: f32,
    /// Master output aspect ratio label (16:9, 4:3, 1:1, 9:16, 21:9).
    #[serde(default = "default_ratio")]
    pub output_ratio: String,
    /// Master output quality (length of the shorter side: 720, 1080, 1440).
    #[serde(default = "default_quality")]
    pub output_quality: u32,
    /// Current output canvas width in pixels (derived from ratio + quality).
    #[serde(default)]
    pub output_width: u32,
    /// Current output canvas height in pixels (derived from ratio + quality).
    #[serde(default)]
    pub output_height: u32,
    /// Export progress: 0.0 = idle, 0.0..1.0 = rendering, 1.0 = done
    #[serde(default)]
    pub export_progress: f32,
    /// Non-empty when export encountered an error
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub export_error: String,
    /// Master param automations: param name → expression text.
    #[serde(default)]
    pub automations: HashMap<String, String>,
    /// Master automation parse errors: param name → error message.
    #[serde(default)]
    pub automation_errors: HashMap<String, String>,
    /// Current tap-tempo tempo (beats per minute).
    #[serde(default)]
    pub bpm: f32,
    /// Beats elapsed since the last tap downbeat (drives the UI beat pulse).
    #[serde(default)]
    pub beat: f32,
    /// Master audio volume in dB (−60..+6, 0 = unity).
    #[serde(default)]
    pub master_volume: f32,
    /// Master limiter (brick-wall clip guard) on/off.
    #[serde(default = "default_true")]
    pub master_limiter: bool,
    /// Live output peak level (0..1) for the master meter.
    #[serde(default)]
    pub meter: f32,
}

fn default_true() -> bool {
    true
}

fn default_framerate() -> f32 {
    30.0
}

fn default_ratio() -> String {
    "16:9".to_string()
}

fn default_quality() -> u32 {
    1080
}

impl Default for AppSnapshot {
    fn default() -> Self {
        Self {
            msg_type: "state".to_string(),
            effects: EffectsSnapshot::default(),
            ntsc: NtscSnapshot::default(),
            layers: Vec::new(),
            library: Vec::new(),
            patches: Vec::new(),
            paused: false,
            framerate: 30.0,
            output_ratio: "16:9".to_string(),
            output_quality: 1080,
            output_width: 0,
            output_height: 0,
            export_progress: 0.0,
            export_error: String::new(),
            automations: HashMap::new(),
            automation_errors: HashMap::new(),
            bpm: 120.0,
            beat: 0.0,
            master_volume: 0.0,
            master_limiter: true,
            meter: 0.0,
        }
    }
}

/// A JSON-friendly snapshot of the current effect parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectsSnapshot {
    pub pixelate: f32,
    pub rgb_split: f32,
    pub hue_shift: f32,
    pub saturation: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub posterize: f32,
    pub invert: bool,
    pub grain_intensity: f32,
    pub grain_size: f32,
    pub grain_algo: u32,
    pub color_grain: bool,
    pub vignette: f32,
    pub color_drift: f32,
    pub breathe_scale: f32,
    pub breathe_rotation: f32,
    pub breathe_position: f32,
}

impl Default for EffectsSnapshot {
    fn default() -> Self {
        Self {
            pixelate: 1.0,
            rgb_split: 0.0,
            hue_shift: 0.0,
            saturation: 0.0,
            brightness: 0.0,
            contrast: 0.0,
            posterize: 0.0,
            invert: false,
            grain_intensity: 0.0,
            grain_size: 1.0,
            grain_algo: 0,
            color_grain: false,
            vignette: 0.0,
            color_drift: 0.0,
            breathe_scale: 0.0,
            breathe_rotation: 0.0,
            breathe_position: 0.0,
        }
    }
}

/// NTSC/VHS effect parameters sent to the browser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NtscSnapshot {
    pub enabled: bool,
    pub tape_speed: u32,
    pub chroma_loss: f32,
    pub edge_wave_enabled: bool,
    pub edge_wave_intensity: f32,
    pub edge_wave_speed: f32,
    pub head_switching_enabled: bool,
    pub head_switching_height: i32,
    pub head_switching_shift: f32,
    pub tracking_noise_enabled: bool,
    pub tracking_noise_height: i32,
    pub tracking_noise_wave: f32,
    pub tracking_noise_snow: f32,
    pub snow_intensity: f32,
    pub composite_noise_intensity: f32,
    pub luma_noise_intensity: f32,
    pub chroma_noise_intensity: f32,
    pub luma_smear: f32,
    pub composite_sharpening: f32,
}

impl Default for NtscSnapshot {
    fn default() -> Self {
        Self {
            enabled: false,
            tape_speed: 0,
            chroma_loss: 0.0,
            edge_wave_enabled: false,
            edge_wave_intensity: 0.0,
            edge_wave_speed: 0.5,
            head_switching_enabled: false,
            head_switching_height: 8,
            head_switching_shift: 0.0,
            tracking_noise_enabled: false,
            tracking_noise_height: 24,
            tracking_noise_wave: 0.0,
            tracking_noise_snow: 0.0,
            snow_intensity: 0.0,
            composite_noise_intensity: 0.0,
            luma_noise_intensity: 0.0,
            chroma_noise_intensity: 0.0,
            luma_smear: 0.0,
            composite_sharpening: 0.0,
        }
    }
}

impl NtscSnapshot {
    pub fn from_params(p: &crate::ntsc::NtscParams) -> Self {
        Self {
            enabled: p.enabled,
            tape_speed: p.tape_speed,
            chroma_loss: p.chroma_loss,
            edge_wave_enabled: p.edge_wave_enabled,
            edge_wave_intensity: p.edge_wave_intensity,
            edge_wave_speed: p.edge_wave_speed,
            head_switching_enabled: p.head_switching_enabled,
            head_switching_height: p.head_switching_height,
            head_switching_shift: p.head_switching_shift,
            tracking_noise_enabled: p.tracking_noise_enabled,
            tracking_noise_height: p.tracking_noise_height,
            tracking_noise_wave: p.tracking_noise_wave,
            tracking_noise_snow: p.tracking_noise_snow,
            snow_intensity: p.snow_intensity,
            composite_noise_intensity: p.composite_noise_intensity,
            luma_noise_intensity: p.luma_noise_intensity,
            chroma_noise_intensity: p.chroma_noise_intensity,
            luma_smear: p.luma_smear,
            composite_sharpening: p.composite_sharpening,
        }
    }
}

/// Per-layer info sent to the browser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSnapshot {
    pub id: u64,
    pub filename: String,
    pub visible: bool,
    pub paused: bool,
    pub opacity: f32,
    pub speed: f32,
    pub fps: f32,
    pub blend_mode: String,
    pub progress: f32,
    /// Audio-only clip (no video): the grid blanks video-only rows for it.
    #[serde(default)]
    pub audio_only: bool,
    /// Per-layer param automations: param name → expression text.
    #[serde(default)]
    pub automations: HashMap<String, String>,
    /// Per-layer automation parse errors: param name → error message.
    #[serde(default)]
    pub automation_errors: HashMap<String, String>,
    // Per-layer audio
    #[serde(default)]
    pub mute: bool,
    #[serde(default)]
    pub volume: f32,
    #[serde(default)]
    pub pan: f32,
    /// Live post-FX peak level (0..1) for the per-layer audio meter.
    #[serde(default)]
    pub meter: f32,
    // Per-layer audio FX (3-band EQ + tap delay)
    #[serde(default)]
    pub eq_low: f32,
    #[serde(default)]
    pub eq_mid: f32,
    #[serde(default)]
    pub eq_high: f32,
    #[serde(default)]
    pub delay_time: f32,
    #[serde(default)]
    pub delay_feedback: f32,
    #[serde(default)]
    pub delay_mix: f32,
    // Per-layer effects (color)
    pub hue_shift: f32,
    pub saturation: f32,
    pub brightness: f32,
    pub contrast: f32,
    // Per-layer effects (digital)
    pub pixelate: f32,
    pub rgb_split: f32,
    pub posterize: f32,
    pub invert: bool,
    // Per-layer effects (warp)
    pub wave_amp: f32,
    pub wave_freq: f32,
    pub wave_speed: f32,
    pub wave_axis: f32,
    pub swirl_angle: f32,
    pub swirl_radius: f32,
    pub bulge_strength: f32,
    pub bulge_radius: f32,
    // Per-layer effects (chroma key)
    pub chroma_enable: bool,
    pub chroma_threshold: f32,
    pub chroma_smoothness: f32,
    pub chroma_spill: f32,
    pub chroma_color: String,
    pub chroma_bg_enable: bool,
    pub chroma_bg_color: String,
    // Per-layer effects (pixel shift / glitch)
    pub slice_intensity: f32,
    pub slice_height: f32,
    pub slice_prob: f32,
    pub slice_speed: f32,
    pub block_size: f32,
    pub block_intensity: f32,
    pub block_prob: f32,
    pub block_speed: f32,
    pub shift_chroma: f32,
    pub slice_axis: f32,
    pub jitter_amount: f32,
    pub jitter_speed: f32,
    pub datamosh: f32,
    // Per-layer effects (feedback / obliteration)
    pub feedback_persistence: f32,
    pub feedback_zoom: f32,
    pub feedback_rotate: f32,
    pub feedback_luma_key: f32,
    pub feedback_chroma: f32,
    pub feedback_additive: f32,
    // Per-layer transform (position / size)
    pub layer_x: f32,
    pub layer_y: f32,
    pub layer_scale: f32,
    /// Fit mode: 0=stretch, 1=fit/contain, 2=fill/cover
    pub fit_mode: u32,
}

/// Actions the browser can request (processed by the render loop).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum WebAction {
    /// Set a master effect parameter
    #[serde(rename = "set_param")]
    SetParam {
        param: String,
        value: serde_json::Value,
    },
    /// Add a layer from the library by filename
    #[serde(rename = "add_layer")]
    AddLayer { filename: String },
    /// Swap a layer's source clip in place (keeps FX, opacity, blend, position)
    #[serde(rename = "set_layer_clip")]
    SetLayerClip { index: usize, filename: String },
    /// Remove a layer by index
    #[serde(rename = "remove_layer")]
    RemoveLayer { index: usize },
    /// Move a layer from one index to another (drag-and-drop reorder)
    #[serde(rename = "move_layer")]
    MoveLayer { from: usize, to: usize },
    /// Toggle layer visibility
    #[serde(rename = "toggle_visibility")]
    ToggleVisibility { index: usize },
    /// Toggle layer play/pause
    #[serde(rename = "toggle_layer_pause")]
    ToggleLayerPause { index: usize },
    /// Toggle master play/pause
    #[serde(rename = "toggle_master_pause")]
    ToggleMasterPause,
    /// Reset all master effects
    #[serde(rename = "reset_fx")]
    ResetFx,
    /// Reset a specific effect group (digital, analog, motion)
    #[serde(rename = "reset_group")]
    ResetGroup { group: String },
    /// Reset a specific per-layer effect group (blend, color, digital, warp, key, shift, transform)
    #[serde(rename = "reset_layer_group")]
    ResetLayerGroup { index: usize, group: String },
    /// Set a per-layer parameter (opacity, speed, blend_mode)
    #[serde(rename = "set_layer_param")]
    SetLayerParam {
        index: usize,
        param: String,
        value: serde_json::Value,
    },
    /// Set the master content framerate (frame-hold / stutter look)
    #[serde(rename = "set_master_framerate")]
    SetMasterFramerate { value: f32 },
    /// Set the master output size / aspect ratio (rebuilds the composite canvas)
    #[serde(rename = "set_output_size")]
    SetOutputSize { ratio: String, quality: u32 },
    /// Set an NTSC/VHS effect parameter
    #[serde(rename = "set_ntsc_param")]
    SetNtscParam {
        param: String,
        value: serde_json::Value,
    },
    /// Set a master audio bus parameter (master_volume, limiter)
    #[serde(rename = "set_master_audio_param")]
    SetMasterAudioParam {
        param: String,
        value: serde_json::Value,
    },
    /// Start an offline render export
    #[serde(rename = "start_export")]
    StartExport {
        width: u32,
        height: u32,
        fps: u32,
        duration_secs: f32,
        #[serde(default)]
        match_preview: bool,
    },
    /// Cancel a running export
    #[serde(rename = "cancel_export")]
    CancelExport,
    /// Automate a master param with an expression
    #[serde(rename = "set_automation")]
    SetAutomation { param: String, expr: String },
    /// Clear automation on a master param
    #[serde(rename = "clear_automation")]
    ClearAutomation { param: String },
    /// Automate a per-layer param with an expression
    #[serde(rename = "set_layer_automation")]
    SetLayerAutomation {
        index: usize,
        param: String,
        expr: String,
    },
    /// Clear automation on a per-layer param
    #[serde(rename = "clear_layer_automation")]
    ClearLayerAutomation { index: usize, param: String },
    /// Register a tempo tap (render loop timestamps it and updates bpm/downbeat)
    #[serde(rename = "tap_tempo")]
    TapTempo,
    /// Set the tempo directly (manual BPM entry)
    #[serde(rename = "set_bpm")]
    SetBpm { value: f32 },
    /// Save the current state as a named patch in the patches folder
    #[serde(rename = "save_patch")]
    SavePatch { name: String },
    /// Load a named patch from the patches folder
    #[serde(rename = "load_patch")]
    LoadPatch { name: String },
    /// Delete a named patch from the patches folder
    #[serde(rename = "delete_patch")]
    DeletePatch { name: String },
    /// Raise the native preview (render output) window and bring it to the front
    #[serde(rename = "focus_window")]
    FocusWindow,
}

impl EffectsSnapshot {
    pub fn from_uniforms(u: &EffectUniforms) -> Self {
        Self {
            pixelate: u.pixelate_size,
            rgb_split: u.rgb_split,
            hue_shift: u.hue_shift,
            saturation: u.saturation,
            brightness: u.brightness,
            contrast: u.contrast,
            posterize: u.posterize,
            invert: u.invert > 0.5,
            grain_intensity: u.grain_intensity,
            grain_size: u.grain_size,
            grain_algo: u.grain_algo as u32,
            color_grain: u.color_grain > 0.5,
            vignette: u.vignette,
            color_drift: u.color_drift,
            breathe_scale: u.breathe_scale,
            breathe_rotation: u.breathe_rotation,
            breathe_position: u.breathe_position,
        }
    }

    pub fn apply_to_uniforms(&self, u: &mut EffectUniforms) {
        u.pixelate_size = self.pixelate.clamp(1.0, 32.0);
        u.rgb_split = self.rgb_split.clamp(0.0, 30.0);
        u.hue_shift = self.hue_shift.clamp(-180.0, 180.0);
        u.saturation = self.saturation.clamp(-1.0, 1.0);
        u.brightness = self.brightness.clamp(-1.0, 1.0);
        u.contrast = self.contrast.clamp(-1.0, 1.0);
        u.posterize = self.posterize.clamp(0.0, 16.0);
        u.invert = if self.invert { 1.0 } else { 0.0 };
        u.grain_intensity = self.grain_intensity.clamp(0.0, 0.3);
        u.grain_size = self.grain_size.clamp(1.0, 4.0);
        u.grain_algo = (self.grain_algo.min(3)) as f32;
        u.color_grain = if self.color_grain { 1.0 } else { 0.0 };
        u.vignette = self.vignette.clamp(0.0, 1.5);
        u.color_drift = self.color_drift.clamp(0.0, 0.02);
        u.breathe_scale = self.breathe_scale.clamp(0.0, 0.05);
        u.breathe_rotation = self.breathe_rotation.clamp(0.0, 2.0);
        u.breathe_position = self.breathe_position.clamp(0.0, 0.02);
    }

    pub fn apply_param(&mut self, param: &str, value: &serde_json::Value) {
        let v = value;
        match param {
            "pixelate" => {
                if let Some(n) = v.as_f64() {
                    self.pixelate = n as f32;
                }
            }
            "rgb_split" => {
                if let Some(n) = v.as_f64() {
                    self.rgb_split = n as f32;
                }
            }
            "hue_shift" => {
                if let Some(n) = v.as_f64() {
                    self.hue_shift = n as f32;
                }
            }
            "saturation" => {
                if let Some(n) = v.as_f64() {
                    self.saturation = n as f32;
                }
            }
            "brightness" => {
                if let Some(n) = v.as_f64() {
                    self.brightness = n as f32;
                }
            }
            "contrast" => {
                if let Some(n) = v.as_f64() {
                    self.contrast = n as f32;
                }
            }
            "posterize" => {
                if let Some(n) = v.as_f64() {
                    self.posterize = n as f32;
                }
            }
            "invert" => {
                if let Some(b) = v.as_bool() {
                    self.invert = b;
                }
            }
            "grain_intensity" => {
                if let Some(n) = v.as_f64() {
                    self.grain_intensity = n as f32;
                }
            }
            "grain_size" => {
                if let Some(n) = v.as_f64() {
                    self.grain_size = n as f32;
                }
            }
            "grain_algo" => {
                if let Some(n) = v.as_u64() {
                    self.grain_algo = n as u32;
                }
            }
            "color_grain" => {
                if let Some(b) = v.as_bool() {
                    self.color_grain = b;
                }
            }
            "vignette" => {
                if let Some(n) = v.as_f64() {
                    self.vignette = n as f32;
                }
            }
            "color_drift" => {
                if let Some(n) = v.as_f64() {
                    self.color_drift = n as f32;
                }
            }
            "breathe_scale" => {
                if let Some(n) = v.as_f64() {
                    self.breathe_scale = n as f32;
                }
            }
            "breathe_rotation" => {
                if let Some(n) = v.as_f64() {
                    self.breathe_rotation = n as f32;
                }
            }
            "breathe_position" => {
                if let Some(n) = v.as_f64() {
                    self.breathe_position = n as f32;
                }
            }
            _ => {}
        }
    }
}

impl WebState {
    pub fn new() -> Arc<Self> {
        let (tx, _) = broadcast::channel(64);
        Arc::new(Self {
            app: RwLock::new(AppSnapshot::default()),
            tx,
            actions: Mutex::new(Vec::new()),
            thumbnails: std::sync::RwLock::new(HashMap::new()),
            preview_frames: std::sync::RwLock::new(HashMap::new()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn effects_snapshot_round_trips_through_uniforms() {
        // A snapshot of in-range values must survive snapshot → uniforms →
        // snapshot unchanged (the browser↔engine effects mirror).
        let mut snap = EffectsSnapshot::default();
        snap.pixelate = 8.0;
        snap.rgb_split = 12.0;
        snap.hue_shift = 90.0;
        snap.saturation = 0.5;
        snap.invert = true;
        snap.grain_intensity = 0.2;
        snap.grain_algo = 2;
        snap.color_grain = true;
        snap.vignette = 1.0;

        let mut u = EffectUniforms::default();
        snap.apply_to_uniforms(&mut u);
        let back = EffectsSnapshot::from_uniforms(&u);

        assert_eq!(back.pixelate, 8.0);
        assert_eq!(back.rgb_split, 12.0);
        assert_eq!(back.hue_shift, 90.0);
        assert_eq!(back.saturation, 0.5);
        assert!(back.invert);
        assert_eq!(back.grain_intensity, 0.2);
        assert_eq!(back.grain_algo, 2);
        assert!(back.color_grain);
        assert_eq!(back.vignette, 1.0);
    }

    #[test]
    fn apply_to_uniforms_clamps_out_of_range() {
        let mut snap = EffectsSnapshot::default();
        snap.pixelate = 999.0; // > 32
        snap.rgb_split = -5.0; // < 0
        snap.hue_shift = 500.0; // > 180
        snap.saturation = -9.0; // < -1
        snap.grain_intensity = 9.0; // > 0.3
        snap.grain_algo = 99; // > 3
        snap.vignette = 9.0; // > 1.5

        let mut u = EffectUniforms::default();
        snap.apply_to_uniforms(&mut u);

        assert_eq!(u.pixelate_size, 32.0);
        assert_eq!(u.rgb_split, 0.0);
        assert_eq!(u.hue_shift, 180.0);
        assert_eq!(u.saturation, -1.0);
        assert_eq!(u.grain_intensity, 0.3);
        assert_eq!(u.grain_algo, 3.0);
        assert_eq!(u.vignette, 1.5);
    }

    #[test]
    fn apply_param_parses_value_types() {
        let mut snap = EffectsSnapshot::default();
        // f64 → f32
        snap.apply_param("rgb_split", &serde_json::json!(15.0));
        assert_eq!(snap.rgb_split, 15.0);
        // bool
        snap.apply_param("invert", &serde_json::json!(true));
        assert!(snap.invert);
        // u64 → u32
        snap.apply_param("grain_algo", &serde_json::json!(2));
        assert_eq!(snap.grain_algo, 2);
        // Unknown param → no-op
        snap.apply_param("not_a_param", &serde_json::json!(5.0));
        // Wrong type for a numeric param → ignored (still default 0.0)
        snap.apply_param("saturation", &serde_json::json!("nope"));
        assert_eq!(snap.saturation, 0.0);
    }

    #[test]
    fn ntsc_snapshot_mirrors_params() {
        let mut p = crate::ntsc::NtscParams::default();
        p.enabled = true;
        p.tape_speed = 2;
        p.chroma_loss = 0.4;
        p.head_switching_height = 12;
        p.snow_intensity = 0.7;
        let snap = NtscSnapshot::from_params(&p);
        assert!(snap.enabled);
        assert_eq!(snap.tape_speed, 2);
        assert_abs_diff_eq!(snap.chroma_loss, 0.4, epsilon = 1e-6);
        assert_eq!(snap.head_switching_height, 12);
        assert_abs_diff_eq!(snap.snow_intensity, 0.7, epsilon = 1e-6);
    }

    /// Serialize → deserialize → re-serialize and compare the JSON values.
    /// `WebAction` has no `PartialEq`, so we compare `serde_json::Value`.
    fn assert_action_round_trips(action: WebAction) {
        let json = serde_json::to_value(&action).expect("serialize");
        let parsed: WebAction = serde_json::from_value(json.clone()).expect("deserialize");
        let reserialized = serde_json::to_value(&parsed).expect("re-serialize");
        assert_eq!(json, reserialized, "round-trip mismatch for {action:?}");
    }

    #[test]
    fn web_action_serde_tag_round_trips_every_variant() {
        let v = serde_json::json!(5.0);
        let cases = vec![
            WebAction::SetParam {
                param: "rgb_split".into(),
                value: v.clone(),
            },
            WebAction::AddLayer {
                filename: "clip.mp4".into(),
            },
            WebAction::SetLayerClip {
                index: 1,
                filename: "b.mp4".into(),
            },
            WebAction::RemoveLayer { index: 0 },
            WebAction::MoveLayer { from: 0, to: 2 },
            WebAction::ToggleVisibility { index: 3 },
            WebAction::ToggleLayerPause { index: 1 },
            WebAction::ToggleMasterPause,
            WebAction::ResetFx,
            WebAction::ResetGroup {
                group: "digital".into(),
            },
            WebAction::ResetLayerGroup {
                index: 0,
                group: "warp".into(),
            },
            WebAction::SetLayerParam {
                index: 0,
                param: "opacity".into(),
                value: v.clone(),
            },
            WebAction::SetMasterFramerate { value: 24.0 },
            WebAction::SetOutputSize {
                ratio: "16:9".into(),
                quality: 1080,
            },
            WebAction::SetNtscParam {
                param: "chroma_loss".into(),
                value: v.clone(),
            },
            WebAction::SetMasterAudioParam {
                param: "master_volume".into(),
                value: v.clone(),
            },
            WebAction::StartExport {
                width: 1280,
                height: 720,
                fps: 30,
                duration_secs: 10.0,
                match_preview: false,
            },
            WebAction::CancelExport,
            WebAction::SetAutomation {
                param: "vignette".into(),
                expr: "sin(t)".into(),
            },
            WebAction::ClearAutomation {
                param: "vignette".into(),
            },
            WebAction::SetLayerAutomation {
                index: 1,
                param: "pixelate".into(),
                expr: "saw(beat)".into(),
            },
            WebAction::ClearLayerAutomation {
                index: 1,
                param: "pixelate".into(),
            },
            WebAction::TapTempo,
            WebAction::SetBpm { value: 128.0 },
            WebAction::SavePatch {
                name: "my-patch".into(),
            },
            WebAction::LoadPatch {
                name: "my-patch".into(),
            },
            WebAction::DeletePatch {
                name: "my-patch".into(),
            },
            WebAction::FocusWindow,
        ];
        for action in cases {
            assert_action_round_trips(action);
        }
    }

    #[test]
    fn web_action_tag_field_names_match_contract() {
        // The "action" tag is the wire contract with the browser; spot-check it.
        let json = serde_json::to_value(WebAction::ToggleMasterPause).unwrap();
        assert_eq!(json["action"], "toggle_master_pause");
        let json = serde_json::to_value(WebAction::SetParam {
            param: "rgb_split".into(),
            value: serde_json::json!(3),
        })
        .unwrap();
        assert_eq!(json["action"], "set_param");
        assert_eq!(json["param"], "rgb_split");
    }

    #[test]
    fn web_action_unknown_tag_fails_to_deserialize() {
        let bad = serde_json::json!({ "action": "not_a_real_action" });
        assert!(serde_json::from_value::<WebAction>(bad).is_err());
    }

    #[test]
    fn app_snapshot_defaults_and_json_round_trips() {
        let snap = AppSnapshot::default();
        assert_eq!(snap.msg_type, "state");
        assert_eq!(snap.framerate, 30.0);
        assert_eq!(snap.output_ratio, "16:9");
        assert_eq!(snap.output_quality, 1080);
        assert!(snap.master_limiter);

        // Full JSON round-trip preserves the key fields.
        let json = serde_json::to_string(&snap).unwrap();
        let back: AppSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back.msg_type, "state");
        assert_eq!(back.framerate, 30.0);
        assert_eq!(back.output_quality, 1080);
        assert!(back.master_limiter);
    }
}
