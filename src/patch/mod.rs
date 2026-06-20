#[allow(dead_code)]
pub mod editor;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::automation::Expr;
use crate::effects::EffectUniforms;
use crate::layers::{BlendMode, Layer};
use crate::ntsc::NtscParams;

// --- Helpers for serde defaults ---
//
// serde fills a missing YAML field with the type's `Default` (0, false, ""),
// but some fields need a non-zero default (opacity should default to 1.0, not
// invisible). `#[serde(default = "one")]` tells serde to call this function when
// the field is absent. They're tiny standalone fns because the attribute needs a
// path to a zero-arg function. This is what lets old patch files keep loading
// after we add new fields — the new field just gets its default.

fn one() -> f32 {
    1.0
}
fn default_fps() -> f32 {
    30.0
}

// --- Parameter metadata for stepping & comments ---

/// Per-parameter UI/validation metadata: the increment a keyboard nudge or
/// slider uses (`step`), the clamp range (`min`/`max`), and a human description
/// (also written as a comment into exported YAML). `&'static str` means the text
/// lives in the binary for the whole program lifetime — no allocation, no owner.
pub struct ParamMeta {
    pub step: f32,
    pub min: f32,
    pub max: f32,
    pub desc: &'static str,
}

pub fn param_meta(name: &str) -> Option<ParamMeta> {
    match name {
        "pixelate" => Some(ParamMeta {
            step: 1.0,
            min: 1.0,
            max: 32.0,
            desc: "pixel block size",
        }),
        "rgb_split" => Some(ParamMeta {
            step: 0.5,
            min: 0.0,
            max: 30.0,
            desc: "chromatic split px",
        }),
        "hue_shift" => Some(ParamMeta {
            step: 5.0,
            min: -180.0,
            max: 180.0,
            desc: "degrees",
        }),
        "saturation" => Some(ParamMeta {
            step: 0.05,
            min: -1.0,
            max: 1.0,
            desc: "color intensity",
        }),
        "brightness" => Some(ParamMeta {
            step: 0.05,
            min: -1.0,
            max: 1.0,
            desc: "exposure",
        }),
        "contrast" => Some(ParamMeta {
            step: 0.05,
            min: -1.0,
            max: 1.0,
            desc: "dynamic range",
        }),
        "posterize" => Some(ParamMeta {
            step: 1.0,
            min: 0.0,
            max: 16.0,
            desc: "color levels (0=off)",
        }),
        "grain_intensity" => Some(ParamMeta {
            step: 0.01,
            min: 0.0,
            max: 0.3,
            desc: "film grain amount",
        }),
        "grain_size" => Some(ParamMeta {
            step: 0.25,
            min: 1.0,
            max: 4.0,
            desc: "grain particle scale",
        }),
        "grain_algo" => Some(ParamMeta {
            step: 1.0,
            min: 0.0,
            max: 3.0,
            desc: "0=value 1=perlin 2=gaussian 3=salt&pepper",
        }),
        "breathe_scale" => Some(ParamMeta {
            step: 0.005,
            min: 0.0,
            max: 0.05,
            desc: "zoom oscillation",
        }),
        "breathe_rotation" => Some(ParamMeta {
            step: 0.1,
            min: 0.0,
            max: 2.0,
            desc: "rotation oscillation deg",
        }),
        "breathe_position" => Some(ParamMeta {
            step: 0.002,
            min: 0.0,
            max: 0.02,
            desc: "position drift",
        }),
        "vignette" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.5,
            desc: "edge darkening",
        }),
        "color_drift" => Some(ParamMeta {
            step: 0.002,
            min: 0.0,
            max: 0.02,
            desc: "chromatic aberration",
        }),
        "opacity" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "layer transparency",
        }),
        "speed" => Some(ParamMeta {
            step: 0.25,
            min: 0.25,
            max: 4.0,
            desc: "playback multiplier",
        }),
        "fps" => Some(ParamMeta {
            step: 1.0,
            min: 1.0,
            max: 60.0,
            desc: "decode frame rate",
        }),
        "loop_start" => Some(ParamMeta {
            step: 0.01,
            min: 0.0,
            max: 1.0,
            desc: "loop in point (fraction)",
        }),
        "loop_end" => Some(ParamMeta {
            step: 0.01,
            min: 0.0,
            max: 1.0,
            desc: "loop out point (fraction)",
        }),
        "volume" => Some(ParamMeta {
            step: 1.0,
            min: -60.0,
            max: 6.0,
            desc: "audio level dB (0=unity)",
        }),
        "pan" => Some(ParamMeta {
            step: 0.05,
            min: -1.0,
            max: 1.0,
            desc: "stereo pan (-1=L 1=R)",
        }),
        "eq_low" => Some(ParamMeta {
            step: 1.0,
            min: -24.0,
            max: 12.0,
            desc: "low shelf dB (120Hz)",
        }),
        "eq_mid" => Some(ParamMeta {
            step: 1.0,
            min: -24.0,
            max: 12.0,
            desc: "mid peak dB (1kHz)",
        }),
        "eq_high" => Some(ParamMeta {
            step: 1.0,
            min: -24.0,
            max: 12.0,
            desc: "high shelf dB (6kHz)",
        }),
        "delay_time" => Some(ParamMeta {
            step: 10.0,
            min: 0.0,
            max: 1000.0,
            desc: "delay time ms (0=off)",
        }),
        "delay_feedback" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 0.95,
            desc: "delay regeneration",
        }),
        "delay_mix" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "delay dry/wet",
        }),
        "wave_amp" => Some(ParamMeta {
            step: 0.005,
            min: 0.0,
            max: 0.1,
            desc: "wave displacement",
        }),
        "wave_freq" => Some(ParamMeta {
            step: 1.0,
            min: 0.0,
            max: 50.0,
            desc: "wave cycles",
        }),
        "wave_speed" => Some(ParamMeta {
            step: 0.5,
            min: 0.0,
            max: 10.0,
            desc: "wave scroll speed",
        }),
        "wave_axis" => Some(ParamMeta {
            step: 1.0,
            min: 0.0,
            max: 2.0,
            desc: "0=horiz 1=vert 2=both",
        }),
        "swirl_angle" => Some(ParamMeta {
            step: 10.0,
            min: -720.0,
            max: 720.0,
            desc: "vortex degrees",
        }),
        "swirl_radius" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "vortex extent",
        }),
        "bulge_strength" => Some(ParamMeta {
            step: 0.05,
            min: -1.0,
            max: 1.0,
            desc: "+bulge / -pinch",
        }),
        "bulge_radius" => Some(ParamMeta {
            step: 0.05,
            min: 0.05,
            max: 1.0,
            desc: "lens extent",
        }),
        "chroma_threshold" => Some(ParamMeta {
            step: 0.02,
            min: 0.0,
            max: 1.0,
            desc: "key tolerance",
        }),
        "chroma_smoothness" => Some(ParamMeta {
            step: 0.02,
            min: 0.0,
            max: 1.0,
            desc: "key feather",
        }),
        "chroma_spill" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "key spill suppress",
        }),
        "slice_intensity" => Some(ParamMeta {
            step: 0.02,
            min: 0.0,
            max: 1.0,
            desc: "band shift amount",
        }),
        "slice_height" => Some(ParamMeta {
            step: 1.0,
            min: 1.0,
            max: 128.0,
            desc: "band thickness px",
        }),
        "slice_prob" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "bands shifted",
        }),
        "slice_speed" => Some(ParamMeta {
            step: 1.0,
            min: 0.0,
            max: 30.0,
            desc: "reseed steps/sec",
        }),
        "block_size" => Some(ParamMeta {
            step: 4.0,
            min: 4.0,
            max: 128.0,
            desc: "block edge px",
        }),
        "block_intensity" => Some(ParamMeta {
            step: 0.02,
            min: 0.0,
            max: 1.0,
            desc: "block offset amount",
        }),
        "block_prob" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "blocks displaced",
        }),
        "block_speed" => Some(ParamMeta {
            step: 1.0,
            min: 0.0,
            max: 30.0,
            desc: "block reseed rate",
        }),
        "shift_chroma" => Some(ParamMeta {
            step: 0.02,
            min: 0.0,
            max: 1.0,
            desc: "glitch chroma fringe",
        }),
        "slice_axis" => Some(ParamMeta {
            step: 1.0,
            min: 0.0,
            max: 2.0,
            desc: "0=horiz 1=vert 2=both",
        }),
        "jitter_amount" => Some(ParamMeta {
            step: 0.01,
            min: 0.0,
            max: 1.0,
            desc: "continuous wobble",
        }),
        "jitter_speed" => Some(ParamMeta {
            step: 1.0,
            min: 0.0,
            max: 30.0,
            desc: "wobble rate",
        }),
        "datamosh" => Some(ParamMeta {
            step: 0.02,
            min: 0.0,
            max: 1.0,
            desc: "prev-frame bleed",
        }),
        "feedback_persistence" => Some(ParamMeta {
            step: 0.01,
            min: 0.0,
            max: 1.0,
            desc: "whole-frame trails (1=freeze)",
        }),
        "feedback_zoom" => Some(ParamMeta {
            step: 0.005,
            min: 0.8,
            max: 1.2,
            desc: "droste zoom (1=off)",
        }),
        "feedback_rotate" => Some(ParamMeta {
            step: 0.5,
            min: -30.0,
            max: 30.0,
            desc: "spiral smear deg",
        }),
        "feedback_luma_key" => Some(ParamMeta {
            step: 0.01,
            min: 0.0,
            max: 1.0,
            desc: "bias trails to bright",
        }),
        "feedback_chroma" => Some(ParamMeta {
            step: 0.01,
            min: 0.0,
            max: 1.0,
            desc: "channel-desync ghosts",
        }),
        "feedback_additive" => Some(ParamMeta {
            step: 0.01,
            min: 0.0,
            max: 1.0,
            desc: "mix->additive bloom",
        }),
        "layer_x" => Some(ParamMeta {
            step: 0.01,
            min: -1.0,
            max: 1.0,
            desc: "horizontal offset",
        }),
        "layer_y" => Some(ParamMeta {
            step: 0.01,
            min: -1.0,
            max: 1.0,
            desc: "vertical offset",
        }),
        "layer_scale" => Some(ParamMeta {
            step: 0.01,
            min: 0.1,
            max: 4.0,
            desc: "zoom (1=unchanged)",
        }),
        "fit_mode" => Some(ParamMeta {
            step: 1.0,
            min: 0.0,
            max: 2.0,
            desc: "0=stretch 1=fit 2=fill",
        }),
        _ => None,
    }
}

// --- Serializable patch state ---

// The on-disk format for a saved performance. `#[derive(Serialize, Deserialize)]`
// auto-generates the YAML↔struct conversion (via serde) — saving is "serialize
// this struct", loading is "deserialize into this struct". These config structs
// are deliberately separate from the live runtime types (Layer, EffectUniforms):
// they hold only the persistable settings, not GPU handles or decoders, so the
// file stays small and stable even as the runtime types change.
#[derive(Serialize, Deserialize, Clone)]
pub struct PatchState {
    pub master: EffectsConfig,
    pub layers: Vec<LayerConfig>,
    #[serde(default)]
    pub ntsc: Option<NtscConfig>,
    /// Master param automations: param name → expression text.
    #[serde(default)]
    pub master_automations: HashMap<String, String>,
    /// Master audio bus (volume/limiter). `Option` so old patches without it
    /// load fine and fall back to engine defaults (unity, limiter on).
    #[serde(default)]
    pub audio: Option<MasterAudioConfig>,
}

/// Serializable master audio bus settings for patch files.
#[derive(Serialize, Deserialize, Clone)]
pub struct MasterAudioConfig {
    /// Master output level in dB (0 = unity).
    #[serde(default)]
    pub volume: f32,
    /// Brick-wall clip guard.
    #[serde(default = "default_true")]
    pub limiter: bool,
}

impl Default for MasterAudioConfig {
    fn default() -> Self {
        Self {
            volume: 0.0,
            limiter: true,
        }
    }
}

/// Serializable NTSC/VHS effect parameters for patch files.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct NtscConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub tape_speed: u32,
    #[serde(default)]
    pub chroma_loss: f32,
    #[serde(default)]
    pub edge_wave_enabled: bool,
    #[serde(default)]
    pub edge_wave_intensity: f32,
    #[serde(default = "default_edge_wave_speed")]
    pub edge_wave_speed: f32,
    #[serde(default)]
    pub head_switching_enabled: bool,
    #[serde(default = "default_head_height")]
    pub head_switching_height: i32,
    #[serde(default)]
    pub head_switching_shift: f32,
    #[serde(default)]
    pub tracking_noise_enabled: bool,
    #[serde(default = "default_tracking_height")]
    pub tracking_noise_height: i32,
    #[serde(default)]
    pub tracking_noise_wave: f32,
    #[serde(default)]
    pub tracking_noise_snow: f32,
    #[serde(default)]
    pub snow_intensity: f32,
    #[serde(default)]
    pub composite_noise_intensity: f32,
    #[serde(default)]
    pub luma_noise_intensity: f32,
    #[serde(default)]
    pub chroma_noise_intensity: f32,
    #[serde(default)]
    pub luma_smear: f32,
    #[serde(default)]
    pub composite_sharpening: f32,
}

fn default_edge_wave_speed() -> f32 {
    0.5
}
fn default_head_height() -> i32 {
    8
}
fn default_tracking_height() -> i32 {
    24
}

impl NtscConfig {
    pub fn from_params(p: &NtscParams) -> Self {
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

    pub fn to_params(&self) -> NtscParams {
        NtscParams {
            enabled: self.enabled,
            tape_speed: self.tape_speed,
            chroma_loss: self.chroma_loss,
            edge_wave_enabled: self.edge_wave_enabled,
            edge_wave_intensity: self.edge_wave_intensity,
            edge_wave_speed: self.edge_wave_speed,
            head_switching_enabled: self.head_switching_enabled,
            head_switching_height: self.head_switching_height,
            head_switching_shift: self.head_switching_shift,
            tracking_noise_enabled: self.tracking_noise_enabled,
            tracking_noise_height: self.tracking_noise_height,
            tracking_noise_wave: self.tracking_noise_wave,
            tracking_noise_snow: self.tracking_noise_snow,
            snow_intensity: self.snow_intensity,
            composite_noise_intensity: self.composite_noise_intensity,
            luma_noise_intensity: self.luma_noise_intensity,
            chroma_noise_intensity: self.chroma_noise_intensity,
            luma_smear: self.luma_smear,
            composite_sharpening: self.composite_sharpening,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LayerConfig {
    pub filename: String,
    #[serde(default = "one")]
    pub opacity: f32,
    #[serde(default = "default_blend")]
    pub blend_mode: String,
    #[serde(default = "one")]
    pub speed: f32,
    #[serde(default = "default_fps")]
    pub fps: f32,
    // Loop window as fractions of the clip (0.0..1.0). Defaulted so patches saved
    // before loop trimming load as a whole-clip loop (0.0..1.0).
    #[serde(default)]
    pub loop_start: f32,
    #[serde(default = "one")]
    pub loop_end: f32,
    #[serde(default)]
    pub paused: bool,
    #[serde(default = "default_true")]
    pub visible: bool,
    #[serde(default)]
    pub effects: EffectsConfig,
    /// Per-layer param automations: param name → expression text.
    #[serde(default)]
    pub automations: HashMap<String, String>,
    // Per-layer audio. `volume` is dB (0 = unity); all default to the silent-of-
    // effect values (not muted, unity, centered) so old patches load unchanged.
    #[serde(default)]
    pub mute: bool,
    #[serde(default)]
    pub volume: f32,
    #[serde(default)]
    pub pan: f32,
    // Per-layer Audio FX (3-band EQ + tap delay); all default to 0 (no effect)
    // so patches saved before Phase 2 load unchanged.
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
}

fn default_blend() -> String {
    "normal".to_string()
}
fn default_true() -> bool {
    true
}
#[derive(Serialize, Deserialize, Clone)]
pub struct EffectsConfig {
    #[serde(default = "one")]
    pub pixelate: f32,
    #[serde(default)]
    pub rgb_split: f32,
    #[serde(default)]
    pub hue_shift: f32,
    #[serde(default)]
    pub saturation: f32,
    #[serde(default)]
    pub brightness: f32,
    #[serde(default)]
    pub contrast: f32,
    #[serde(default)]
    pub posterize: f32,
    #[serde(default)]
    pub invert: bool,
    #[serde(default)]
    pub grain_intensity: f32,
    #[serde(default = "one")]
    pub grain_size: f32,
    #[serde(default)]
    pub grain_algo: u32,
    #[serde(default)]
    pub color_grain: bool,
    #[serde(default)]
    pub breathe_scale: f32,
    #[serde(default)]
    pub breathe_rotation: f32,
    #[serde(default)]
    pub breathe_position: f32,
    #[serde(default)]
    pub vignette: f32,
    #[serde(default)]
    pub color_drift: f32,
    // Warp
    #[serde(default)]
    pub wave_amp: f32,
    #[serde(default = "default_wave_freq")]
    pub wave_freq: f32,
    #[serde(default = "one")]
    pub wave_speed: f32,
    #[serde(default)]
    pub wave_axis: f32,
    #[serde(default)]
    pub swirl_angle: f32,
    #[serde(default = "default_half")]
    pub swirl_radius: f32,
    #[serde(default)]
    pub bulge_strength: f32,
    #[serde(default = "default_half")]
    pub bulge_radius: f32,
    // Chroma key (color stored as sRGB 0..1)
    #[serde(default)]
    pub chroma_enable: bool,
    #[serde(default = "default_chroma_threshold")]
    pub chroma_threshold: f32,
    #[serde(default = "default_chroma_smoothness")]
    pub chroma_smoothness: f32,
    #[serde(default)]
    pub chroma_spill: f32,
    #[serde(default)]
    pub chroma_r: f32,
    #[serde(default = "one")]
    pub chroma_g: f32,
    #[serde(default)]
    pub chroma_b: f32,
    // Chroma key background fill (replace keyed-out regions with a solid color)
    #[serde(default)]
    pub chroma_bg_enable: bool,
    #[serde(default)]
    pub chroma_bg_r: f32,
    #[serde(default)]
    pub chroma_bg_g: f32,
    #[serde(default)]
    pub chroma_bg_b: f32,
    // Pixel shift / glitch
    #[serde(default)]
    pub slice_intensity: f32,
    #[serde(default = "default_slice_height")]
    pub slice_height: f32,
    #[serde(default = "default_slice_prob")]
    pub slice_prob: f32,
    #[serde(default = "default_slice_speed")]
    pub slice_speed: f32,
    #[serde(default = "default_block_size")]
    pub block_size: f32,
    #[serde(default)]
    pub block_intensity: f32,
    #[serde(default = "default_block_prob")]
    pub block_prob: f32,
    #[serde(default = "default_block_speed")]
    pub block_speed: f32,
    #[serde(default)]
    pub shift_chroma: f32,
    #[serde(default)]
    pub slice_axis: f32,
    #[serde(default)]
    pub jitter_amount: f32,
    #[serde(default = "default_jitter_speed")]
    pub jitter_speed: f32,
    #[serde(default)]
    pub datamosh: f32,
    // Feedback / obliteration family
    #[serde(default)]
    pub feedback_persistence: f32,
    // Must default to 1.0: a 0.0 default would divide-collapse the feedback UV.
    #[serde(default = "one")]
    pub feedback_zoom: f32,
    #[serde(default)]
    pub feedback_rotate: f32,
    #[serde(default)]
    pub feedback_luma_key: f32,
    #[serde(default)]
    pub feedback_chroma: f32,
    #[serde(default)]
    pub feedback_additive: f32,
    // Layer transform (position / size)
    #[serde(default)]
    pub layer_x: f32,
    #[serde(default)]
    pub layer_y: f32,
    // Must default to 1.0: a plain default would load old patches as 0.0 and
    // collapse the layer to a point.
    #[serde(default = "one")]
    pub layer_scale: f32,
    // Fit mode: 0=stretch (default, = old behavior), 1=fit/contain, 2=fill/cover
    #[serde(default)]
    pub fit_mode: f32,
}

fn default_wave_freq() -> f32 {
    8.0
}
fn default_half() -> f32 {
    0.5
}
fn default_chroma_threshold() -> f32 {
    0.4
}
fn default_chroma_smoothness() -> f32 {
    0.1
}
fn default_slice_height() -> f32 {
    16.0
}
fn default_slice_prob() -> f32 {
    0.3
}
fn default_slice_speed() -> f32 {
    8.0
}
fn default_block_size() -> f32 {
    32.0
}
fn default_block_prob() -> f32 {
    0.2
}
fn default_block_speed() -> f32 {
    6.0
}
fn default_jitter_speed() -> f32 {
    8.0
}

impl Default for EffectsConfig {
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
            breathe_scale: 0.0,
            breathe_rotation: 0.0,
            breathe_position: 0.0,
            vignette: 0.0,
            color_drift: 0.0,
            wave_amp: 0.0,
            wave_freq: 8.0,
            wave_speed: 1.0,
            wave_axis: 0.0,
            swirl_angle: 0.0,
            swirl_radius: 0.5,
            bulge_strength: 0.0,
            bulge_radius: 0.5,
            chroma_enable: false,
            chroma_threshold: 0.4,
            chroma_smoothness: 0.1,
            chroma_spill: 0.0,
            chroma_r: 0.0,
            chroma_g: 1.0,
            chroma_b: 0.0,
            chroma_bg_enable: false,
            chroma_bg_r: 0.0,
            chroma_bg_g: 0.0,
            chroma_bg_b: 0.0,
            slice_intensity: 0.0,
            slice_height: 16.0,
            slice_prob: 0.3,
            slice_speed: 8.0,
            block_size: 32.0,
            block_intensity: 0.0,
            block_prob: 0.2,
            block_speed: 6.0,
            shift_chroma: 0.0,
            slice_axis: 0.0,
            jitter_amount: 0.0,
            jitter_speed: 8.0,
            datamosh: 0.0,
            feedback_persistence: 0.0,
            feedback_zoom: 1.0,
            feedback_rotate: 0.0,
            feedback_luma_key: 0.0,
            feedback_chroma: 0.0,
            feedback_additive: 0.0,
            layer_x: 0.0,
            layer_y: 0.0,
            layer_scale: 1.0,
            fit_mode: 0.0,
        }
    }
}

// --- Conversion: EffectUniforms <-> EffectsConfig ---

impl EffectsConfig {
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
            breathe_scale: u.breathe_scale,
            breathe_rotation: u.breathe_rotation,
            breathe_position: u.breathe_position,
            vignette: u.vignette,
            color_drift: u.color_drift,
            wave_amp: u.wave_amp,
            wave_freq: u.wave_freq,
            wave_speed: u.wave_speed,
            wave_axis: u.wave_axis,
            swirl_angle: u.swirl_angle,
            swirl_radius: u.swirl_radius,
            bulge_strength: u.bulge_strength,
            bulge_radius: u.bulge_radius,
            chroma_enable: u.chroma_enable > 0.5,
            chroma_threshold: u.chroma_threshold,
            chroma_smoothness: u.chroma_smoothness,
            chroma_spill: u.chroma_spill,
            chroma_r: u.chroma_color_r,
            chroma_g: u.chroma_color_g,
            chroma_b: u.chroma_color_b,
            chroma_bg_enable: u.chroma_bg_enable > 0.5,
            chroma_bg_r: u.chroma_bg_r,
            chroma_bg_g: u.chroma_bg_g,
            chroma_bg_b: u.chroma_bg_b,
            slice_intensity: u.slice_intensity,
            slice_height: u.slice_height,
            slice_prob: u.slice_prob,
            slice_speed: u.slice_speed,
            block_size: u.block_size,
            block_intensity: u.block_intensity,
            block_prob: u.block_prob,
            block_speed: u.block_speed,
            shift_chroma: u.shift_chroma,
            slice_axis: u.slice_axis,
            jitter_amount: u.jitter_amount,
            jitter_speed: u.jitter_speed,
            datamosh: u.datamosh,
            feedback_persistence: u.feedback_persistence,
            feedback_zoom: u.feedback_zoom,
            feedback_rotate: u.feedback_rotate,
            feedback_luma_key: u.feedback_luma_key,
            feedback_chroma: u.feedback_chroma,
            feedback_additive: u.feedback_additive,
            layer_x: u.layer_x,
            layer_y: u.layer_y,
            layer_scale: u.layer_scale,
            fit_mode: u.fit_mode,
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
        u.breathe_scale = self.breathe_scale.clamp(0.0, 0.05);
        u.breathe_rotation = self.breathe_rotation.clamp(0.0, 2.0);
        u.breathe_position = self.breathe_position.clamp(0.0, 0.02);
        u.vignette = self.vignette.clamp(0.0, 1.5);
        u.color_drift = self.color_drift.clamp(0.0, 0.02);
        u.wave_amp = self.wave_amp.clamp(0.0, 0.1);
        u.wave_freq = self.wave_freq.clamp(0.0, 50.0);
        u.wave_speed = self.wave_speed.clamp(0.0, 10.0);
        u.wave_axis = self.wave_axis.clamp(0.0, 2.0);
        u.swirl_angle = self.swirl_angle.clamp(-720.0, 720.0);
        u.swirl_radius = self.swirl_radius.clamp(0.0, 1.0);
        u.bulge_strength = self.bulge_strength.clamp(-1.0, 1.0);
        u.bulge_radius = self.bulge_radius.clamp(0.05, 1.0);
        u.chroma_enable = if self.chroma_enable { 1.0 } else { 0.0 };
        u.chroma_threshold = self.chroma_threshold.clamp(0.0, 1.0);
        u.chroma_smoothness = self.chroma_smoothness.clamp(0.0, 1.0);
        u.chroma_spill = self.chroma_spill.clamp(0.0, 1.0);
        u.chroma_color_r = self.chroma_r.clamp(0.0, 1.0);
        u.chroma_color_g = self.chroma_g.clamp(0.0, 1.0);
        u.chroma_color_b = self.chroma_b.clamp(0.0, 1.0);
        u.chroma_bg_enable = if self.chroma_bg_enable { 1.0 } else { 0.0 };
        u.chroma_bg_r = self.chroma_bg_r.clamp(0.0, 1.0);
        u.chroma_bg_g = self.chroma_bg_g.clamp(0.0, 1.0);
        u.chroma_bg_b = self.chroma_bg_b.clamp(0.0, 1.0);
        u.slice_intensity = self.slice_intensity.clamp(0.0, 1.0);
        u.slice_height = self.slice_height.clamp(1.0, 128.0);
        u.slice_prob = self.slice_prob.clamp(0.0, 1.0);
        u.slice_speed = self.slice_speed.clamp(0.0, 30.0);
        u.block_size = self.block_size.clamp(4.0, 128.0);
        u.block_intensity = self.block_intensity.clamp(0.0, 1.0);
        u.block_prob = self.block_prob.clamp(0.0, 1.0);
        u.block_speed = self.block_speed.clamp(0.0, 30.0);
        u.shift_chroma = self.shift_chroma.clamp(0.0, 1.0);
        u.slice_axis = self.slice_axis.clamp(0.0, 2.0);
        u.jitter_amount = self.jitter_amount.clamp(0.0, 1.0);
        u.jitter_speed = self.jitter_speed.clamp(0.0, 30.0);
        u.datamosh = self.datamosh.clamp(0.0, 1.0);
        u.feedback_persistence = self.feedback_persistence.clamp(0.0, 1.0);
        u.feedback_zoom = self.feedback_zoom.clamp(0.8, 1.2);
        u.feedback_rotate = self.feedback_rotate.clamp(-30.0, 30.0);
        u.feedback_luma_key = self.feedback_luma_key.clamp(0.0, 1.0);
        u.feedback_chroma = self.feedback_chroma.clamp(0.0, 1.0);
        u.feedback_additive = self.feedback_additive.clamp(0.0, 1.0);
        u.layer_x = self.layer_x.clamp(-1.0, 1.0);
        u.layer_y = self.layer_y.clamp(-1.0, 1.0);
        u.layer_scale = self.layer_scale.clamp(0.1, 4.0);
        // fit_scale_x/y are computed per-frame, not persisted; only fit_mode here.
        u.fit_mode = self.fit_mode.clamp(0.0, 2.0).round();
    }

    /// Get fields organized into groups for display.
    pub fn grouped_fields(&self) -> Vec<(&'static str, Vec<(&'static str, String)>)> {
        vec![
            (
                "digital",
                vec![
                    ("pixelate", format!("{:.1}", self.pixelate)),
                    ("rgb_split", format!("{:.1}", self.rgb_split)),
                    ("hue_shift", format!("{:.1}", self.hue_shift)),
                    ("saturation", format!("{:.2}", self.saturation)),
                    ("brightness", format!("{:.2}", self.brightness)),
                    ("contrast", format!("{:.2}", self.contrast)),
                    ("posterize", format!("{:.1}", self.posterize)),
                    ("invert", format!("{}", self.invert)),
                ],
            ),
            (
                "analog",
                vec![
                    ("grain_intensity", format!("{:.2}", self.grain_intensity)),
                    ("grain_size", format!("{:.2}", self.grain_size)),
                    ("grain_algo", format!("{}", self.grain_algo)),
                    ("color_grain", format!("{}", self.color_grain)),
                    ("vignette", format!("{:.2}", self.vignette)),
                    ("color_drift", format!("{:.3}", self.color_drift)),
                ],
            ),
            (
                "motion",
                vec![
                    ("breathe_scale", format!("{:.3}", self.breathe_scale)),
                    ("breathe_rotation", format!("{:.2}", self.breathe_rotation)),
                    ("breathe_position", format!("{:.3}", self.breathe_position)),
                ],
            ),
            (
                "warp",
                vec![
                    ("wave_amp", format!("{:.3}", self.wave_amp)),
                    ("wave_freq", format!("{:.1}", self.wave_freq)),
                    ("wave_speed", format!("{:.2}", self.wave_speed)),
                    ("wave_axis", format!("{:.0}", self.wave_axis)),
                    ("swirl_angle", format!("{:.1}", self.swirl_angle)),
                    ("swirl_radius", format!("{:.2}", self.swirl_radius)),
                    ("bulge_strength", format!("{:.2}", self.bulge_strength)),
                    ("bulge_radius", format!("{:.2}", self.bulge_radius)),
                ],
            ),
            (
                "key",
                vec![
                    ("chroma_enable", format!("{}", self.chroma_enable)),
                    ("chroma_threshold", format!("{:.2}", self.chroma_threshold)),
                    (
                        "chroma_smoothness",
                        format!("{:.2}", self.chroma_smoothness),
                    ),
                    ("chroma_spill", format!("{:.2}", self.chroma_spill)),
                    ("chroma_r", format!("{:.2}", self.chroma_r)),
                    ("chroma_g", format!("{:.2}", self.chroma_g)),
                    ("chroma_b", format!("{:.2}", self.chroma_b)),
                    ("chroma_bg_enable", format!("{}", self.chroma_bg_enable)),
                    ("chroma_bg_r", format!("{:.2}", self.chroma_bg_r)),
                    ("chroma_bg_g", format!("{:.2}", self.chroma_bg_g)),
                    ("chroma_bg_b", format!("{:.2}", self.chroma_bg_b)),
                ],
            ),
            (
                "shift",
                vec![
                    ("slice_intensity", format!("{:.2}", self.slice_intensity)),
                    ("slice_height", format!("{:.1}", self.slice_height)),
                    ("slice_prob", format!("{:.2}", self.slice_prob)),
                    ("slice_speed", format!("{:.1}", self.slice_speed)),
                    ("block_size", format!("{:.1}", self.block_size)),
                    ("block_intensity", format!("{:.2}", self.block_intensity)),
                    ("block_prob", format!("{:.2}", self.block_prob)),
                    ("block_speed", format!("{:.1}", self.block_speed)),
                    ("shift_chroma", format!("{:.2}", self.shift_chroma)),
                    ("slice_axis", format!("{:.1}", self.slice_axis)),
                    ("jitter_amount", format!("{:.2}", self.jitter_amount)),
                    ("jitter_speed", format!("{:.1}", self.jitter_speed)),
                    ("datamosh", format!("{:.2}", self.datamosh)),
                ],
            ),
            (
                "feedback",
                vec![
                    (
                        "feedback_persistence",
                        format!("{:.2}", self.feedback_persistence),
                    ),
                    ("feedback_zoom", format!("{:.3}", self.feedback_zoom)),
                    ("feedback_rotate", format!("{:.1}", self.feedback_rotate)),
                    (
                        "feedback_luma_key",
                        format!("{:.2}", self.feedback_luma_key),
                    ),
                    ("feedback_chroma", format!("{:.2}", self.feedback_chroma)),
                    (
                        "feedback_additive",
                        format!("{:.2}", self.feedback_additive),
                    ),
                ],
            ),
            (
                "transform",
                vec![
                    ("layer_x", format!("{:.2}", self.layer_x)),
                    ("layer_y", format!("{:.2}", self.layer_y)),
                    ("layer_scale", format!("{:.2}", self.layer_scale)),
                    ("fit_mode", format!("{:.0}", self.fit_mode)),
                ],
            ),
        ]
    }

    /// Set a single field by key name. Returns true if the key was recognized.
    pub fn set_field(&mut self, key: &str, value: &str) -> bool {
        match key {
            "pixelate" => {
                if let Ok(v) = value.parse() {
                    self.pixelate = v;
                    return true;
                }
            }
            "rgb_split" => {
                if let Ok(v) = value.parse() {
                    self.rgb_split = v;
                    return true;
                }
            }
            "hue_shift" => {
                if let Ok(v) = value.parse() {
                    self.hue_shift = v;
                    return true;
                }
            }
            "saturation" => {
                if let Ok(v) = value.parse() {
                    self.saturation = v;
                    return true;
                }
            }
            "brightness" => {
                if let Ok(v) = value.parse() {
                    self.brightness = v;
                    return true;
                }
            }
            "contrast" => {
                if let Ok(v) = value.parse() {
                    self.contrast = v;
                    return true;
                }
            }
            "posterize" => {
                if let Ok(v) = value.parse() {
                    self.posterize = v;
                    return true;
                }
            }
            "invert" => {
                if let Ok(v) = value.parse() {
                    self.invert = v;
                    return true;
                }
            }
            "grain_intensity" => {
                if let Ok(v) = value.parse() {
                    self.grain_intensity = v;
                    return true;
                }
            }
            "grain_size" => {
                if let Ok(v) = value.parse() {
                    self.grain_size = v;
                    return true;
                }
            }
            "grain_algo" => {
                if let Ok(v) = value.parse() {
                    self.grain_algo = v;
                    return true;
                }
            }
            "color_grain" => {
                if let Ok(v) = value.parse() {
                    self.color_grain = v;
                    return true;
                }
            }
            "breathe_scale" => {
                if let Ok(v) = value.parse() {
                    self.breathe_scale = v;
                    return true;
                }
            }
            "breathe_rotation" => {
                if let Ok(v) = value.parse() {
                    self.breathe_rotation = v;
                    return true;
                }
            }
            "breathe_position" => {
                if let Ok(v) = value.parse() {
                    self.breathe_position = v;
                    return true;
                }
            }
            "vignette" => {
                if let Ok(v) = value.parse() {
                    self.vignette = v;
                    return true;
                }
            }
            "color_drift" => {
                if let Ok(v) = value.parse() {
                    self.color_drift = v;
                    return true;
                }
            }
            "wave_amp" => {
                if let Ok(v) = value.parse() {
                    self.wave_amp = v;
                    return true;
                }
            }
            "wave_freq" => {
                if let Ok(v) = value.parse() {
                    self.wave_freq = v;
                    return true;
                }
            }
            "wave_speed" => {
                if let Ok(v) = value.parse() {
                    self.wave_speed = v;
                    return true;
                }
            }
            "wave_axis" => {
                if let Ok(v) = value.parse() {
                    self.wave_axis = v;
                    return true;
                }
            }
            "swirl_angle" => {
                if let Ok(v) = value.parse() {
                    self.swirl_angle = v;
                    return true;
                }
            }
            "swirl_radius" => {
                if let Ok(v) = value.parse() {
                    self.swirl_radius = v;
                    return true;
                }
            }
            "bulge_strength" => {
                if let Ok(v) = value.parse() {
                    self.bulge_strength = v;
                    return true;
                }
            }
            "bulge_radius" => {
                if let Ok(v) = value.parse() {
                    self.bulge_radius = v;
                    return true;
                }
            }
            "chroma_enable" => {
                if let Ok(v) = value.parse() {
                    self.chroma_enable = v;
                    return true;
                }
            }
            "chroma_threshold" => {
                if let Ok(v) = value.parse() {
                    self.chroma_threshold = v;
                    return true;
                }
            }
            "chroma_smoothness" => {
                if let Ok(v) = value.parse() {
                    self.chroma_smoothness = v;
                    return true;
                }
            }
            "chroma_spill" => {
                if let Ok(v) = value.parse() {
                    self.chroma_spill = v;
                    return true;
                }
            }
            "chroma_r" => {
                if let Ok(v) = value.parse() {
                    self.chroma_r = v;
                    return true;
                }
            }
            "chroma_g" => {
                if let Ok(v) = value.parse() {
                    self.chroma_g = v;
                    return true;
                }
            }
            "chroma_b" => {
                if let Ok(v) = value.parse() {
                    self.chroma_b = v;
                    return true;
                }
            }
            "chroma_bg_enable" => {
                if let Ok(v) = value.parse() {
                    self.chroma_bg_enable = v;
                    return true;
                }
            }
            "chroma_bg_r" => {
                if let Ok(v) = value.parse() {
                    self.chroma_bg_r = v;
                    return true;
                }
            }
            "chroma_bg_g" => {
                if let Ok(v) = value.parse() {
                    self.chroma_bg_g = v;
                    return true;
                }
            }
            "chroma_bg_b" => {
                if let Ok(v) = value.parse() {
                    self.chroma_bg_b = v;
                    return true;
                }
            }
            "slice_intensity" => {
                if let Ok(v) = value.parse() {
                    self.slice_intensity = v;
                    return true;
                }
            }
            "slice_height" => {
                if let Ok(v) = value.parse() {
                    self.slice_height = v;
                    return true;
                }
            }
            "slice_prob" => {
                if let Ok(v) = value.parse() {
                    self.slice_prob = v;
                    return true;
                }
            }
            "slice_speed" => {
                if let Ok(v) = value.parse() {
                    self.slice_speed = v;
                    return true;
                }
            }
            "block_size" => {
                if let Ok(v) = value.parse() {
                    self.block_size = v;
                    return true;
                }
            }
            "block_intensity" => {
                if let Ok(v) = value.parse() {
                    self.block_intensity = v;
                    return true;
                }
            }
            "block_prob" => {
                if let Ok(v) = value.parse() {
                    self.block_prob = v;
                    return true;
                }
            }
            "block_speed" => {
                if let Ok(v) = value.parse() {
                    self.block_speed = v;
                    return true;
                }
            }
            "shift_chroma" => {
                if let Ok(v) = value.parse() {
                    self.shift_chroma = v;
                    return true;
                }
            }
            "slice_axis" => {
                if let Ok(v) = value.parse() {
                    self.slice_axis = v;
                    return true;
                }
            }
            "jitter_amount" => {
                if let Ok(v) = value.parse() {
                    self.jitter_amount = v;
                    return true;
                }
            }
            "jitter_speed" => {
                if let Ok(v) = value.parse() {
                    self.jitter_speed = v;
                    return true;
                }
            }
            "datamosh" => {
                if let Ok(v) = value.parse() {
                    self.datamosh = v;
                    return true;
                }
            }
            "feedback_persistence" => {
                if let Ok(v) = value.parse() {
                    self.feedback_persistence = v;
                    return true;
                }
            }
            "feedback_zoom" => {
                if let Ok(v) = value.parse() {
                    self.feedback_zoom = v;
                    return true;
                }
            }
            "feedback_rotate" => {
                if let Ok(v) = value.parse() {
                    self.feedback_rotate = v;
                    return true;
                }
            }
            "feedback_luma_key" => {
                if let Ok(v) = value.parse() {
                    self.feedback_luma_key = v;
                    return true;
                }
            }
            "feedback_chroma" => {
                if let Ok(v) = value.parse() {
                    self.feedback_chroma = v;
                    return true;
                }
            }
            "feedback_additive" => {
                if let Ok(v) = value.parse() {
                    self.feedback_additive = v;
                    return true;
                }
            }
            "layer_x" => {
                if let Ok(v) = value.parse() {
                    self.layer_x = v;
                    return true;
                }
            }
            "layer_y" => {
                if let Ok(v) = value.parse() {
                    self.layer_y = v;
                    return true;
                }
            }
            "layer_scale" => {
                if let Ok(v) = value.parse() {
                    self.layer_scale = v;
                    return true;
                }
            }
            "fit_mode" => {
                if let Ok(v) = value.parse() {
                    self.fit_mode = v;
                    return true;
                }
            }
            _ => {}
        }
        false
    }
}

// --- Conversion: Layer <-> LayerConfig ---

impl LayerConfig {
    pub fn from_layer(layer: &Layer) -> Self {
        Self {
            filename: layer.filename.clone(),
            opacity: layer.opacity,
            blend_mode: match layer.blend_mode {
                BlendMode::Normal => "normal",
                BlendMode::Screen => "screen",
                BlendMode::Multiply => "multiply",
                BlendMode::Difference => "difference",
            }
            .to_string(),
            speed: layer.speed,
            fps: layer.fps,
            loop_start: layer.loop_start,
            loop_end: layer.loop_end,
            paused: layer.paused,
            visible: layer.visible,
            effects: EffectsConfig::from_uniforms(&layer.effects),
            automations: layer
                .automations
                .iter()
                .map(|(k, v)| (k.clone(), v.source.clone()))
                .collect(),
            mute: layer.audio.mute,
            volume: layer.audio.volume,
            pan: layer.audio.pan,
            eq_low: layer.audio.eq_low,
            eq_mid: layer.audio.eq_mid,
            eq_high: layer.audio.eq_high,
            delay_time: layer.audio.delay_time,
            delay_feedback: layer.audio.delay_feedback,
            delay_mix: layer.audio.delay_mix,
        }
    }

    pub fn apply_to_layer(&self, layer: &mut Layer) {
        layer.opacity = self.opacity.clamp(0.0, 1.0);
        layer.blend_mode = match self.blend_mode.as_str() {
            "screen" => BlendMode::Screen,
            "multiply" => BlendMode::Multiply,
            "difference" => BlendMode::Difference,
            _ => BlendMode::Normal,
        };
        layer.speed = self.speed.clamp(0.25, 4.0);
        layer.fps = self.fps.clamp(1.0, 60.0);
        // Set the loop window via the helper so the decoder picks it up too
        // (clamping/ordering is enforced inside the decoder's `set_loop`).
        layer.set_loop(self.loop_start, self.loop_end);
        layer.paused = self.paused;
        layer.visible = self.visible;
        self.effects.apply_to_uniforms(&mut layer.effects);
        // Per-layer audio: written onto the Layer mirror here; the caller is
        // responsible for resyncing these to the audio engine (the engine is
        // keyed by layer id and not reachable from this conversion).
        layer.audio.mute = self.mute;
        layer.audio.volume = self.volume.clamp(-60.0, 6.0);
        layer.audio.pan = self.pan.clamp(-1.0, 1.0);
        layer.audio.eq_low = self.eq_low.clamp(-24.0, 12.0);
        layer.audio.eq_mid = self.eq_mid.clamp(-24.0, 12.0);
        layer.audio.eq_high = self.eq_high.clamp(-24.0, 12.0);
        layer.audio.delay_time = self.delay_time.clamp(0.0, 1000.0);
        layer.audio.delay_feedback = self.delay_feedback.clamp(0.0, 0.95);
        layer.audio.delay_mix = self.delay_mix.clamp(0.0, 1.0);
        // Recompile saved automation expressions into the layer.
        layer.automations.clear();
        layer.automation_errors.clear();
        for (param, expr) in &self.automations {
            match Expr::new(expr) {
                Ok(compiled) => {
                    layer.automations.insert(param.clone(), compiled);
                }
                Err(e) => {
                    layer.automation_errors.insert(param.clone(), e);
                }
            }
        }
    }

    /// Get top-level layer fields as (key, value_string) pairs.
    pub fn top_fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("filename", self.filename.clone()),
            ("opacity", format!("{:.2}", self.opacity)),
            ("blend_mode", self.blend_mode.clone()),
            ("speed", format!("{:.2}", self.speed)),
            ("fps", format!("{:.1}", self.fps)),
            ("loop_start", format!("{:.2}", self.loop_start)),
            ("loop_end", format!("{:.2}", self.loop_end)),
            ("paused", format!("{}", self.paused)),
            ("visible", format!("{}", self.visible)),
            ("mute", format!("{}", self.mute)),
            ("volume", format!("{:.1}", self.volume)),
            ("pan", format!("{:.2}", self.pan)),
            ("eq_low", format!("{:.1}", self.eq_low)),
            ("eq_mid", format!("{:.1}", self.eq_mid)),
            ("eq_high", format!("{:.1}", self.eq_high)),
            ("delay_time", format!("{:.0}", self.delay_time)),
            ("delay_feedback", format!("{:.2}", self.delay_feedback)),
            ("delay_mix", format!("{:.2}", self.delay_mix)),
        ]
    }

    /// Set a top-level field by key name. Returns true if recognized.
    pub fn set_field(&mut self, key: &str, value: &str) -> bool {
        match key {
            "opacity" => {
                if let Ok(v) = value.parse() {
                    self.opacity = v;
                    return true;
                }
            }
            "blend_mode" => {
                self.blend_mode = value.to_string();
                return true;
            }
            "speed" => {
                if let Ok(v) = value.parse() {
                    self.speed = v;
                    return true;
                }
            }
            "fps" => {
                if let Ok(v) = value.parse() {
                    self.fps = v;
                    return true;
                }
            }
            "loop_start" => {
                if let Ok(v) = value.parse() {
                    self.loop_start = v;
                    return true;
                }
            }
            "loop_end" => {
                if let Ok(v) = value.parse() {
                    self.loop_end = v;
                    return true;
                }
            }
            "paused" => {
                if let Ok(v) = value.parse() {
                    self.paused = v;
                    return true;
                }
            }
            "visible" => {
                if let Ok(v) = value.parse() {
                    self.visible = v;
                    return true;
                }
            }
            "mute" => {
                if let Ok(v) = value.parse() {
                    self.mute = v;
                    return true;
                }
            }
            "volume" => {
                if let Ok(v) = value.parse() {
                    self.volume = v;
                    return true;
                }
            }
            "pan" => {
                if let Ok(v) = value.parse() {
                    self.pan = v;
                    return true;
                }
            }
            "eq_low" => {
                if let Ok(v) = value.parse() {
                    self.eq_low = v;
                    return true;
                }
            }
            "eq_mid" => {
                if let Ok(v) = value.parse() {
                    self.eq_mid = v;
                    return true;
                }
            }
            "eq_high" => {
                if let Ok(v) = value.parse() {
                    self.eq_high = v;
                    return true;
                }
            }
            "delay_time" => {
                if let Ok(v) = value.parse() {
                    self.delay_time = v;
                    return true;
                }
            }
            "delay_feedback" => {
                if let Ok(v) = value.parse() {
                    self.delay_feedback = v;
                    return true;
                }
            }
            "delay_mix" => {
                if let Ok(v) = value.parse() {
                    self.delay_mix = v;
                    return true;
                }
            }
            _ => {}
        }
        false
    }
}

// --- Full patch snapshot ---

impl PatchState {
    // `capture` and `apply` are inverses: `capture` snapshots live runtime state
    // into the serializable config (for saving), and `apply` writes a loaded
    // config back onto live state. Keeping them as a mirror pair is what makes
    // save→load round-trip faithfully.
    pub fn capture(
        master: &EffectUniforms,
        master_automations: &HashMap<String, Expr>,
        layers: &[Layer],
        ntsc_params: &NtscParams,
        master_volume: f32,
        master_limiter: bool,
    ) -> Self {
        Self {
            master: EffectsConfig::from_uniforms(master),
            layers: layers.iter().map(LayerConfig::from_layer).collect(),
            ntsc: Some(NtscConfig::from_params(ntsc_params)),
            master_automations: master_automations
                .iter()
                .map(|(k, v)| (k.clone(), v.source.clone()))
                .collect(),
            audio: Some(MasterAudioConfig {
                volume: master_volume,
                limiter: master_limiter,
            }),
        }
    }

    /// The saved master audio bus settings, or engine defaults (unity gain,
    /// limiter on) for patches that predate the audio field. Callers apply
    /// these to both their `App` mirror and the live audio engine.
    pub fn master_audio(&self) -> (f32, bool) {
        let a = self.audio.clone().unwrap_or_default();
        (a.volume, a.limiter)
    }

    pub fn apply(
        &self,
        master: &mut EffectUniforms,
        layers: &mut [Layer],
        ntsc_params: &mut NtscParams,
    ) {
        self.master.apply_to_uniforms(master);
        // `zip` pairs each saved config with the matching live layer and stops at
        // the shorter of the two — so extra live layers or extra saved configs are
        // simply left untouched rather than erroring.
        for (config, layer) in self.layers.iter().zip(layers.iter_mut()) {
            config.apply_to_layer(layer);
        }
        if let Some(ref ntsc) = self.ntsc {
            *ntsc_params = ntsc.to_params();
        }
    }

    /// Recompile the saved master automation expressions, returning the
    /// compiled map and any parse errors. The caller installs these into its
    /// own `App` automation maps (master automations are not stored on the
    /// `EffectUniforms` struct).
    pub fn compile_master_automations(&self) -> (HashMap<String, Expr>, HashMap<String, String>) {
        let mut compiled = HashMap::new();
        let mut errors = HashMap::new();
        for (param, expr) in &self.master_automations {
            match Expr::new(expr) {
                Ok(c) => {
                    compiled.insert(param.clone(), c);
                }
                Err(e) => {
                    errors.insert(param.clone(), e);
                }
            }
        }
        (compiled, errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Parse a small representative patch from YAML (exercises the on-disk format).
    fn sample_patch() -> PatchState {
        let yaml = r#"
master:
  rgb_split: 12.0
  hue_shift: 90.0
  invert: true
layers:
  - filename: a.mp4
    opacity: 0.5
    blend_mode: screen
    volume: -6.0
  - filename: b.mp4
master_automations:
  vignette: "0.5+0.5*sin(t)"
audio:
  volume: -3.0
  limiter: false
"#;
        serde_yaml::from_str(yaml).expect("parse sample patch")
    }

    /// A representative patch survives a YAML serialize → deserialize round-trip
    /// with all key fields preserved.
    #[test]
    fn patch_state_yaml_round_trips() {
        eprintln!("patch: PatchState round-trips through YAML");
        let patch = sample_patch();
        let yaml = serde_yaml::to_string(&patch).expect("serialize");
        let back: PatchState = serde_yaml::from_str(&yaml).expect("deserialize");

        assert_eq!(back.layers.len(), 2);
        assert_eq!(back.layers[0].filename, "a.mp4");
        assert_abs_diff_eq!(back.layers[0].opacity, 0.5, epsilon = 1e-6);
        assert_eq!(back.layers[0].blend_mode, "screen");
        assert_abs_diff_eq!(back.layers[0].volume, -6.0, epsilon = 1e-6);
        assert_abs_diff_eq!(back.master.rgb_split, 12.0, epsilon = 1e-6);
        assert_abs_diff_eq!(back.master.hue_shift, 90.0, epsilon = 1e-6);
        assert!(back.master.invert);
        assert_eq!(
            back.master_automations.get("vignette").map(String::as_str),
            Some("0.5+0.5*sin(t)")
        );
        let (vol, lim) = back.master_audio();
        assert_abs_diff_eq!(vol, -3.0, epsilon = 1e-6);
        assert!(!lim);
    }

    /// A minimal patch with only required fields loads, filling every absent
    /// field from its serde default.
    #[test]
    fn minimal_yaml_relies_on_serde_defaults() {
        eprintln!("patch: minimal YAML falls back to serde defaults");
        // An old/minimal patch: only the required fields. Everything else must
        // fall back to its serde default (guards patch-format drift).
        let yaml = "master: {}\nlayers:\n  - filename: only.mp4\n";
        let patch: PatchState = serde_yaml::from_str(yaml).expect("minimal parse");
        let l = &patch.layers[0];
        assert_eq!(l.filename, "only.mp4");
        assert_eq!(l.opacity, 1.0); // default = "one", not 0.0
        assert_eq!(l.blend_mode, "normal");
        assert_eq!(l.speed, 1.0);
        assert_eq!(l.fps, 30.0);
        // Loop fields absent in old patches → whole-clip window (0.0..1.0).
        assert_eq!(l.loop_start, 0.0);
        assert_eq!(l.loop_end, 1.0);
        assert!(l.visible);
        assert_eq!(l.effects.pixelate, 1.0);
        assert_eq!(l.effects.feedback_zoom, 1.0);
        assert_eq!(l.effects.layer_scale, 1.0);
        // Missing optional sections → None, and master_audio falls back to defaults.
        assert!(patch.ntsc.is_none());
        assert!(patch.audio.is_none());
        let (vol, lim) = patch.master_audio();
        assert_eq!(vol, 0.0);
        assert!(lim);
    }

    /// A `LayerConfig` carrying an explicit loop window survives a YAML
    /// round-trip with both fractions preserved.
    #[test]
    fn layer_config_loop_window_yaml_round_trips() {
        eprintln!("patch: LayerConfig loop_start/loop_end round-trip through YAML");
        let yaml = "master: {}\nlayers:\n  - filename: clip.mp4\n    loop_start: 0.25\n    loop_end: 0.75\n";
        let patch: PatchState = serde_yaml::from_str(yaml).expect("parse");
        let l = &patch.layers[0];
        assert_abs_diff_eq!(l.loop_start, 0.25, epsilon = 1e-6);
        assert_abs_diff_eq!(l.loop_end, 0.75, epsilon = 1e-6);

        // Re-serialize and read back: the window must be unchanged.
        let out = serde_yaml::to_string(&patch).expect("serialize");
        let back: PatchState = serde_yaml::from_str(&out).expect("deserialize");
        assert_abs_diff_eq!(back.layers[0].loop_start, 0.25, epsilon = 1e-6);
        assert_abs_diff_eq!(back.layers[0].loop_end, 0.75, epsilon = 1e-6);
    }

    /// Out-of-range `EffectsConfig` values are clamped (and `fit_mode` rounded)
    /// when applied to the GPU uniforms.
    #[test]
    fn effects_config_apply_to_uniforms_clamps() {
        eprintln!("patch: EffectsConfig apply_to_uniforms clamps out-of-range values");
        let mut cfg = EffectsConfig::default();
        cfg.pixelate = 999.0;
        cfg.rgb_split = -5.0;
        cfg.hue_shift = 500.0;
        cfg.bulge_radius = 0.0; // min is 0.05
        cfg.feedback_zoom = 5.0; // 0.8..1.2
        cfg.layer_scale = 99.0; // 0.1..4.0
        cfg.fit_mode = 1.7; // rounds to 2

        let mut u = EffectUniforms::default();
        cfg.apply_to_uniforms(&mut u);

        assert_eq!(u.pixelate_size, 32.0);
        assert_eq!(u.rgb_split, 0.0);
        assert_eq!(u.hue_shift, 180.0);
        assert_eq!(u.bulge_radius, 0.05);
        assert_eq!(u.feedback_zoom, 1.2);
        assert_eq!(u.layer_scale, 4.0);
        assert_eq!(u.fit_mode, 2.0);
    }

    /// In-range uniform values round-trip through uniforms → config → uniforms
    /// unchanged.
    #[test]
    fn effects_config_from_apply_symmetry() {
        eprintln!("patch: EffectsConfig from/apply uniforms is symmetric");
        // Round-trip in-range uniform values: uniforms → config → uniforms.
        let mut u = EffectUniforms::default();
        u.rgb_split = 10.0;
        u.hue_shift = 45.0;
        u.swirl_angle = 180.0;
        u.bulge_strength = -0.5;
        u.chroma_enable = 1.0;
        u.feedback_zoom = 1.1;
        u.layer_scale = 2.0;

        let cfg = EffectsConfig::from_uniforms(&u);
        let mut back = EffectUniforms::default();
        cfg.apply_to_uniforms(&mut back);

        assert_eq!(back.rgb_split, 10.0);
        assert_eq!(back.hue_shift, 45.0);
        assert_eq!(back.swirl_angle, 180.0);
        assert_eq!(back.bulge_strength, -0.5);
        assert_eq!(back.chroma_enable, 1.0);
        assert_eq!(back.feedback_zoom, 1.1);
        assert_eq!(back.layer_scale, 2.0);
    }

    /// Known parameter names resolve to metadata with a sane range, and unknown
    /// names return none.
    #[test]
    fn param_meta_lookup_coverage() {
        eprintln!("patch: param_meta resolves known params and rejects unknown");
        // A representative set of known params must resolve with a sane range.
        for name in [
            "pixelate",
            "rgb_split",
            "hue_shift",
            "opacity",
            "speed",
            "volume",
            "eq_low",
            "delay_time",
            "wave_amp",
            "swirl_angle",
            "bulge_radius",
            "feedback_zoom",
            "layer_scale",
            "fit_mode",
        ] {
            let m = param_meta(name).unwrap_or_else(|| panic!("missing meta for {name}"));
            assert!(m.min <= m.max, "{name}: min>max");
            assert!(m.step > 0.0, "{name}: step must be positive");
        }
        // Unknown params have no metadata.
        assert!(param_meta("not_a_param").is_none());
    }

    /// `EffectsConfig::set_field` parses valid values, and rejects unparseable
    /// values and unknown keys without mutating state.
    #[test]
    fn effects_config_set_field_parses_and_rejects() {
        eprintln!("patch: EffectsConfig set_field parses valid input and rejects bad keys/values");
        let mut cfg = EffectsConfig::default();
        // Valid numeric.
        assert!(cfg.set_field("rgb_split", "7.5"));
        assert_eq!(cfg.rgb_split, 7.5);
        // Valid bool.
        assert!(cfg.set_field("invert", "true"));
        assert!(cfg.invert);
        // Valid u32.
        assert!(cfg.set_field("grain_algo", "2"));
        assert_eq!(cfg.grain_algo, 2);
        // Unparseable value for a known key → returns false, leaves value intact.
        assert!(!cfg.set_field("rgb_split", "not-a-number"));
        assert_eq!(cfg.rgb_split, 7.5);
        // Unknown key → false.
        assert!(!cfg.set_field("nonexistent", "1.0"));
    }

    /// `LayerConfig::set_field` accepts blend_mode strings verbatim, parses
    /// numeric/bool fields, and rejects bad values and unknown keys.
    #[test]
    fn layer_config_set_field_handles_blend_mode_strings() {
        eprintln!("patch: LayerConfig set_field handles blend_mode and typed fields");
        // blend_mode accepts any string verbatim (always recognized).
        let mut cfg = sample_patch().layers.remove(1); // b.mp4 (defaults)
        assert!(cfg.set_field("blend_mode", "multiply"));
        assert_eq!(cfg.blend_mode, "multiply");
        assert!(cfg.set_field("opacity", "0.25"));
        assert_abs_diff_eq!(cfg.opacity, 0.25, epsilon = 1e-6);
        assert!(cfg.set_field("mute", "true"));
        assert!(cfg.mute);
        // Bad numeric value → false.
        assert!(!cfg.set_field("speed", "fast"));
        // Unknown key → false.
        assert!(!cfg.set_field("nope", "x"));
    }

    /// `NtscConfig` round-trips faithfully through from_params → to_params.
    #[test]
    fn ntsc_config_from_to_params_round_trips() {
        eprintln!("patch: NtscConfig from_params/to_params round-trips");
        let mut p = NtscParams::default();
        p.enabled = true;
        p.tape_speed = 2;
        p.chroma_loss = 0.3;
        p.head_switching_height = 16;
        p.snow_intensity = 0.5;

        let cfg = NtscConfig::from_params(&p);
        let back = cfg.to_params();

        assert!(back.enabled);
        assert_eq!(back.tape_speed, 2);
        assert_abs_diff_eq!(back.chroma_loss, 0.3, epsilon = 1e-6);
        assert_eq!(back.head_switching_height, 16);
        assert_abs_diff_eq!(back.snow_intensity, 0.5, epsilon = 1e-6);
    }

    /// Compiling master automations keeps valid expressions and routes invalid
    /// ones into the errors map.
    #[test]
    fn compile_master_automations_maps_valid_and_errors() {
        eprintln!("patch: compile_master_automations separates valid exprs from errors");
        let mut patch = sample_patch();
        patch.master_automations.clear();
        patch
            .master_automations
            .insert("vignette".into(), "0.5*sin(t)".into());
        patch
            .master_automations
            .insert("rgb_split".into(), "((((".into()); // bad

        let (compiled, errors) = patch.compile_master_automations();
        assert!(compiled.contains_key("vignette"));
        assert!(!compiled.contains_key("rgb_split"));
        assert!(errors.contains_key("rgb_split"));
        assert!(!errors.contains_key("vignette"));
    }
}
