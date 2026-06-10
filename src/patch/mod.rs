#[allow(dead_code)]
pub mod editor;

use serde::{Deserialize, Serialize};

use crate::effects::EffectUniforms;
use crate::layers::{BlendMode, Layer};
use crate::ntsc::NtscParams;

// --- Helpers for serde defaults ---

fn one() -> f32 {
    1.0
}
fn default_fps() -> f32 {
    30.0
}

// --- Parameter metadata for stepping & comments ---

pub struct ParamMeta {
    pub step: f32,
    pub min: f32,
    pub max: f32,
    pub desc: &'static str,
}

pub fn param_meta(name: &str) -> Option<ParamMeta> {
    match name {
        "pixelate" => Some(ParamMeta { step: 1.0, min: 1.0, max: 32.0, desc: "pixel block size" }),
        "rgb_split" => Some(ParamMeta { step: 0.5, min: 0.0, max: 30.0, desc: "chromatic split px" }),
        "hue_shift" => Some(ParamMeta { step: 5.0, min: -180.0, max: 180.0, desc: "degrees" }),
        "saturation" => Some(ParamMeta { step: 0.05, min: -1.0, max: 1.0, desc: "color intensity" }),
        "brightness" => Some(ParamMeta { step: 0.05, min: -1.0, max: 1.0, desc: "exposure" }),
        "contrast" => Some(ParamMeta { step: 0.05, min: -1.0, max: 1.0, desc: "dynamic range" }),
        "posterize" => Some(ParamMeta { step: 1.0, min: 0.0, max: 16.0, desc: "color levels (0=off)" }),
        "grain_intensity" => Some(ParamMeta { step: 0.01, min: 0.0, max: 0.3, desc: "film grain amount" }),
        "grain_size" => Some(ParamMeta { step: 0.25, min: 1.0, max: 4.0, desc: "grain particle scale" }),
        "grain_algo" => Some(ParamMeta { step: 1.0, min: 0.0, max: 3.0, desc: "0=value 1=perlin 2=gaussian 3=salt&pepper" }),
        "breathe_scale" => Some(ParamMeta { step: 0.005, min: 0.0, max: 0.05, desc: "zoom oscillation" }),
        "breathe_rotation" => Some(ParamMeta { step: 0.1, min: 0.0, max: 2.0, desc: "rotation oscillation deg" }),
        "breathe_position" => Some(ParamMeta { step: 0.002, min: 0.0, max: 0.02, desc: "position drift" }),
        "vignette" => Some(ParamMeta { step: 0.05, min: 0.0, max: 1.5, desc: "edge darkening" }),
        "color_drift" => Some(ParamMeta { step: 0.002, min: 0.0, max: 0.02, desc: "chromatic aberration" }),
        "opacity" => Some(ParamMeta { step: 0.05, min: 0.0, max: 1.0, desc: "layer transparency" }),
        "speed" => Some(ParamMeta { step: 0.25, min: 0.25, max: 4.0, desc: "playback multiplier" }),
        "fps" => Some(ParamMeta { step: 1.0, min: 1.0, max: 60.0, desc: "decode frame rate" }),
        "wave_amp" => Some(ParamMeta { step: 0.005, min: 0.0, max: 0.1, desc: "wave displacement" }),
        "wave_freq" => Some(ParamMeta { step: 1.0, min: 0.0, max: 50.0, desc: "wave cycles" }),
        "wave_speed" => Some(ParamMeta { step: 0.5, min: 0.0, max: 10.0, desc: "wave scroll speed" }),
        "wave_axis" => Some(ParamMeta { step: 1.0, min: 0.0, max: 2.0, desc: "0=horiz 1=vert 2=both" }),
        "swirl_angle" => Some(ParamMeta { step: 10.0, min: -720.0, max: 720.0, desc: "vortex degrees" }),
        "swirl_radius" => Some(ParamMeta { step: 0.05, min: 0.0, max: 1.0, desc: "vortex extent" }),
        "bulge_strength" => Some(ParamMeta { step: 0.05, min: -1.0, max: 1.0, desc: "+bulge / -pinch" }),
        "bulge_radius" => Some(ParamMeta { step: 0.05, min: 0.05, max: 1.0, desc: "lens extent" }),
        "chroma_threshold" => Some(ParamMeta { step: 0.02, min: 0.0, max: 1.0, desc: "key tolerance" }),
        "chroma_smoothness" => Some(ParamMeta { step: 0.02, min: 0.0, max: 1.0, desc: "key feather" }),
        "chroma_spill" => Some(ParamMeta { step: 0.05, min: 0.0, max: 1.0, desc: "key spill suppress" }),
        "slice_intensity" => Some(ParamMeta { step: 0.02, min: 0.0, max: 1.0, desc: "band shift amount" }),
        "slice_height" => Some(ParamMeta { step: 1.0, min: 1.0, max: 128.0, desc: "band thickness px" }),
        "slice_prob" => Some(ParamMeta { step: 0.05, min: 0.0, max: 1.0, desc: "bands shifted" }),
        "slice_speed" => Some(ParamMeta { step: 1.0, min: 0.0, max: 30.0, desc: "reseed steps/sec" }),
        "block_size" => Some(ParamMeta { step: 4.0, min: 4.0, max: 128.0, desc: "block edge px" }),
        "block_intensity" => Some(ParamMeta { step: 0.02, min: 0.0, max: 1.0, desc: "block offset amount" }),
        "block_prob" => Some(ParamMeta { step: 0.05, min: 0.0, max: 1.0, desc: "blocks displaced" }),
        "block_speed" => Some(ParamMeta { step: 1.0, min: 0.0, max: 30.0, desc: "block reseed rate" }),
        "shift_chroma" => Some(ParamMeta { step: 0.02, min: 0.0, max: 1.0, desc: "glitch chroma fringe" }),
        "slice_axis" => Some(ParamMeta { step: 1.0, min: 0.0, max: 2.0, desc: "0=horiz 1=vert 2=both" }),
        "jitter_amount" => Some(ParamMeta { step: 0.01, min: 0.0, max: 1.0, desc: "continuous wobble" }),
        "jitter_speed" => Some(ParamMeta { step: 1.0, min: 0.0, max: 30.0, desc: "wobble rate" }),
        "datamosh" => Some(ParamMeta { step: 0.02, min: 0.0, max: 1.0, desc: "prev-frame bleed" }),
        _ => None,
    }
}


// --- Serializable patch state ---

#[derive(Serialize, Deserialize, Clone)]
pub struct PatchState {
    pub master: EffectsConfig,
    pub layers: Vec<LayerConfig>,
    #[serde(default)]
    pub ntsc: Option<NtscConfig>,
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

fn default_edge_wave_speed() -> f32 { 0.5 }
fn default_head_height() -> i32 { 8 }
fn default_tracking_height() -> i32 { 24 }

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
    #[serde(default)]
    pub paused: bool,
    #[serde(default = "default_true")]
    pub visible: bool,
    #[serde(default)]
    pub effects: EffectsConfig,
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
}

fn default_wave_freq() -> f32 { 8.0 }
fn default_half() -> f32 { 0.5 }
fn default_chroma_threshold() -> f32 { 0.4 }
fn default_chroma_smoothness() -> f32 { 0.1 }
fn default_slice_height() -> f32 { 16.0 }
fn default_slice_prob() -> f32 { 0.3 }
fn default_slice_speed() -> f32 { 8.0 }
fn default_block_size() -> f32 { 32.0 }
fn default_block_prob() -> f32 { 0.2 }
fn default_block_speed() -> f32 { 6.0 }
fn default_jitter_speed() -> f32 { 8.0 }

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
    }

    /// Get fields organized into groups for display.
    pub fn grouped_fields(&self) -> Vec<(&'static str, Vec<(&'static str, String)>)> {
        vec![
            ("digital", vec![
                ("pixelate", format!("{:.1}", self.pixelate)),
                ("rgb_split", format!("{:.1}", self.rgb_split)),
                ("hue_shift", format!("{:.1}", self.hue_shift)),
                ("saturation", format!("{:.2}", self.saturation)),
                ("brightness", format!("{:.2}", self.brightness)),
                ("contrast", format!("{:.2}", self.contrast)),
                ("posterize", format!("{:.1}", self.posterize)),
                ("invert", format!("{}", self.invert)),
            ]),
            ("analog", vec![
                ("grain_intensity", format!("{:.2}", self.grain_intensity)),
                ("grain_size", format!("{:.2}", self.grain_size)),
                ("grain_algo", format!("{}", self.grain_algo)),
                ("color_grain", format!("{}", self.color_grain)),
                ("vignette", format!("{:.2}", self.vignette)),
                ("color_drift", format!("{:.3}", self.color_drift)),
            ]),
            ("motion", vec![
                ("breathe_scale", format!("{:.3}", self.breathe_scale)),
                ("breathe_rotation", format!("{:.2}", self.breathe_rotation)),
                ("breathe_position", format!("{:.3}", self.breathe_position)),
            ]),
            ("warp", vec![
                ("wave_amp", format!("{:.3}", self.wave_amp)),
                ("wave_freq", format!("{:.1}", self.wave_freq)),
                ("wave_speed", format!("{:.2}", self.wave_speed)),
                ("wave_axis", format!("{:.0}", self.wave_axis)),
                ("swirl_angle", format!("{:.1}", self.swirl_angle)),
                ("swirl_radius", format!("{:.2}", self.swirl_radius)),
                ("bulge_strength", format!("{:.2}", self.bulge_strength)),
                ("bulge_radius", format!("{:.2}", self.bulge_radius)),
            ]),
            ("key", vec![
                ("chroma_enable", format!("{}", self.chroma_enable)),
                ("chroma_threshold", format!("{:.2}", self.chroma_threshold)),
                ("chroma_smoothness", format!("{:.2}", self.chroma_smoothness)),
                ("chroma_spill", format!("{:.2}", self.chroma_spill)),
                ("chroma_r", format!("{:.2}", self.chroma_r)),
                ("chroma_g", format!("{:.2}", self.chroma_g)),
                ("chroma_b", format!("{:.2}", self.chroma_b)),
            ]),
            ("shift", vec![
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
            ]),
        ]
    }

    /// Set a single field by key name. Returns true if the key was recognized.
    pub fn set_field(&mut self, key: &str, value: &str) -> bool {
        match key {
            "pixelate" => { if let Ok(v) = value.parse() { self.pixelate = v; return true; } }
            "rgb_split" => { if let Ok(v) = value.parse() { self.rgb_split = v; return true; } }
            "hue_shift" => { if let Ok(v) = value.parse() { self.hue_shift = v; return true; } }
            "saturation" => { if let Ok(v) = value.parse() { self.saturation = v; return true; } }
            "brightness" => { if let Ok(v) = value.parse() { self.brightness = v; return true; } }
            "contrast" => { if let Ok(v) = value.parse() { self.contrast = v; return true; } }
            "posterize" => { if let Ok(v) = value.parse() { self.posterize = v; return true; } }
            "invert" => { if let Ok(v) = value.parse() { self.invert = v; return true; } }
            "grain_intensity" => { if let Ok(v) = value.parse() { self.grain_intensity = v; return true; } }
            "grain_size" => { if let Ok(v) = value.parse() { self.grain_size = v; return true; } }
            "grain_algo" => { if let Ok(v) = value.parse() { self.grain_algo = v; return true; } }
            "color_grain" => { if let Ok(v) = value.parse() { self.color_grain = v; return true; } }
            "breathe_scale" => { if let Ok(v) = value.parse() { self.breathe_scale = v; return true; } }
            "breathe_rotation" => { if let Ok(v) = value.parse() { self.breathe_rotation = v; return true; } }
            "breathe_position" => { if let Ok(v) = value.parse() { self.breathe_position = v; return true; } }
            "vignette" => { if let Ok(v) = value.parse() { self.vignette = v; return true; } }
            "color_drift" => { if let Ok(v) = value.parse() { self.color_drift = v; return true; } }
            "wave_amp" => { if let Ok(v) = value.parse() { self.wave_amp = v; return true; } }
            "wave_freq" => { if let Ok(v) = value.parse() { self.wave_freq = v; return true; } }
            "wave_speed" => { if let Ok(v) = value.parse() { self.wave_speed = v; return true; } }
            "wave_axis" => { if let Ok(v) = value.parse() { self.wave_axis = v; return true; } }
            "swirl_angle" => { if let Ok(v) = value.parse() { self.swirl_angle = v; return true; } }
            "swirl_radius" => { if let Ok(v) = value.parse() { self.swirl_radius = v; return true; } }
            "bulge_strength" => { if let Ok(v) = value.parse() { self.bulge_strength = v; return true; } }
            "bulge_radius" => { if let Ok(v) = value.parse() { self.bulge_radius = v; return true; } }
            "chroma_enable" => { if let Ok(v) = value.parse() { self.chroma_enable = v; return true; } }
            "chroma_threshold" => { if let Ok(v) = value.parse() { self.chroma_threshold = v; return true; } }
            "chroma_smoothness" => { if let Ok(v) = value.parse() { self.chroma_smoothness = v; return true; } }
            "chroma_spill" => { if let Ok(v) = value.parse() { self.chroma_spill = v; return true; } }
            "chroma_r" => { if let Ok(v) = value.parse() { self.chroma_r = v; return true; } }
            "chroma_g" => { if let Ok(v) = value.parse() { self.chroma_g = v; return true; } }
            "chroma_b" => { if let Ok(v) = value.parse() { self.chroma_b = v; return true; } }
            "slice_intensity" => { if let Ok(v) = value.parse() { self.slice_intensity = v; return true; } }
            "slice_height" => { if let Ok(v) = value.parse() { self.slice_height = v; return true; } }
            "slice_prob" => { if let Ok(v) = value.parse() { self.slice_prob = v; return true; } }
            "slice_speed" => { if let Ok(v) = value.parse() { self.slice_speed = v; return true; } }
            "block_size" => { if let Ok(v) = value.parse() { self.block_size = v; return true; } }
            "block_intensity" => { if let Ok(v) = value.parse() { self.block_intensity = v; return true; } }
            "block_prob" => { if let Ok(v) = value.parse() { self.block_prob = v; return true; } }
            "block_speed" => { if let Ok(v) = value.parse() { self.block_speed = v; return true; } }
            "shift_chroma" => { if let Ok(v) = value.parse() { self.shift_chroma = v; return true; } }
            "slice_axis" => { if let Ok(v) = value.parse() { self.slice_axis = v; return true; } }
            "jitter_amount" => { if let Ok(v) = value.parse() { self.jitter_amount = v; return true; } }
            "jitter_speed" => { if let Ok(v) = value.parse() { self.jitter_speed = v; return true; } }
            "datamosh" => { if let Ok(v) = value.parse() { self.datamosh = v; return true; } }
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
            paused: layer.paused,
            visible: layer.visible,
            effects: EffectsConfig::from_uniforms(&layer.effects),
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
        layer.paused = self.paused;
        layer.visible = self.visible;
        self.effects.apply_to_uniforms(&mut layer.effects);
    }

    /// Get top-level layer fields as (key, value_string) pairs.
    pub fn top_fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("filename", self.filename.clone()),
            ("opacity", format!("{:.2}", self.opacity)),
            ("blend_mode", self.blend_mode.clone()),
            ("speed", format!("{:.2}", self.speed)),
            ("fps", format!("{:.1}", self.fps)),
            ("paused", format!("{}", self.paused)),
            ("visible", format!("{}", self.visible)),
        ]
    }

    /// Set a top-level field by key name. Returns true if recognized.
    pub fn set_field(&mut self, key: &str, value: &str) -> bool {
        match key {
            "opacity" => { if let Ok(v) = value.parse() { self.opacity = v; return true; } }
            "blend_mode" => { self.blend_mode = value.to_string(); return true; }
            "speed" => { if let Ok(v) = value.parse() { self.speed = v; return true; } }
            "fps" => { if let Ok(v) = value.parse() { self.fps = v; return true; } }
            "paused" => { if let Ok(v) = value.parse() { self.paused = v; return true; } }
            "visible" => { if let Ok(v) = value.parse() { self.visible = v; return true; } }
            _ => {}
        }
        false
    }
}

// --- Full patch snapshot ---

impl PatchState {
    pub fn capture(master: &EffectUniforms, layers: &[Layer], ntsc_params: &NtscParams) -> Self {
        Self {
            master: EffectsConfig::from_uniforms(master),
            layers: layers.iter().map(LayerConfig::from_layer).collect(),
            ntsc: Some(NtscConfig::from_params(ntsc_params)),
        }
    }

    pub fn apply(&self, master: &mut EffectUniforms, layers: &mut [Layer], ntsc_params: &mut NtscParams) {
        self.master.apply_to_uniforms(master);
        for (config, layer) in self.layers.iter().zip(layers.iter_mut()) {
            config.apply_to_layer(layer);
        }
        if let Some(ref ntsc) = self.ntsc {
            *ntsc_params = ntsc.to_params();
        }
    }
}
