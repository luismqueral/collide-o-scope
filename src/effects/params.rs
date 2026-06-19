/// GPU-side effect parameters, uploaded as a uniform buffer each frame.
/// Must be 16-byte aligned (272 bytes total = 17 × vec4).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EffectUniforms {
    // vec4 #1
    pub pixelate_size: f32, // 1.0 = off, 2..32 = block size in pixels
    pub rgb_split: f32,     // 0.0 = off, 1..30 = horizontal pixel offset
    pub resolution: [f32; 2],
    // vec4 #2
    pub hue_shift: f32,  // -180..180 degrees
    pub saturation: f32, // -1..1 (0 = no change)
    pub brightness: f32, // -1..1 (0 = no change)
    pub contrast: f32,   // -1..1 (0 = no change)
    // vec4 #3
    pub posterize: f32,  // 0 = off, 2..16 = color levels
    pub invert: f32,     // 0.0 = off, 1.0 = full invert
    pub downsample: f32, // 1.0 = full res, 0.05..1.0 = fraction (lower = blurrier)
    pub time: f32,       // elapsed seconds (for animated noise)
    // vec4 #4 — Analog: grain
    pub grain_intensity: f32, // 0.0 = off, 0.01..0.3
    pub grain_size: f32,      // 1.0 = fine, 2..4 = coarse
    pub grain_algo: f32,      // 0=gaussian, 1=perlin, 2=salt_pepper, 3=blue
    pub color_grain: f32,     // 0=mono, 1=chromatic
    // vec4 #5 — Analog: breathing + vignette
    pub breathe_scale: f32,    // 0.0 = off, 0.0..0.05 (±zoom)
    pub breathe_rotation: f32, // 0.0 = off, 0.0..2.0 (degrees)
    pub breathe_position: f32, // 0.0 = off, 0.0..0.02 (drift)
    pub vignette: f32,         // 0.0 = off, 0.0..1.5
    // vec4 #6 — Analog: color drift + Warp: wave
    pub color_drift: f32, // 0.0 = off, 0.0..0.02 (per-frame random aberration)
    pub wave_amp: f32,    // 0.0 = off, 0.0..0.10 (UV displacement amplitude)
    pub wave_freq: f32,   // wave cycles across the frame
    pub wave_speed: f32,  // scroll speed (multiplies time)
    // vec4 #7 — Warp: wave axis + swirl + bulge strength
    pub wave_axis: f32,      // 0 = horizontal, 1 = vertical, 2 = both
    pub swirl_angle: f32,    // -720..720 degrees at center (0 = off)
    pub swirl_radius: f32,   // 0..1 radius of influence (UV)
    pub bulge_strength: f32, // -1..1 (+ bulge / - pinch, 0 = off)
    // vec4 #8 — Warp: bulge radius + Chroma: enable/threshold/smoothness
    pub bulge_radius: f32,      // 0..1 extent of the lens
    pub chroma_enable: f32,     // 0.0 = off, 1.0 = key on
    pub chroma_threshold: f32,  // 0..1 how close to key color counts as keyed
    pub chroma_smoothness: f32, // 0..1 soft edge / feather past threshold
    // vec4 #9 — Chroma: spill + key color (sRGB 0..1)
    pub chroma_spill: f32,   // 0..1 suppress residual key tint
    pub chroma_color_r: f32, // key color red (sRGB 0..1)
    pub chroma_color_g: f32, // key color green (sRGB 0..1)
    pub chroma_color_b: f32, // key color blue (sRGB 0..1)
    // vec4 #10 — Shift: slice (scanline-band displacement glitch)
    pub slice_intensity: f32, // 0..1 max horizontal shift (fraction of width)
    pub slice_height: f32,    // 1..128 band thickness in pixels
    pub slice_prob: f32,      // 0..1 fraction of bands that shift each step
    pub slice_speed: f32,     // 0..30 reseed rate (steps/sec)
    // vec4 #11 — Shift: block (rectangular block displacement)
    pub block_size: f32,      // 4..128 block edge in pixels
    pub block_intensity: f32, // 0..1 max offset (fraction of frame)
    pub block_prob: f32,      // 0..1 fraction of blocks displaced
    pub block_speed: f32,     // 0..30 reseed rate (steps/sec)
    // vec4 #12 — Shift: chroma fringing + slice axis + continuous jitter
    pub shift_chroma: f32,  // 0..1 R/B channel offset on displaced regions
    pub slice_axis: f32,    // 0 = horizontal bands (shift X), 1 = vertical (shift Y), 2 = both
    pub jitter_amount: f32, // 0..1 continuous per-line wobble amplitude
    pub jitter_speed: f32,  // 0..30 wobble evolution rate
    // vec4 #13 — Shift: datamosh + Layer transform: position/size
    pub datamosh: f32, // 0..1 how much displaced blocks sample the previous frame
    pub layer_x: f32,  // -1..1 horizontal offset (+ = right), 0 = centered
    pub layer_y: f32,  // -1..1 vertical offset (+ = up), 0 = centered
    pub layer_scale: f32, // 0.1..4 zoom (1.0 = unchanged)
    // vec4 #14 — Fit mode (computed scale on CPU) + pad
    pub fit_mode: f32,    // 0=stretch (default), 1=fit/contain, 2=fill/cover
    pub fit_scale_x: f32, // computed per-frame from fit_mode + source/canvas aspects
    pub fit_scale_y: f32,
    pub _pad_fit: f32,
    // vec4 #15 — Feedback: persistence + transform + luma key
    pub feedback_persistence: f32, // 0..1 ungated whole-frame trails (1.0 = freeze/bloom-out)
    pub feedback_zoom: f32,        // 0.8..1.2 droste infinite-zoom (1.0 = off)
    pub feedback_rotate: f32,      // -30..30 degrees spiral smear (0 = off)
    pub feedback_luma_key: f32,    // 0..1 bias bleed toward bright regions
    // vec4 #16 — Feedback: channel desync + additive blend + pad
    pub feedback_chroma: f32, // 0..1 R/G/B fed back at offset UVs (color ghosts)
    pub feedback_additive: f32, // 0..1 crossfade mix -> additive accumulation
    pub _pad_fb0: f32,
    pub _pad_fb1: f32,
    // vec4 #17 — Chroma key background fill (replace keyed-out regions w/ solid color)
    pub chroma_bg_enable: f32, // 0.0 = transparent key (default), 1.0 = fill bg color
    pub chroma_bg_r: f32,      // bg color red (sRGB 0..1)
    pub chroma_bg_g: f32,      // bg color green (sRGB 0..1)
    pub chroma_bg_b: f32,      // bg color blue (sRGB 0..1)
}

impl Default for EffectUniforms {
    fn default() -> Self {
        Self {
            pixelate_size: 1.0,
            rgb_split: 0.0,
            resolution: [1280.0, 720.0],
            hue_shift: 0.0,
            saturation: 0.0,
            brightness: 0.0,
            contrast: 0.0,
            posterize: 0.0,
            invert: 0.0,
            downsample: 1.0,
            time: 0.0,
            grain_intensity: 0.0,
            grain_size: 1.0,
            grain_algo: 0.0,
            color_grain: 0.0,
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
            chroma_enable: 0.0,
            chroma_threshold: 0.4,
            chroma_smoothness: 0.1,
            chroma_spill: 0.0,
            chroma_color_r: 0.0,
            chroma_color_g: 1.0,
            chroma_color_b: 0.0,
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
            layer_x: 0.0,
            layer_y: 0.0,
            layer_scale: 1.0,
            fit_mode: 0.0,
            fit_scale_x: 1.0,
            fit_scale_y: 1.0,
            _pad_fit: 0.0,
            feedback_persistence: 0.0,
            feedback_zoom: 1.0,
            feedback_rotate: 0.0,
            feedback_luma_key: 0.0,
            feedback_chroma: 0.0,
            feedback_additive: 0.0,
            _pad_fb0: 0.0,
            _pad_fb1: 0.0,
            chroma_bg_enable: 0.0,
            chroma_bg_r: 0.0,
            chroma_bg_g: 0.0,
            chroma_bg_b: 0.0,
        }
    }
}

impl EffectUniforms {
    pub fn increase_pixelate(&mut self) {
        self.pixelate_size = (self.pixelate_size * 2.0).min(32.0).max(2.0);
    }

    pub fn decrease_pixelate(&mut self) {
        self.pixelate_size = (self.pixelate_size / 2.0).max(1.0);
    }

    pub fn increase_rgb_split(&mut self) {
        self.rgb_split = (self.rgb_split + 5.0).min(30.0);
    }

    pub fn decrease_rgb_split(&mut self) {
        self.rgb_split = (self.rgb_split - 5.0).max(0.0);
    }

    pub fn reset(&mut self) {
        let res = self.resolution;
        *self = Self::default();
        self.resolution = res;
    }

    /// Set a numeric field by its web param name (the same names used by
    /// `EffectsSnapshot::apply_param`), clamping to the field's documented
    /// range. Used to write automation-driven values each frame. Non-numeric
    /// params (invert, color_grain, grain_algo) are intentionally ignored —
    /// only continuous params are automatable.
    pub fn set_by_name(&mut self, param: &str, v: f32) {
        match param {
            "pixelate" => self.pixelate_size = v.clamp(1.0, 32.0),
            "rgb_split" => self.rgb_split = v.clamp(0.0, 30.0),
            "hue_shift" => self.hue_shift = v.clamp(-180.0, 180.0),
            "saturation" => self.saturation = v.clamp(-1.0, 1.0),
            "brightness" => self.brightness = v.clamp(-1.0, 1.0),
            "contrast" => self.contrast = v.clamp(-1.0, 1.0),
            "posterize" => self.posterize = v.clamp(0.0, 16.0),
            "grain_intensity" => self.grain_intensity = v.clamp(0.0, 0.3),
            "grain_size" => self.grain_size = v.clamp(1.0, 4.0),
            "vignette" => self.vignette = v.clamp(0.0, 1.5),
            "color_drift" => self.color_drift = v.clamp(0.0, 0.02),
            "breathe_scale" => self.breathe_scale = v.clamp(0.0, 0.05),
            "breathe_rotation" => self.breathe_rotation = v.clamp(0.0, 2.0),
            "breathe_position" => self.breathe_position = v.clamp(0.0, 0.02),
            // Warp (clamps mirror the SetLayerParam handler in main.rs)
            "wave_amp" => self.wave_amp = v.clamp(0.0, 0.1),
            "wave_freq" => self.wave_freq = v.clamp(0.0, 50.0),
            "wave_speed" => self.wave_speed = v.clamp(0.0, 10.0),
            "swirl_angle" => self.swirl_angle = v.clamp(-720.0, 720.0),
            "swirl_radius" => self.swirl_radius = v.clamp(0.0, 1.0),
            "bulge_strength" => self.bulge_strength = v.clamp(-1.0, 1.0),
            "bulge_radius" => self.bulge_radius = v.clamp(0.05, 1.0),
            // Chroma key
            "chroma_threshold" => self.chroma_threshold = v.clamp(0.0, 1.0),
            "chroma_smoothness" => self.chroma_smoothness = v.clamp(0.0, 1.0),
            "chroma_spill" => self.chroma_spill = v.clamp(0.0, 1.0),
            // Shift / glitch
            "slice_intensity" => self.slice_intensity = v.clamp(0.0, 1.0),
            "slice_height" => self.slice_height = v.clamp(1.0, 128.0),
            "slice_prob" => self.slice_prob = v.clamp(0.0, 1.0),
            "slice_speed" => self.slice_speed = v.clamp(0.0, 30.0),
            "block_size" => self.block_size = v.clamp(4.0, 128.0),
            "block_intensity" => self.block_intensity = v.clamp(0.0, 1.0),
            "block_prob" => self.block_prob = v.clamp(0.0, 1.0),
            "block_speed" => self.block_speed = v.clamp(0.0, 30.0),
            "shift_chroma" => self.shift_chroma = v.clamp(0.0, 1.0),
            "jitter_amount" => self.jitter_amount = v.clamp(0.0, 1.0),
            "jitter_speed" => self.jitter_speed = v.clamp(0.0, 30.0),
            "datamosh" => self.datamosh = v.clamp(0.0, 1.0),
            // Feedback
            "feedback_persistence" => self.feedback_persistence = v.clamp(0.0, 1.0),
            "feedback_zoom" => self.feedback_zoom = v.clamp(0.8, 1.2),
            "feedback_rotate" => self.feedback_rotate = v.clamp(-30.0, 30.0),
            "feedback_luma_key" => self.feedback_luma_key = v.clamp(0.0, 1.0),
            "feedback_chroma" => self.feedback_chroma = v.clamp(0.0, 1.0),
            "feedback_additive" => self.feedback_additive = v.clamp(0.0, 1.0),
            // Transform
            "layer_x" => self.layer_x = v.clamp(-1.0, 1.0),
            "layer_y" => self.layer_y = v.clamp(-1.0, 1.0),
            "layer_scale" => self.layer_scale = v.clamp(0.1, 4.0),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniforms_are_exactly_272_bytes() {
        // The Rust struct and the WGSL `Uniforms` must stay byte-identical
        // (17 × vec4 = 272). If this fails, the shader binding is broken.
        assert_eq!(std::mem::size_of::<EffectUniforms>(), 272);
    }

    #[test]
    fn set_by_name_sets_and_clamps_known_params() {
        let mut u = EffectUniforms::default();
        // In-range value is stored verbatim.
        u.set_by_name("rgb_split", 12.0);
        assert_eq!(u.rgb_split, 12.0);
        // Over the max clamps down.
        u.set_by_name("rgb_split", 999.0);
        assert_eq!(u.rgb_split, 30.0);
        // Under the min clamps up.
        u.set_by_name("rgb_split", -5.0);
        assert_eq!(u.rgb_split, 0.0);
        // Signed param clamps both ends.
        u.set_by_name("hue_shift", 500.0);
        assert_eq!(u.hue_shift, 180.0);
        u.set_by_name("hue_shift", -500.0);
        assert_eq!(u.hue_shift, -180.0);
        // A non-zero-based min (bulge_radius 0.05..1.0).
        u.set_by_name("bulge_radius", 0.0);
        assert_eq!(u.bulge_radius, 0.05);
    }

    #[test]
    fn set_by_name_ignores_unknown_param() {
        let mut u = EffectUniforms::default();
        let before = u; // EffectUniforms is Copy
                        // Unknown key (and non-automatable params like "invert") must no-op.
        u.set_by_name("not_a_real_param", 5.0);
        u.set_by_name("invert", 1.0);
        assert_eq!(
            bytemuck::bytes_of(&u),
            bytemuck::bytes_of(&before),
            "unknown param must not mutate state"
        );
    }

    #[test]
    fn reset_restores_defaults_but_keeps_resolution() {
        let mut u = EffectUniforms::default();
        u.resolution = [1920.0, 1080.0];
        u.rgb_split = 20.0;
        u.hue_shift = 90.0;
        u.reset();
        assert_eq!(
            u.resolution,
            [1920.0, 1080.0],
            "reset must preserve resolution"
        );
        assert_eq!(u.rgb_split, 0.0);
        assert_eq!(u.hue_shift, 0.0);
    }

    #[test]
    fn pixelate_increase_decrease_double_halve_and_clamp() {
        let mut u = EffectUniforms::default(); // pixelate_size = 1.0
                                               // First increase jumps from 1.0 to max(2.0, 2.0) = 2.0.
        u.increase_pixelate();
        assert_eq!(u.pixelate_size, 2.0);
        u.increase_pixelate();
        assert_eq!(u.pixelate_size, 4.0);
        // Saturates at 32.
        for _ in 0..10 {
            u.increase_pixelate();
        }
        assert_eq!(u.pixelate_size, 32.0);
        // Decrease halves down to a floor of 1.0.
        for _ in 0..10 {
            u.decrease_pixelate();
        }
        assert_eq!(u.pixelate_size, 1.0);
    }
}
