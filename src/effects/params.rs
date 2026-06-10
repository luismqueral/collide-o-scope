/// GPU-side effect parameters, uploaded as a uniform buffer each frame.
/// Must be 16-byte aligned (96 bytes total = 6 × vec4).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EffectUniforms {
    // vec4 #1
    pub pixelate_size: f32,  // 1.0 = off, 2..32 = block size in pixels
    pub rgb_split: f32,      // 0.0 = off, 1..30 = horizontal pixel offset
    pub resolution: [f32; 2],
    // vec4 #2
    pub hue_shift: f32,      // -180..180 degrees
    pub saturation: f32,     // -1..1 (0 = no change)
    pub brightness: f32,     // -1..1 (0 = no change)
    pub contrast: f32,       // -1..1 (0 = no change)
    // vec4 #3
    pub posterize: f32,      // 0 = off, 2..16 = color levels
    pub invert: f32,         // 0.0 = off, 1.0 = full invert
    pub downsample: f32,     // 1.0 = full res, 0.05..1.0 = fraction (lower = blurrier)
    pub time: f32,           // elapsed seconds (for animated noise)
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
    // vec4 #6 — Analog: color drift
    pub color_drift: f32,     // 0.0 = off, 0.0..0.02 (per-frame random aberration)
    pub _pad: [f32; 3],
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
            _pad: [0.0; 3],
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
            _ => {}
        }
    }
}
