//! Per-layer audio DSP (Phase 2): fixed 3-band EQ + tap delay.
//!
//! Runs inside the mixer thread, once per *output* (device-rate) frame, after
//! varispeed interpolation and before gain/pan. Because it operates at the
//! device rate, the band frequencies and delay time map directly to the output
//! clock (no per-source rate bookkeeping).
//!
//! Per-channel signal chain:
//! ```text
//!   sample → low-shelf → mid-peak → high-shelf → tap delay → out
//! ```
//! The EQ is three cascaded RBJ-cookbook biquads at fixed corners (120 Hz
//! low-shelf, 1 kHz peak, 6 kHz high-shelf); only the dB gains are user-driven.
//! The delay is a single feedback tap: `out = dry·(1−mix) + delayed·mix`, and we
//! write `dry + delayed·feedback` back into the line.
//!
//! State (filter history + delay ring) is *per channel* and lives here on the
//! `LayerSource`, never in [`AudioParams`] (which is `Copy` plain data shuttled
//! over the command channel). Coefficients are recomputed only when an EQ gain
//! actually changes, so the steady-state cost is three biquads + a ring read.

use super::AudioParams;

/// Maximum delay time in milliseconds; the per-channel ring is sized for this.
const MAX_DELAY_MS: f32 = 1000.0;

/// Fixed EQ band corners / centre (Hz) and the mid band's Q.
const LOW_FREQ: f32 = 120.0;
const MID_FREQ: f32 = 1000.0;
const HIGH_FREQ: f32 = 6000.0;
const MID_Q: f32 = 0.9;
/// Shelf slope S (1.0 = max slope with no overshoot) for the RBJ shelves.
const SHELF_S: f32 = 1.0;

/// Direct Form I biquad: `y = b0·x + b1·x1 + b2·x2 − a1·y1 − a2·y2`
/// (coefficients pre-normalised by a0). History is kept across coefficient
/// swaps so a live gain tweak doesn't click.
#[derive(Clone, Copy)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    /// Pass-through filter (flat, no effect).
    fn identity() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Swap in new coefficients, leaving the running history untouched.
    fn set_coeffs(&mut self, c: [f32; 5]) {
        self.b0 = c[0];
        self.b1 = c[1];
        self.b2 = c[2];
        self.a1 = c[3];
        self.a2 = c[4];
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// One channel's filter + delay state.
struct ChannelDsp {
    low: Biquad,
    mid: Biquad,
    high: Biquad,
    /// Delay ring buffer (device-rate samples, sized for `MAX_DELAY_MS`).
    delay: Vec<f32>,
    delay_pos: usize,
}

impl ChannelDsp {
    fn new(rate: u32) -> Self {
        let max_samples = ((MAX_DELAY_MS / 1000.0) * rate as f32).ceil() as usize + 1;
        Self {
            low: Biquad::identity(),
            mid: Biquad::identity(),
            high: Biquad::identity(),
            delay: vec![0.0; max_samples.max(1)],
            delay_pos: 0,
        }
    }

    fn reset(&mut self) {
        self.low.reset();
        self.mid.reset();
        self.high.reset();
        for s in self.delay.iter_mut() {
            *s = 0.0;
        }
        self.delay_pos = 0;
    }
}

/// All per-layer DSP: one [`ChannelDsp`] per output channel plus cached params
/// so we only recompute biquad coefficients when an EQ gain changes.
pub struct LayerDsp {
    rate: u32,
    channels: Vec<ChannelDsp>,
    // Cached EQ gains (dB). `coeffs_ready=false` forces the first compute.
    cached_eq_low: f32,
    cached_eq_mid: f32,
    cached_eq_high: f32,
    coeffs_ready: bool,
    // Cached delay params, refreshed every update_params (cheap to copy).
    delay_samples: usize,
    delay_feedback: f32,
    delay_mix: f32,
}

impl LayerDsp {
    pub fn new(rate: u32, channels: u16) -> Self {
        let chans = channels.max(1) as usize;
        Self {
            rate,
            channels: (0..chans).map(|_| ChannelDsp::new(rate)).collect(),
            cached_eq_low: 0.0,
            cached_eq_mid: 0.0,
            cached_eq_high: 0.0,
            coeffs_ready: false,
            delay_samples: 0,
            delay_feedback: 0.0,
            delay_mix: 0.0,
        }
    }

    /// Refresh from the layer's params: recompute EQ coefficients only if a band
    /// gain moved, and cache the (clamped) delay settings. Call once per block.
    pub fn update_params(&mut self, p: &AudioParams) {
        if !self.coeffs_ready
            || p.eq_low != self.cached_eq_low
            || p.eq_mid != self.cached_eq_mid
            || p.eq_high != self.cached_eq_high
        {
            let fs = self.rate as f32;
            let low = low_shelf(LOW_FREQ, fs, p.eq_low);
            let mid = peaking(MID_FREQ, fs, MID_Q, p.eq_mid);
            let high = high_shelf(HIGH_FREQ, fs, p.eq_high);
            for c in self.channels.iter_mut() {
                c.low.set_coeffs(low);
                c.mid.set_coeffs(mid);
                c.high.set_coeffs(high);
            }
            self.cached_eq_low = p.eq_low;
            self.cached_eq_mid = p.eq_mid;
            self.cached_eq_high = p.eq_high;
            self.coeffs_ready = true;
        }

        let ms = p.delay_time.clamp(0.0, MAX_DELAY_MS);
        self.delay_samples = ((ms / 1000.0) * self.rate as f32).round() as usize;
        self.delay_feedback = p.delay_feedback.clamp(0.0, 0.95);
        self.delay_mix = p.delay_mix.clamp(0.0, 1.0);
    }

    /// Process one sample for `ch` through EQ then the tap delay.
    #[inline]
    pub fn process(&mut self, ch: usize, x: f32) -> f32 {
        let delay_samples = self.delay_samples;
        let feedback = self.delay_feedback;
        let mix = self.delay_mix;

        let c = &mut self.channels[ch];
        // EQ: cascade the three shelves/peak.
        let mut s = c.low.process(x);
        s = c.mid.process(s);
        s = c.high.process(s);

        // Tap delay. delay_samples == 0 means "no delay configured" → bypass
        // entirely (and leave the ring alone). Whenever a time is set we keep the
        // ring coherent even at mix 0 so the tail doesn't pop when it's raised.
        if delay_samples == 0 {
            return s;
        }
        let len = c.delay.len();
        let d = delay_samples.min(len - 1);
        let read_idx = (c.delay_pos + len - d) % len;
        let delayed = c.delay[read_idx];
        let out = s * (1.0 - mix) + delayed * mix;
        c.delay[c.delay_pos] = s + delayed * feedback;
        c.delay_pos = (c.delay_pos + 1) % len;
        out
    }

    /// Clear all filter history + delay tails (used on clip swap).
    pub fn reset(&mut self) {
        for c in self.channels.iter_mut() {
            c.reset();
        }
    }
}

/// Normalise raw biquad coefficients by a0 → `[b0, b1, b2, a1, a2]`.
fn normalize(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> [f32; 5] {
    [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0]
}

/// RBJ-cookbook low-shelf at `f0` (Hz) for sample rate `fs`, gain `db`.
fn low_shelf(f0: f32, fs: f32, db: f32) -> [f32; 5] {
    let a = 10f32.powf(db / 40.0);
    let w0 = 2.0 * std::f32::consts::PI * f0 / fs;
    let (sw, cw) = w0.sin_cos();
    let alpha = sw / 2.0 * ((a + 1.0 / a) * (1.0 / SHELF_S - 1.0) + 2.0).sqrt();
    let beta = 2.0 * a.sqrt() * alpha;
    let b0 = a * ((a + 1.0) - (a - 1.0) * cw + beta);
    let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cw);
    let b2 = a * ((a + 1.0) - (a - 1.0) * cw - beta);
    let a0 = (a + 1.0) + (a - 1.0) * cw + beta;
    let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cw);
    let a2 = (a + 1.0) + (a - 1.0) * cw - beta;
    normalize(b0, b1, b2, a0, a1, a2)
}

/// RBJ-cookbook high-shelf at `f0` (Hz) for sample rate `fs`, gain `db`.
fn high_shelf(f0: f32, fs: f32, db: f32) -> [f32; 5] {
    let a = 10f32.powf(db / 40.0);
    let w0 = 2.0 * std::f32::consts::PI * f0 / fs;
    let (sw, cw) = w0.sin_cos();
    let alpha = sw / 2.0 * ((a + 1.0 / a) * (1.0 / SHELF_S - 1.0) + 2.0).sqrt();
    let beta = 2.0 * a.sqrt() * alpha;
    let b0 = a * ((a + 1.0) + (a - 1.0) * cw + beta);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cw);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cw - beta);
    let a0 = (a + 1.0) - (a - 1.0) * cw + beta;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cw);
    let a2 = (a + 1.0) - (a - 1.0) * cw - beta;
    normalize(b0, b1, b2, a0, a1, a2)
}

/// RBJ-cookbook peaking EQ at centre `f0` (Hz), sample rate `fs`, quality `q`,
/// gain `db`.
fn peaking(f0: f32, fs: f32, q: f32, db: f32) -> [f32; 5] {
    let a = 10f32.powf(db / 40.0);
    let w0 = 2.0 * std::f32::consts::PI * f0 / fs;
    let (sw, cw) = w0.sin_cos();
    let alpha = sw / (2.0 * q);
    let b0 = 1.0 + alpha * a;
    let b1 = -2.0 * cw;
    let b2 = 1.0 - alpha * a;
    let a0 = 1.0 + alpha / a;
    let a1 = -2.0 * cw;
    let a2 = 1.0 - alpha / a;
    normalize(b0, b1, b2, a0, a1, a2)
}
