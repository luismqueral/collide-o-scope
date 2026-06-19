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

/// One-pole smoothing time-constants (seconds) for click-free delay moves.
/// The tap length glides slowly (tape-style pitch slide when you sweep the
/// time); the wet level ramps quickly (clean fade when enabling/disabling).
const DELAY_GLIDE_SECS: f32 = 0.03;
const WET_GLIDE_SECS: f32 = 0.008;

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
    /// Smoothed read-tap length in samples (fractional); glides toward the
    /// target so sweeping the time slides instead of jumping → no click.
    delay_cur: f32,
    /// Smoothed wet amount (ramps on/off and tracks mix) to avoid level pops.
    wet_cur: f32,
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
            delay_cur: 0.0,
            wet_cur: 0.0,
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
        self.delay_cur = 0.0;
        self.wet_cur = 0.0;
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
    // `delay_target` is fractional samples; per-channel `delay_cur` glides to it.
    delay_target: f32,
    delay_feedback: f32,
    delay_mix: f32,
    // Per-sample one-pole glide coefficients (derived from the rate once).
    delay_glide: f32,
    wet_glide: f32,
}

impl LayerDsp {
    pub fn new(rate: u32, channels: u16) -> Self {
        let chans = channels.max(1) as usize;
        let fs = rate as f32;
        Self {
            rate,
            channels: (0..chans).map(|_| ChannelDsp::new(rate)).collect(),
            cached_eq_low: 0.0,
            cached_eq_mid: 0.0,
            cached_eq_high: 0.0,
            coeffs_ready: false,
            delay_target: 0.0,
            delay_feedback: 0.0,
            delay_mix: 0.0,
            delay_glide: 1.0 - (-1.0 / (DELAY_GLIDE_SECS * fs)).exp(),
            wet_glide: 1.0 - (-1.0 / (WET_GLIDE_SECS * fs)).exp(),
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
        self.delay_target = (ms / 1000.0) * self.rate as f32;
        self.delay_feedback = p.delay_feedback.clamp(0.0, 0.95);
        self.delay_mix = p.delay_mix.clamp(0.0, 1.0);
    }

    /// Process one sample for `ch` through EQ then the tap delay.
    #[inline]
    pub fn process(&mut self, ch: usize, x: f32) -> f32 {
        let target = self.delay_target;
        let feedback = self.delay_feedback;
        let mix = self.delay_mix;
        let delay_glide = self.delay_glide;
        let wet_glide = self.wet_glide;

        let c = &mut self.channels[ch];
        // EQ: cascade the three shelves/peak.
        let mut s = c.low.process(x);
        s = c.mid.process(s);
        s = c.high.process(s);

        // If we were fully dry (delay effectively off), jump the tap straight to
        // the target so a fresh enable fades in *at* the chosen time instead of
        // sweeping up from zero. Then glide the tap (tape-style) toward the
        // target and ramp the wet level — both per-sample so moving the slider
        // slides rather than clicks.
        if c.wet_cur < 1.0e-4 {
            c.delay_cur = target;
        }
        c.delay_cur += (target - c.delay_cur) * delay_glide;
        let wet_target = if target > 0.0 { mix } else { 0.0 };
        c.wet_cur += (wet_target - c.wet_cur) * wet_glide;

        // Settled fully dry with no delay configured → bypass, leave ring idle.
        if c.wet_cur < 1.0e-4 && target <= 0.0 {
            return s;
        }

        // Fractional read tap: linear-interpolate between the two ring samples
        // straddling `delay_cur` samples back from the write head.
        let len = c.delay.len();
        let d = c.delay_cur.clamp(1.0, (len - 1) as f32);
        let di = d.floor();
        let frac = d - di;
        let i0 = (c.delay_pos + len - di as usize) % len; // newer neighbour
        let i1 = (i0 + len - 1) % len; // one sample older
        let delayed = c.delay[i0] + (c.delay[i1] - c.delay[i0]) * frac;

        let out = s * (1.0 - c.wet_cur) + delayed * c.wet_cur;
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    const FS: f32 = 48_000.0;

    fn coeffs_finite(c: &[f32; 5]) -> bool {
        c.iter().all(|v| v.is_finite())
    }

    #[test]
    fn shelves_and_peak_are_identity_at_0db() {
        // At 0 dB gain every RBJ band collapses to H(z)=1: b0≈1 and the
        // numerator equals the denominator (b1≈a1, b2≈a2).
        for c in [
            low_shelf(LOW_FREQ, FS, 0.0),
            high_shelf(HIGH_FREQ, FS, 0.0),
            peaking(MID_FREQ, FS, MID_Q, 0.0),
        ] {
            assert_abs_diff_eq!(c[0], 1.0, epsilon = 1e-5); // b0
            assert_abs_diff_eq!(c[1], c[3], epsilon = 1e-5); // b1 == a1
            assert_abs_diff_eq!(c[2], c[4], epsilon = 1e-5); // b2 == a2
        }
    }

    #[test]
    fn identity_coeffs_pass_signal_through_unchanged() {
        // Feeding a signal through a biquad loaded with 0 dB coeffs returns the
        // input (the transfer function is exactly 1).
        let mut bq = Biquad::identity();
        bq.set_coeffs(low_shelf(LOW_FREQ, FS, 0.0));
        let signal = [0.0f32, 0.5, -0.3, 0.9, -0.8, 0.1, 0.0];
        for &x in &signal {
            assert_abs_diff_eq!(bq.process(x), x, epsilon = 1e-3);
        }
    }

    #[test]
    fn coeffs_stay_finite_across_gain_range() {
        let mut db = -24.0;
        while db <= 12.0 {
            assert!(coeffs_finite(&low_shelf(LOW_FREQ, FS, db)), "low {db}");
            assert!(coeffs_finite(&high_shelf(HIGH_FREQ, FS, db)), "high {db}");
            assert!(
                coeffs_finite(&peaking(MID_FREQ, FS, MID_Q, db)),
                "peak {db}"
            );
            db += 1.5;
        }
    }

    #[test]
    fn biquad_process_updates_history() {
        // b1 = 1, everything else 0 → a pure one-sample delay (y[n] = x[n-1]).
        let mut bq = Biquad::identity();
        bq.set_coeffs([0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(bq.process(1.0), 0.0); // x1 was 0
        assert_eq!(bq.process(0.0), 1.0); // now emits the previous input
        assert_eq!(bq.process(0.0), 0.0);
        // reset clears history.
        bq.process(0.7);
        bq.reset();
        assert_eq!(bq.process(0.0), 0.0);
    }

    #[test]
    fn update_params_clamps_delay_settings() {
        let mut dsp = LayerDsp::new(48_000, 2);
        let mut p = AudioParams::default();
        p.delay_time = 5_000.0; // over MAX_DELAY_MS (1000)
        p.delay_feedback = 2.0; // over 0.95
        p.delay_mix = 3.0; // over 1.0
        dsp.update_params(&p);
        // 1000 ms at 48 kHz = 48000 samples (clamped from the 5 s request).
        assert_abs_diff_eq!(dsp.delay_target, 48_000.0, epsilon = 1.0);
        assert_eq!(dsp.delay_feedback, 0.95);
        assert_eq!(dsp.delay_mix, 1.0);

        // Negative values clamp up to zero.
        p.delay_time = -100.0;
        p.delay_feedback = -1.0;
        p.delay_mix = -1.0;
        dsp.update_params(&p);
        assert_eq!(dsp.delay_target, 0.0);
        assert_eq!(dsp.delay_feedback, 0.0);
        assert_eq!(dsp.delay_mix, 0.0);
    }
}
