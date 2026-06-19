//! Parameter automation: drive a numeric param with a small math expression.
//!
//! A param can hold an `Expr` (a compiled fasteval expression) instead of a
//! fixed value. The expression is evaluated every frame against `t` (elapsed
//! seconds) — the SAME `t` used by both the live render loop and the offline
//! exporter — so live preview matches exported video exactly.
//!
//! Everything here is a pure function of the inputs (`t` plus any literal
//! arguments). `wiggle`/`noise` use hash-based value noise, NOT real randomness,
//! so a given `t` always yields the same result.

use fasteval::Compiler; // brings the `.compile()` method into scope
use fasteval::Evaler; // brings the `.eval()` method into scope

/// A compiled automation expression. Parse once, evaluate cheaply each frame.
pub struct Expr {
    /// The original text the user typed (kept so we can echo it back to the UI).
    pub source: String,
    // fasteval keeps the parsed AST and the compiled program in these two slabs.
    // The compiled `Instruction` only stores indices into the slab, so it is safe
    // to keep them together as owned fields.
    slab: fasteval::Slab,
    compiled: fasteval::Instruction,
}

impl Expr {
    /// Parse + compile an expression. Returns an error message on a parse error
    /// so the caller can surface it without panicking.
    pub fn new(source: &str) -> Result<Self, String> {
        let parser = fasteval::Parser::new();
        let mut slab = fasteval::Slab::new();
        let compiled = parser
            .parse(source, &mut slab.ps)
            .map_err(|e| e.to_string())?
            .from(&slab.ps)
            .compile(&slab.ps, &mut slab.cs);
        Ok(Self {
            source: source.to_string(),
            slab,
            compiled,
        })
    }

    /// Evaluate the expression at time `t` (seconds). `beat` is the number of
    /// musical beats elapsed since the last tap downbeat and `bpm` is the current
    /// tempo — both let formulas sync to music (e.g. `sin(beat*tau)` pulses once
    /// per beat). Never panics: any eval error (e.g. unknown function) collapses
    /// to 0.0.
    pub fn eval(&self, t: f32, beat: f32, bpm: f32) -> f32 {
        let t = t as f64;
        let beat = beat as f64;
        let bpm = bpm as f64;
        // The namespace closure resolves `t`, constants, and all helper functions.
        // fasteval passes unknown function names here too, so our custom
        // oscillators/shaping helpers are handled in this single match.
        let mut ns = |name: &str, args: Vec<f64>| -> Option<f64> {
            let arg = |i: usize| args.get(i).copied().unwrap_or(0.0);
            match name {
                // Variables / constants
                "t" => Some(t),
                "pi" => Some(std::f64::consts::PI),
                "tau" => Some(std::f64::consts::TAU),
                // Musical time: `beat` advances 1.0 per beat at the tapped tempo,
                // `bpm` is the current tempo. Pair with the period-1 oscillators
                // (e.g. `saw(beat)` ramps once per beat, `square(beat/4)` flips
                // once per 4-beat bar).
                "beat" => Some(beat),
                "bpm" => Some(bpm),

                // Oscillators (period = 1 second, output -1..1)
                "tri" => Some(tri(arg(0))),
                "saw" => Some(saw(arg(0))),
                "square" => Some(square(arg(0))),
                "pulse" => Some(pulse(arg(0))),

                // Procedural / random shapes (deterministic value-noise)
                "fbm" => Some(fbm(arg(0))),
                "hold" => Some(hold(arg(0))),

                // Shaping helpers
                "clamp" => Some(arg(0).clamp(arg(1), arg(2))),
                "lerp" => Some(arg(0) + (arg(1) - arg(0)) * arg(2)),
                "smoothstep" => Some(smoothstep(arg(0), arg(1), arg(2))),

                // Motion helpers (deterministic value-noise, seeded by t)
                "wiggle" => Some(wiggle(arg(0), arg(1), t)),
                "noise" => Some(noise(arg(0))),

                // Everything else falls through to fasteval's built-ins
                // (sin, cos, tan, abs, min, max, floor, ceil, round, sqrt,
                //  pow via `^`, exp, log, etc.).
                _ => None,
            }
        };
        match self.compiled.eval(&self.slab, &mut ns) {
            Ok(v) if v.is_finite() => v as f32,
            _ => 0.0,
        }
    }
}

// --- Oscillators (1 Hz: one full cycle per second) ---

/// Triangle wave, output -1..1.
fn tri(t: f64) -> f64 {
    let p = (t - t.floor()).fract(); // 0..1 phase
                                     // 0 -> -1, 0.5 -> 1, 1 -> -1
    1.0 - 4.0 * (p - 0.5).abs()
}

/// Sawtooth ramp, output -1..1.
fn saw(t: f64) -> f64 {
    let p = t - t.floor(); // 0..1 phase
    2.0 * p - 1.0
}

/// Square wave, output -1..1.
fn square(t: f64) -> f64 {
    if (t - t.floor()) < 0.5 {
        1.0
    } else {
        -1.0
    }
}

/// Sharp recurring throb: mostly low with a fast Gaussian spike once per period,
/// output -1..1. `w` sets the spike width.
fn pulse(t: f64) -> f64 {
    let p = t - t.floor(); // 0..1 phase
    let d = p - 0.5;
    let w = 0.12;
    2.0 * (-(d * d) / (w * w)).exp() - 1.0
}

/// Fractal value-noise (4 octaves): organic, layered wander. Output -1..1
/// (each octave is value_noise, already -1..1, normalized by total amplitude).
fn fbm(x: f64) -> f64 {
    let (mut v, mut amp, mut f, mut tot) = (0.0, 0.5, 1.0, 0.0);
    for o in 0..4 {
        v += value_noise(x * f + (o as f64) * 13.0) * amp;
        tot += amp;
        amp *= 0.5;
        f *= 2.0;
    }
    v / tot
}

/// Sample-and-hold: a hard random level held for each integer step of the input
/// (re-rolls on every whole number). Output -1..1.
fn hold(x: f64) -> f64 {
    hash01(x.floor())
}

/// Smooth Hermite interpolation between `lo` and `hi`, like GLSL smoothstep.
fn smoothstep(lo: f64, hi: f64, v: f64) -> f64 {
    if (hi - lo).abs() < f64::EPSILON {
        return 0.0;
    }
    let x = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Hash a float into a pseudo-random value in -1..1. Pure function: same input
/// always yields the same output (required for live/offline parity).
fn hash01(x: f64) -> f64 {
    // Cheap integer-ish hash on the bit pattern, scaled to -1..1.
    let bits = x.to_bits();
    let mut h = bits ^ (bits >> 33);
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    // Map the top 32 bits to 0..1, then to -1..1.
    let frac = (h >> 32) as f64 / u32::MAX as f64;
    frac * 2.0 - 1.0
}

/// Smooth value-noise over a 1D input: interpolates between hashed integer
/// samples so the output is continuous (no jitter between frames).
fn value_noise(x: f64) -> f64 {
    let i = x.floor();
    let f = x - i;
    let a = hash01(i);
    let b = hash01(i + 1.0);
    // Smoothstep blend between the two integer samples.
    let u = f * f * (3.0 - 2.0 * f);
    a + (b - a) * u
}

/// Deterministic wiggle: smooth value-noise oscillation at `freq` Hz, scaled
/// by `amp`. Output in roughly -amp..amp.
fn wiggle(freq: f64, amp: f64, t: f64) -> f64 {
    value_noise(t * freq) * amp
}

/// Deterministic noise sample for a given seed, output -1..1.
fn noise(seed: f64) -> f64 {
    value_noise(seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Valid expressions compile successfully while unbalanced or empty source
    /// surfaces a parse error instead of panicking.
    #[test]
    fn expr_parses_valid_and_rejects_garbage() {
        eprintln!("automation: Expr::new accepts valid exprs and rejects garbage");
        assert!(Expr::new("0.5 + 0.5*sin(t*tau)").is_ok());
        assert!(Expr::new("sin(beat * tau)").is_ok());
        // Unbalanced / nonsense should surface a parse error, not panic.
        assert!(Expr::new("((((").is_err());
        assert!(Expr::new("").is_err());
    }

    /// Evaluating the same expression with identical inputs yields identical
    /// output, guaranteeing live/offline render parity.
    #[test]
    fn eval_is_deterministic_for_same_inputs() {
        eprintln!("automation: Expr::eval is deterministic for equal inputs");
        let e = Expr::new("sin(t) + wiggle(2, 1) + noise(t)").unwrap();
        let a = e.eval(1.234, 0.0, 120.0);
        let b = e.eval(1.234, 0.0, 120.0);
        assert_eq!(
            a, b,
            "same inputs must yield identical output (live/offline parity)"
        );
    }

    /// Eval errors (unknown functions) and non-finite results (divide by zero)
    /// collapse to 0.0 rather than panicking.
    #[test]
    fn eval_collapses_errors_and_non_finite_to_zero() {
        eprintln!("automation: Expr::eval collapses errors and non-finite to 0.0");
        // Unknown function → fasteval eval error → 0.0 (never panics).
        let bad = Expr::new("totally_unknown_fn(t)").unwrap();
        assert_eq!(bad.eval(0.5, 0.0, 120.0), 0.0);
        // 1/0 → non-finite → 0.0.
        let div0 = Expr::new("1 / 0").unwrap();
        assert_eq!(div0.eval(0.0, 0.0, 120.0), 0.0);
    }

    /// The `t`, `beat`, `bpm`, and `pi` variables resolve to their expected
    /// values inside an evaluated expression.
    #[test]
    fn eval_exposes_time_and_tempo_vars() {
        eprintln!("automation: Expr::eval exposes t/beat/bpm/pi variables");
        assert_abs_diff_eq!(
            Expr::new("t").unwrap().eval(3.5, 0.0, 120.0),
            3.5,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            Expr::new("beat").unwrap().eval(0.0, 2.0, 120.0),
            2.0,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            Expr::new("bpm").unwrap().eval(0.0, 0.0, 128.0),
            128.0,
            epsilon = 1e-4
        );
        assert_abs_diff_eq!(
            Expr::new("pi").unwrap().eval(0.0, 0.0, 120.0),
            std::f32::consts::PI,
            epsilon = 1e-6
        );
    }

    /// The triangle oscillator hits -1 at phase 0, +1 at the midpoint, and is
    /// periodic at 1.
    #[test]
    fn triangle_wave_shape() {
        eprintln!("automation: tri() has correct triangle-wave shape");
        assert_abs_diff_eq!(tri(0.0), -1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(tri(0.5), 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(tri(1.0), -1.0, epsilon = 1e-9); // periodic
    }

    /// The sawtooth oscillator ramps from -1 at phase 0 through 0 at the midpoint
    /// and approaches +1 just before wrapping.
    #[test]
    fn sawtooth_wave_shape() {
        eprintln!("automation: saw() has correct sawtooth-wave shape");
        assert_abs_diff_eq!(saw(0.0), -1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(saw(0.5), 0.0, epsilon = 1e-9);
        // Just before the wrap it approaches +1.
        assert!(saw(0.999) > 0.99);
    }

    /// The square oscillator is +1 over the first half of its period and -1 over
    /// the second half.
    #[test]
    fn square_wave_shape() {
        eprintln!("automation: square() has correct square-wave shape");
        assert_eq!(square(0.0), 1.0);
        assert_eq!(square(0.25), 1.0);
        assert_eq!(square(0.5), -1.0);
        assert_eq!(square(0.75), -1.0);
    }

    /// Every oscillator and noise shape stays within the [-1, 1] range across a
    /// sweep of input values.
    #[test]
    fn oscillators_stay_in_unit_range() {
        eprintln!("automation: oscillators stay within [-1, 1] across a sweep");
        let mut x = -3.0;
        while x < 3.0 {
            for v in [tri(x), saw(x), square(x), pulse(x), fbm(x), hold(x)] {
                assert!((-1.0..=1.0).contains(&v), "osc out of [-1,1] at x={x}: {v}");
            }
            x += 0.013;
        }
    }

    /// Smoothstep returns its endpoints exactly, clamps outside the range, and
    /// returns 0.0 for a degenerate lo == hi (no divide-by-zero).
    #[test]
    fn smoothstep_endpoints_and_degenerate() {
        eprintln!("automation: smoothstep() endpoints, clamping, and degenerate case");
        assert_abs_diff_eq!(smoothstep(0.0, 1.0, 0.0), 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(smoothstep(0.0, 1.0, 1.0), 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(smoothstep(0.0, 1.0, 0.5), 0.5, epsilon = 1e-9);
        // Below/above the range clamps.
        assert_eq!(smoothstep(0.0, 1.0, -2.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
        // Degenerate lo == hi → 0.0 (no divide-by-zero).
        assert_eq!(smoothstep(1.0, 1.0, 5.0), 0.0);
    }

    /// The hash01 helper is a pure function bounded to [-1, 1] for a range of
    /// inputs.
    #[test]
    fn hash01_is_deterministic_and_bounded() {
        eprintln!("automation: hash01() is deterministic and bounded to [-1, 1]");
        for x in [-10.0, -1.5, 0.0, 1.0, 3.14159, 1000.0] {
            let v = hash01(x);
            assert!((-1.0..=1.0).contains(&v), "hash01({x})={v} out of range");
            assert_eq!(v, hash01(x), "hash01 must be a pure function");
        }
    }

    /// Value-noise equals hash01 at integer inputs and is deterministic between
    /// repeated evaluations.
    #[test]
    fn value_noise_is_continuous_at_integers() {
        eprintln!("automation: value_noise() is continuous and deterministic at integers");
        // At an integer the interpolation weight is 0, so it equals hash01(i).
        assert_abs_diff_eq!(value_noise(4.0), hash01(4.0), epsilon = 1e-9);
        // Deterministic between frames.
        assert_eq!(value_noise(2.37), value_noise(2.37));
    }
}
