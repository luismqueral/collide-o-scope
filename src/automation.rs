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
