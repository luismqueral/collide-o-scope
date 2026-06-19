//! Audio subsystem (Phase 1: per-layer mix + master bus).
//!
//! Audio can't ride the 30 fps render-loop "pull" clock — the hardware needs a
//! *continuous* sample stream delivered to a real-time callback. So this is its
//! own little pipeline, decoupled from rendering:
//!
//! ```text
//!   render loop ──cmd──► mixer thread ──one ring──► cpal callback ──► device
//!   (main.rs)            (decode+DSP+sum)  (lock-free)  (master+meter)
//! ```
//!
//! **One mixer thread** owns every per-layer source ([`LayerSource`]), decodes
//! + resamples + applies per-layer DSP (mute, dB volume, equal-power pan,
//! varispeed), and sums them all into a *single* output ring buffer. The
//! real-time cpal callback stays trivial: drain that one ring, apply the master
//! gain + limiter, store a meter peak. Per-layer changes flow over the command
//! channel and so lag by the (deliberately small, ~150 ms) output-ring latency;
//! the master fader/limiter live in the callback via atomics, so they respond
//! instantly. See docs/audio-plan.md §2 (we use a single mixer thread rather
//! than the per-layer-rings-summed-in-callback sketch — simpler + keeps the
//! callback lock-free with no multi-consumer juggling).
//!
//! Phase 1 scope: per-layer mute/volume/pan, varispeed (pitch-follows-speed),
//! play/pause + loop, master volume/limiter/meter. EQ/delay (Phase 2) and
//! standalone audio files (Phase 3) come later.

mod decoder;
mod dsp;
mod output;

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use decoder::AudioDecoder;
use dsp::LayerDsp;
use output::MasterControls;

/// dB floor: at or below this a layer (or the master) is fully silent rather
/// than an inaudible `10^(-60/20)` trickle.
const DB_FLOOR: f32 = -60.0;

/// Stereo frames produced per mixer iteration before checking the ring again.
const BLOCK_FRAMES: usize = 512;

/// Per-layer audio parameters. Plain data (not a GPU uniform) mirrored from the
/// layer's UI controls. Mute keeps the source *advancing* (so unmuting resumes
/// in time, like a mixer channel); `paused` (sent separately) freezes it.
#[derive(Clone, Copy, Debug)]
pub struct AudioParams {
    pub mute: bool,
    pub volume: f32, // dB, −60..+6 (0 = unity)
    pub pan: f32,    // −1 (L) .. 0 (C) .. +1 (R)
    // Phase 2 — Audio FX. Fixed 3-band EQ (shelves + mid peak), dB gains.
    pub eq_low: f32,  // −24..+12 dB (120 Hz low-shelf)
    pub eq_mid: f32,  // −24..+12 dB (1 kHz peak)
    pub eq_high: f32, // −24..+12 dB (6 kHz high-shelf)
    // Tap delay.
    pub delay_time: f32,     // 0..1000 ms (0 = bypass)
    pub delay_feedback: f32, // 0..0.95
    pub delay_mix: f32,      // 0..1 (dry↔wet)
}

impl Default for AudioParams {
    fn default() -> Self {
        Self {
            mute: false,
            volume: 0.0,
            pan: 0.0,
            eq_low: 0.0,
            eq_mid: 0.0,
            eq_high: 0.0,
            delay_time: 0.0,
            delay_feedback: 0.0,
            delay_mix: 0.0,
        }
    }
}

/// dB → linear gain, with a hard floor to true silence at/under `DB_FLOOR`.
fn db_to_gain(db: f32) -> f32 {
    if db <= DB_FLOOR {
        0.0
    } else {
        10f32.powf(db / 20.0)
    }
}

/// Equal-power pan → (left, right) multipliers. Center (0) gives ~0.707 each,
/// so panning hard doesn't change perceived loudness.
fn equal_power_pan(pan: f32) -> (f32, f32) {
    let angle = (pan.clamp(-1.0, 1.0) + 1.0) * 0.25 * std::f32::consts::PI; // 0..π/2
    (angle.cos(), angle.sin())
}

/// Commands the render loop sends to the mixer thread. Layers are keyed by the
/// stable `Layer::id` (survives reorders/removals), never by index.
enum AudioCommand {
    /// Register a new source decoding `path`'s audio track. No-ops silently if
    /// the file has no audio stream. `meter` is the shared peak cell the engine
    /// reads for this layer's UI meter.
    AddSource {
        id: u64,
        path: String,
        meter: Arc<AtomicU32>,
    },
    /// Drop a source (layer removed).
    RemoveSource { id: u64 },
    /// Swap a source's file in place (clip swap), keeping its params/speed.
    SetSourcePath { id: u64, path: String },
    /// Drop every source (patch load rebuilds layers from scratch).
    ClearSources,
    /// Freeze/unfreeze one source (per-layer pause).
    SetPaused { id: u64, paused: bool },
    /// Varispeed: playback multiplier (pitch-follows-speed).
    SetSpeed { id: u64, speed: f32 },
    /// Update one source's mute/volume/pan.
    SetParams { id: u64, params: AudioParams },
    /// Freeze/unfreeze the whole mix (master pause).
    SetMasterPaused { paused: bool },
}

/// One audio source = one layer's track, with a small device-rate decoded FIFO
/// and a fractional read cursor for varispeed.
struct LayerSource {
    id: u64,
    decoder: AudioDecoder,
    params: AudioParams,
    speed: f32,
    paused: bool,
    /// Decoded interleaved f32 at device rate (width = `out_channels`).
    fifo: VecDeque<f32>,
    /// Fractional read position in *frames*, normalised to [0, 1) each frame by
    /// dropping consumed frames off the FIFO front.
    read_pos: f64,
    /// Per-layer Audio FX state (EQ + delay), one slot per output channel.
    dsp: LayerDsp,
    /// Decoder hit an unrecoverable error — go silent but stay addressable so
    /// the layer can still be removed / clip-swapped by id.
    dead: bool,
    /// This source's post-FX peak level (0..1, f32 bits), written each mixed
    /// block and read by the UI for the per-layer meter. Shared with the engine
    /// (main thread) via an `Arc` keyed by layer id.
    meter: Arc<AtomicU32>,
}

impl LayerSource {
    fn frames_buffered(&self, ch: usize) -> usize {
        self.fifo.len() / ch
    }

    /// Pull one more decoded chunk into the FIFO. Returns false if nothing was
    /// added (EOF/unrecoverable or repeated empty chunks) so callers stop
    /// waiting for samples that aren't coming.
    fn decode_more(&mut self) -> bool {
        if self.dead {
            return false;
        }
        // A few empty chunks can occur during resampler warmup; bound the
        // attempts so we never spin forever inside the fill loop.
        for _ in 0..8 {
            match self.decoder.next_chunk() {
                Some(chunk) if !chunk.is_empty() => {
                    self.fifo.extend(chunk);
                    return true;
                }
                Some(_) => continue,
                None => {
                    self.dead = true;
                    return false;
                }
            }
        }
        false
    }

    /// Mix this source into `block` (interleaved, `out_channels` wide), summing
    /// on top of whatever is already there. Advances the varispeed read cursor;
    /// a paused source contributes nothing and does not advance.
    fn mix_into(&mut self, block: &mut [f32], out_channels: u16) {
        if self.paused {
            // Paused contributes nothing — drop the meter so the UI shows idle.
            self.meter.store(0.0f32.to_bits(), Ordering::Relaxed);
            return;
        }
        let ch = out_channels as usize;
        let frames = block.len() / ch;
        let gain = if self.params.mute {
            0.0
        } else {
            db_to_gain(self.params.volume)
        };
        let (lg, rg) = equal_power_pan(self.params.pan);
        let step = self.speed.max(0.0) as f64;

        // Refresh EQ coefficients / delay settings once per block (cheap unless
        // an EQ gain actually changed). DSP runs even when muted so the delay
        // tail stays time-aligned; gain=0 is applied *after* it.
        self.dsp.update_params(&self.params);

        // Track this source's post-FX peak across the block for the UI meter.
        let mut peak = 0.0f32;

        for f in 0..frames {
            // Drop frames we've advanced past so read_pos stays in [0, 1) and
            // the FIFO front is always our interpolation base.
            let drop = self.read_pos.floor() as usize;
            if drop > 0 {
                for _ in 0..drop {
                    if self.fifo.is_empty() {
                        break;
                    }
                    for _ in 0..ch {
                        self.fifo.pop_front();
                    }
                }
                self.read_pos -= drop as f64;
            }

            // Need frame 0 and frame 1 to interpolate between.
            while self.frames_buffered(ch) < 2 {
                if !self.decode_more() {
                    break;
                }
            }

            let buffered = self.frames_buffered(ch);
            if buffered == 0 {
                // Nothing to play (dead/drained). Keep advancing so we don't
                // spin; output stays silent for this source.
                self.read_pos += step.max(0.0001);
                continue;
            }

            let frac = self.read_pos as f32;
            for c in 0..ch {
                let s0 = self.fifo[c];
                let s1 = if buffered >= 2 {
                    self.fifo[ch + c]
                } else {
                    s0 // last frame: hold rather than interpolate into nothing
                };
                // interpolate → Audio FX (EQ + delay) → gain → pan
                let interp = s0 + (s1 - s0) * frac;
                let mut v = self.dsp.process(c, interp) * gain;
                if ch >= 2 {
                    if c == 0 {
                        v *= lg;
                    } else if c == 1 {
                        v *= rg;
                    }
                }
                block[f * ch + c] += v;
                let a = v.abs();
                if a > peak {
                    peak = a;
                }
            }
            self.read_pos += step;
        }

        self.meter.store(peak.to_bits(), Ordering::Relaxed);
    }
}

/// Owns the audio output stream + mixer thread + shared master controls. Built
/// once at startup and held on the main thread inside `App`. Dropping it stops
/// the stream and (by disconnecting the command channel) ends the mixer thread.
pub struct AudioEngine {
    cmd_tx: Sender<AudioCommand>,
    master: Arc<MasterControls>,
    // Per-layer peak cells keyed by layer id. The mixer thread holds a clone of
    // each `Arc` inside its `LayerSource` and writes the peak per block; the
    // main thread reads here for the UI meter. Only touched on the main thread,
    // so a `RefCell` suffices (no cross-thread access to the map itself).
    layer_meters: RefCell<HashMap<u64, Arc<AtomicU32>>>,
    // The cpal stream is kept alive for its whole lifetime; it is `!Send` on
    // some platforms, which is why `AudioEngine` lives on the main thread.
    _stream: cpal::Stream,
    _mixer_thread: thread::JoinHandle<()>,
}

impl AudioEngine {
    /// Initialise the default output device, ring buffer, output stream, and
    /// mixer thread. Returns an error (so the caller can run silently) if no
    /// usable f32 output device is available.
    pub fn new() -> Result<Self, String> {
        let (device, config) = output::default_output()?;
        let out_rate = config.sample_rate.0;
        let out_channels = config.channels;

        // ~150 ms of interleaved output buffer: small enough that per-layer
        // param changes (which flow through the mixer) feel responsive, large
        // enough to absorb mixer-thread scheduling jitter and 4× varispeed
        // worst case. Master controls bypass this latency (applied in callback).
        let capacity = (out_rate as usize * out_channels as usize * 3) / 20;
        let (producer, consumer) = rtrb::RingBuffer::<f32>::new(capacity);

        let master = Arc::new(MasterControls::new());
        let stream = output::build_stream(&device, &config, consumer, Arc::clone(&master))?;

        let (cmd_tx, cmd_rx) = mpsc::channel::<AudioCommand>();
        let mixer_thread = thread::Builder::new()
            .name("audio-mixer".into())
            .spawn(move || mixer_loop(cmd_rx, producer, out_rate, out_channels))
            .map_err(|e| format!("Failed to spawn audio mixer thread: {e}"))?;

        Ok(Self {
            cmd_tx,
            master,
            layer_meters: RefCell::new(HashMap::new()),
            _stream: stream,
            _mixer_thread: mixer_thread,
        })
    }

    /// Register a layer's audio track. Fire-and-forget: a missing audio stream
    /// (e.g. a silent clip or image) just yields no source (the meter cell then
    /// simply stays at 0).
    pub fn add_source(&self, id: u64, path: &str) {
        let meter = Arc::new(AtomicU32::new(0.0f32.to_bits()));
        self.layer_meters
            .borrow_mut()
            .insert(id, Arc::clone(&meter));
        let _ = self.cmd_tx.send(AudioCommand::AddSource {
            id,
            path: path.to_string(),
            meter,
        });
    }

    /// Drop a layer's source (layer removed).
    pub fn remove_source(&self, id: u64) {
        self.layer_meters.borrow_mut().remove(&id);
        let _ = self.cmd_tx.send(AudioCommand::RemoveSource { id });
    }

    /// Swap a layer's underlying file in place (clip swap), keeping params.
    pub fn set_source_path(&self, id: u64, path: &str) {
        let _ = self.cmd_tx.send(AudioCommand::SetSourcePath {
            id,
            path: path.to_string(),
        });
    }

    /// Drop all sources (patch load rebuilds layers from scratch).
    pub fn clear_sources(&self) {
        self.layer_meters.borrow_mut().clear();
        let _ = self.cmd_tx.send(AudioCommand::ClearSources);
    }

    /// Freeze/unfreeze one layer's audio (per-layer pause).
    pub fn set_paused(&self, id: u64, paused: bool) {
        let _ = self.cmd_tx.send(AudioCommand::SetPaused { id, paused });
    }

    /// Varispeed: set a layer's playback multiplier.
    pub fn set_speed(&self, id: u64, speed: f32) {
        let _ = self.cmd_tx.send(AudioCommand::SetSpeed { id, speed });
    }

    /// Update a layer's mute/volume/pan.
    pub fn set_params(&self, id: u64, params: AudioParams) {
        let _ = self.cmd_tx.send(AudioCommand::SetParams { id, params });
    }

    /// Freeze/unfreeze the whole mix (master pause).
    pub fn set_master_paused(&self, paused: bool) {
        let _ = self.cmd_tx.send(AudioCommand::SetMasterPaused { paused });
    }

    /// Set master volume in dB (instant — written straight to the callback's
    /// atomic, bypassing the mix ring latency).
    pub fn set_master_volume(&self, db: f32) {
        self.master
            .gain
            .store(db_to_gain(db).to_bits(), Ordering::Relaxed);
    }

    /// Enable/disable the master limiter (instant).
    pub fn set_master_limiter(&self, on: bool) {
        self.master.limiter.store(on, Ordering::Relaxed);
    }

    /// Current output peak level (0..1) for the master UI meter.
    pub fn meter(&self) -> f32 {
        f32::from_bits(self.master.meter.load(Ordering::Relaxed))
    }

    /// Current post-FX peak level (0..1) for one layer's UI meter. Returns 0 for
    /// an unknown id or a layer with no audio source.
    pub fn layer_meter(&self, id: u64) -> f32 {
        self.layer_meters
            .borrow()
            .get(&id)
            .map(|m| f32::from_bits(m.load(Ordering::Relaxed)))
            .unwrap_or(0.0)
    }
}

/// Mixer-thread body: maintain the live set of sources, and each iteration sum
/// one block of every source's varispeed/DSP output into the single output ring
/// for the callback to drain. Block-based and *not* real-time-critical, so it's
/// free to sleep/back off when the ring is full.
fn mixer_loop(
    cmd_rx: Receiver<AudioCommand>,
    mut producer: rtrb::Producer<f32>,
    out_rate: u32,
    out_channels: u16,
) {
    let mut sources: Vec<LayerSource> = Vec::new();
    let mut master_paused = false;
    let block_len = BLOCK_FRAMES * out_channels as usize;
    let mut block = vec![0.0f32; block_len];

    loop {
        // Drain all pending commands (non-blocking).
        loop {
            match cmd_rx.try_recv() {
                Ok(cmd) => apply_command(
                    cmd,
                    &mut sources,
                    &mut master_paused,
                    out_rate,
                    out_channels,
                ),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return, // engine dropped → exit
            }
        }

        // Back off until there's room for a full block (keeps pushes simple and
        // avoids partial-block bookkeeping).
        if producer.slots() < block_len {
            thread::sleep(Duration::from_millis(2));
            continue;
        }

        // Sum this block. Master pause = push silence (keeps the callback fed so
        // it doesn't underrun-crackle) without advancing any source.
        for s in block.iter_mut() {
            *s = 0.0;
        }
        if !master_paused {
            for src in sources.iter_mut() {
                src.mix_into(&mut block, out_channels);
            }
        }

        for &s in block.iter() {
            let _ = producer.push(s); // room guaranteed by the slots() check
        }
    }
}

/// Apply one command to the mixer's source set. Split out for readability;
/// `out_rate`/`out_channels` are needed to (re)open decoders at device rate.
fn apply_command(
    cmd: AudioCommand,
    sources: &mut Vec<LayerSource>,
    master_paused: &mut bool,
    out_rate: u32,
    out_channels: u16,
) {
    match cmd {
        AudioCommand::AddSource { id, path, meter } => {
            match AudioDecoder::open(&path, out_rate, out_channels) {
                Ok(decoder) => sources.push(LayerSource {
                    id,
                    decoder,
                    params: AudioParams::default(),
                    speed: 1.0,
                    paused: false,
                    fifo: VecDeque::new(),
                    read_pos: 0.0,
                    dsp: LayerDsp::new(out_rate, out_channels),
                    dead: false,
                    meter,
                }),
                // No audio stream (image / silent clip) is normal — stay silent.
                Err(e) => eprintln!("audio: no source for {path}: {e}"),
            }
        }
        AudioCommand::RemoveSource { id } => sources.retain(|s| s.id != id),
        AudioCommand::ClearSources => sources.clear(),
        AudioCommand::SetSourcePath { id, path } => {
            if let Some(src) = sources.iter_mut().find(|s| s.id == id) {
                match AudioDecoder::open(&path, out_rate, out_channels) {
                    Ok(decoder) => {
                        src.decoder = decoder;
                        src.fifo.clear();
                        src.read_pos = 0.0;
                        src.dsp.reset();
                        src.dead = false;
                    }
                    Err(e) => {
                        // New clip has no audio: go silent but keep the slot.
                        eprintln!("audio: clip swap {path} has no audio: {e}");
                        src.fifo.clear();
                        src.read_pos = 0.0;
                        src.dsp.reset();
                        src.dead = true;
                    }
                }
            }
        }
        AudioCommand::SetPaused { id, paused } => {
            if let Some(src) = sources.iter_mut().find(|s| s.id == id) {
                src.paused = paused;
            }
        }
        AudioCommand::SetSpeed { id, speed } => {
            if let Some(src) = sources.iter_mut().find(|s| s.id == id) {
                src.speed = speed;
            }
        }
        AudioCommand::SetParams { id, params } => {
            if let Some(src) = sources.iter_mut().find(|s| s.id == id) {
                src.params = params;
            }
        }
        AudioCommand::SetMasterPaused { paused } => *master_paused = paused,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// `db_to_gain` hits its reference points: 0 dB unity, −6 dB ≈ half, +6 dB ≈ 2×.
    #[test]
    fn db_to_gain_reference_points() {
        eprintln!("audio: db_to_gain matches its dB reference points");
        assert_abs_diff_eq!(db_to_gain(0.0), 1.0, epsilon = 1e-6); // unity
        assert_abs_diff_eq!(db_to_gain(-6.0), 0.5012, epsilon = 1e-3); // ≈ half amplitude
        assert_abs_diff_eq!(db_to_gain(6.0), 1.9953, epsilon = 1e-3); // ≈ +6 dB
    }

    /// `db_to_gain` returns exactly zero at or below the dB floor.
    #[test]
    fn db_to_gain_floors_to_silence() {
        eprintln!("audio: db_to_gain floors to exact silence at/below the dB floor");
        // At or below the floor it is *exactly* zero, not a tiny trickle.
        assert_eq!(db_to_gain(DB_FLOOR), 0.0);
        assert_eq!(db_to_gain(-60.0), 0.0);
        assert_eq!(db_to_gain(-120.0), 0.0);
    }

    /// `equal_power_pan` gives ~0.707 each at center and full level on hard sides.
    #[test]
    fn equal_power_pan_center_and_hard_sides() {
        eprintln!("audio: equal_power_pan yields center 0.707 and hard-side full level");
        let (l, r) = equal_power_pan(0.0);
        assert_abs_diff_eq!(l, std::f32::consts::FRAC_1_SQRT_2, epsilon = 1e-6);
        assert_abs_diff_eq!(r, std::f32::consts::FRAC_1_SQRT_2, epsilon = 1e-6);

        let (l, r) = equal_power_pan(-1.0); // hard left
        assert_abs_diff_eq!(l, 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(r, 0.0, epsilon = 1e-6);

        let (l, r) = equal_power_pan(1.0); // hard right
        assert_abs_diff_eq!(l, 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(r, 1.0, epsilon = 1e-6);
    }

    /// `equal_power_pan` clamps out-of-range input and keeps l²+r²≈1 across the sweep.
    #[test]
    fn equal_power_pan_clamps_and_conserves_power() {
        eprintln!("audio: equal_power_pan clamps input and conserves power across the sweep");
        // Out-of-range pan is clamped to the hard-side result.
        assert_eq!(equal_power_pan(5.0), equal_power_pan(1.0));
        assert_eq!(equal_power_pan(-5.0), equal_power_pan(-1.0));
        // Equal-power: l² + r² ≈ 1 across the sweep (loudness preserved).
        let mut p = -1.0f32;
        while p <= 1.0 {
            let (l, r) = equal_power_pan(p);
            assert_abs_diff_eq!(l * l + r * r, 1.0, epsilon = 1e-5);
            p += 0.1;
        }
    }

    /// `AudioParams::default` is fully neutral: no mute and all gains/FX at zero.
    #[test]
    fn audio_params_default_is_neutral() {
        eprintln!("audio: AudioParams::default is neutral with all gains and FX at zero");
        let p = AudioParams::default();
        assert!(!p.mute);
        assert_eq!(p.volume, 0.0);
        assert_eq!(p.pan, 0.0);
        assert_eq!(p.eq_low, 0.0);
        assert_eq!(p.eq_mid, 0.0);
        assert_eq!(p.eq_high, 0.0);
        assert_eq!(p.delay_time, 0.0);
        assert_eq!(p.delay_feedback, 0.0);
        assert_eq!(p.delay_mix, 0.0);
    }
}
