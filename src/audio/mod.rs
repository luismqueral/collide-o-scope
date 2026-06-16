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
mod output;

use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use decoder::AudioDecoder;
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
}

impl Default for AudioParams {
    fn default() -> Self {
        Self {
            mute: false,
            volume: 0.0,
            pan: 0.0,
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
    /// the file has no audio stream.
    AddSource { id: u64, path: String },
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
    /// Decoder hit an unrecoverable error — go silent but stay addressable so
    /// the layer can still be removed / clip-swapped by id.
    dead: bool,
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
                let mut v = (s0 + (s1 - s0) * frac) * gain;
                if ch >= 2 {
                    if c == 0 {
                        v *= lg;
                    } else if c == 1 {
                        v *= rg;
                    }
                }
                block[f * ch + c] += v;
            }
            self.read_pos += step;
        }
    }
}

/// Owns the audio output stream + mixer thread + shared master controls. Built
/// once at startup and held on the main thread inside `App`. Dropping it stops
/// the stream and (by disconnecting the command channel) ends the mixer thread.
pub struct AudioEngine {
    cmd_tx: Sender<AudioCommand>,
    master: Arc<MasterControls>,
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
            _stream: stream,
            _mixer_thread: mixer_thread,
        })
    }

    /// Register a layer's audio track. Fire-and-forget: a missing audio stream
    /// (e.g. a silent clip or image) just yields no source.
    pub fn add_source(&self, id: u64, path: &str) {
        let _ = self.cmd_tx.send(AudioCommand::AddSource {
            id,
            path: path.to_string(),
        });
    }

    /// Drop a layer's source (layer removed).
    pub fn remove_source(&self, id: u64) {
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
        let _ = self
            .cmd_tx
            .send(AudioCommand::SetMasterPaused { paused });
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

    /// Current output peak level (0..1) for the UI meter.
    pub fn meter(&self) -> f32 {
        f32::from_bits(self.master.meter.load(Ordering::Relaxed))
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
                Ok(cmd) => apply_command(cmd, &mut sources, &mut master_paused, out_rate, out_channels),
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
        AudioCommand::AddSource { id, path } => {
            match AudioDecoder::open(&path, out_rate, out_channels) {
                Ok(decoder) => sources.push(LayerSource {
                    id,
                    decoder,
                    params: AudioParams::default(),
                    speed: 1.0,
                    paused: false,
                    fifo: VecDeque::new(),
                    read_pos: 0.0,
                    dead: false,
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
                        src.dead = false;
                    }
                    Err(e) => {
                        // New clip has no audio: go silent but keep the slot.
                        eprintln!("audio: clip swap {path} has no audio: {e}");
                        src.fifo.clear();
                        src.read_pos = 0.0;
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
