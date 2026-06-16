//! cpal output device + real-time stream setup.
//!
//! The consumer end of the lock-free ring buffer lives in cpal's audio
//! callback, which runs on a dedicated real-time thread. The callback does the
//! absolute minimum: drain the *pre-master* mix out of the ring buffer, apply
//! the master gain + (optional) limiter, write a peak level to the meter, and
//! substitute silence (0.0) on underrun. Master controls live here — read from
//! atomics — so a fader move is heard immediately rather than after the
//! ~150 ms of mix already buffered ahead in the ring. No locks, no allocation,
//! no decode — anything heavier would risk glitches/dropouts.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// Pick the default output device and its default config, requiring f32 output
/// (the common macOS CoreAudio default). Returns the device + the concrete
/// `StreamConfig` (channel count + sample rate) the rest of the engine targets.
pub fn default_output() -> Result<(cpal::Device, cpal::StreamConfig), String> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or("No default output audio device")?;

    let supported = device
        .default_output_config()
        .map_err(|e| format!("No default output config: {e}"))?;

    // v1 only handles f32 streams. Most hosts (incl. CoreAudio) default to f32,
    // so we bail clearly rather than silently mishandling another format.
    if supported.sample_format() != cpal::SampleFormat::F32 {
        return Err(format!(
            "Unsupported output sample format {:?} (only f32 supported in v1)",
            supported.sample_format()
        ));
    }

    Ok((device, supported.config()))
}

/// Shared master controls, written by the main-thread `AudioEngine` handle and
/// read by the real-time callback. Atomics (not a `Mutex`) so the callback
/// never blocks. `gain`/`meter` store `f32` bit patterns via `to_bits`/
/// `from_bits`.
pub struct MasterControls {
    /// Linear master gain (f32 bits). 1.0 = unity.
    pub gain: AtomicU32,
    /// Brick-wall clamp to [-1, 1] when true.
    pub limiter: AtomicBool,
    /// Output peak (0..1, f32 bits) written each callback for the UI meter.
    pub meter: AtomicU32,
}

impl MasterControls {
    pub fn new() -> Self {
        Self {
            gain: AtomicU32::new(1.0f32.to_bits()),
            limiter: AtomicBool::new(true),
            meter: AtomicU32::new(0.0f32.to_bits()),
        }
    }
}

/// Build and start the output stream. The data callback drains `consumer`
/// (interleaved f32 matching `config.channels`), applies the master controls,
/// and writes silence whenever the producer hasn't kept up. The returned
/// `Stream` must be kept alive — dropping it stops playback.
pub fn build_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut consumer: rtrb::Consumer<f32>,
    master: Arc<MasterControls>,
) -> Result<cpal::Stream, String> {
    let stream = device
        .build_output_stream(
            config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let gain = f32::from_bits(master.gain.load(Ordering::Relaxed));
                let limit = master.limiter.load(Ordering::Relaxed);
                let mut peak = 0.0f32;
                for sample in data.iter_mut() {
                    let mut v = consumer.pop().unwrap_or(0.0) * gain;
                    if limit {
                        v = v.clamp(-1.0, 1.0);
                    }
                    let a = v.abs();
                    if a > peak {
                        peak = a;
                    }
                    *sample = v;
                }
                master.meter.store(peak.to_bits(), Ordering::Relaxed);
            },
            |err| eprintln!("audio output stream error: {err}"),
            None,
        )
        .map_err(|e| format!("Failed to build output stream: {e}"))?;

    stream
        .play()
        .map_err(|e| format!("Failed to start output stream: {e}"))?;

    Ok(stream)
}
