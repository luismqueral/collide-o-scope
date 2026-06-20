//! Audio decoding via ffmpeg.
//!
//! Sibling of `video/decoder.rs`: wraps `ffmpeg-next` to pull decoded audio out
//! of a media file, but instead of one RGBA frame per call it yields a *chunk*
//! of interleaved stereo f32 samples already resampled to the output device's
//! rate. The audio engine's decode thread calls `next_chunk()` repeatedly and
//! pushes the result into a ring buffer for the real-time callback to drain.
//!
//! Like the video path, it loops at EOF by reopening the file from the top
//! rather than seeking — simplest and robust across codecs (see `reopen`).

use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::{context::Input, input};
use ffmpeg_next::media::Type;
use ffmpeg_next::software::resampling::context::Context as Resampler;
use ffmpeg_next::util::format::sample::{Sample, Type as SampleType};
use ffmpeg_next::util::frame::audio::Audio as AudioFrame;
use ffmpeg_next::ChannelLayout;

/// Owns one open media file and decodes its audio track, resampling each frame
/// to interleaved f32 at the device's sample rate / channel count.
pub struct AudioDecoder {
    path: String,                    // kept so we can reopen to loop
    input_ctx: Input,                // ffmpeg demuxer
    stream_index: usize,             // which stream is the audio track
    decoder: ffmpeg::decoder::Audio, // compressed packets → raw sample frames
    resampler: Resampler,            // native format/rate → f32 packed @ out_rate
    src_rate: u32,                   // decoder's native sample rate (for capacity math)
    out_rate: u32,                   // device sample rate we resample to
    out_channels: u16,               // device channel count (2 = stereo for v1)
    /// Loop window as fractions of the clip `0.0..1.0`, mirroring the video
    /// decoder so audio trims to the *same* slice. Default `0.0..1.0` = loop the
    /// whole file (the original behavior). `set_loop` narrows it.
    loop_start: f64,
    loop_end: f64,
    /// Total clip length measured in *output frames* (one frame = one sample per
    /// channel, at `out_rate`). An estimate from the container duration; only
    /// consulted when the clip is trimmed, so a bad estimate can't affect the
    /// default whole-file loop.
    total_out_frames: u64,
    /// Output frames emitted since the current loop-in point. The audio analog
    /// of the video decoder's `frame_count`: drives the out-point check.
    emitted_frames: u64,
}

impl AudioDecoder {
    /// Open a file's audio stream and set up a resampler to `out_rate` /
    /// `out_channels` interleaved f32. Errors (as `String`) if the file has no
    /// audio stream or ffmpeg can't build the decoder/resampler.
    pub fn open(path: &str, out_rate: u32, out_channels: u16) -> Result<Self, String> {
        ffmpeg::init().map_err(|e| format!("ffmpeg init failed: {e}"))?;

        let input_ctx = input(&path).map_err(|e| format!("Cannot open {path}: {e}"))?;

        // Pick the primary audio track. `ok_or` turns the Option into a Result
        // so `?` can bail with our message if the file has no audio at all.
        let stream = input_ctx
            .streams()
            .best(Type::Audio)
            .ok_or("No audio stream found")?;
        let stream_index = stream.index();

        let codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("Codec params: {e}"))?;
        let decoder = codec_ctx
            .decoder()
            .audio()
            .map_err(|e| format!("Decoder: {e}"))?;

        let src_rate = decoder.rate();

        // Estimate the clip length in output frames from the container duration.
        // `duration()` is in AV_TIME_BASE units (microseconds); × out_rate gives
        // frames at our device rate. Clamp to ≥1 so the window maths never divide
        // by zero. Only used when trimmed — untrimmed playback ignores it.
        let duration_secs = input_ctx.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE);
        let total_out_frames = ((duration_secs.max(0.0) * out_rate as f64) as u64).max(1);

        // Build the resampler: native (format, layout, rate) → f32 packed,
        // requested layout, device rate. `decoder.resampler(...)` reads the
        // input side off the decoder itself, so we only specify the output.
        let out_layout = ChannelLayout::default(out_channels as i32);
        let resampler = decoder
            .resampler(Sample::F32(SampleType::Packed), out_layout, out_rate)
            .map_err(|e| format!("Resampler: {e}"))?;

        Ok(Self {
            path: path.to_string(),
            input_ctx,
            stream_index,
            decoder,
            resampler,
            src_rate,
            out_rate,
            out_channels,
            // Default window = whole clip; `set_loop` narrows it later.
            loop_start: 0.0,
            loop_end: 1.0,
            total_out_frames,
            emitted_frames: 0,
        })
    }

    /// Decode and resample the next available chunk of audio, returning it as
    /// interleaved f32 (`[L, R, L, R, ...]`). Loops back to the window's start
    /// (the loop-in point, or the file start when untrimmed) at the out-point or
    /// true EOF. Returns None only on an unrecoverable error.
    ///
    /// Trimmed clips can't land on the loop-out point exactly — chunks are
    /// coarse (~21 ms) — so we truncate the chunk that crosses it and accept a
    /// little drift, consistent with the looping philosophy in docs/audio-plan
    /// §6. Untrimmed clips (the default 0..1 window) skip all of this and just
    /// loop at EOF, byte-for-byte the original behavior.
    pub fn next_chunk(&mut self) -> Option<Vec<f32>> {
        // Reached the out-point on a previous call → loop before decoding more.
        if self.is_trimmed() && self.emitted_frames >= self.end_frame() {
            self.loop_to_start().ok()?;
        }

        // Decode the next chunk; on true EOF, loop to the window start and retry
        // once (covers untrimmed whole-file looping and short-clip estimates).
        let chunk = match self.decode_chunk() {
            Some(c) => c,
            None => {
                self.loop_to_start().ok()?;
                self.decode_chunk()?
            }
        };

        let per_frame = self.out_channels.max(1) as usize;
        let frames = (chunk.len() / per_frame) as u64;

        // If this chunk would carry us past the out-point, keep only the part
        // up to it; the next call loops back to the in-point.
        if self.is_trimmed() && self.emitted_frames + frames > self.end_frame() {
            let allow = self.end_frame().saturating_sub(self.emitted_frames) as usize;
            let mut chunk = chunk;
            chunk.truncate(allow * per_frame);
            self.emitted_frames = self.end_frame();
            return Some(chunk);
        }

        self.emitted_frames += frames;
        Some(chunk)
    }

    /// One "push packets in, pull frames out" decode step: returns the next
    /// resampled chunk, or `None` at true end-of-file (no auto-reopen — callers
    /// decide whether/where to loop). One packet may yield zero or many frames,
    /// so we drain ready frames first and only feed a new packet when none are
    /// buffered.
    fn decode_chunk(&mut self) -> Option<Vec<f32>> {
        loop {
            // Try to receive an already-decoded frame first.
            let mut decoded = AudioFrame::empty();
            if self.decoder.receive_frame(&mut decoded).is_ok() {
                return Some(self.resample(&decoded));
            }

            // Feed more packets to the decoder.
            match self.next_audio_packet() {
                Some(packet) => {
                    if self.decoder.send_packet(&packet).is_err() {
                        continue;
                    }
                    let mut decoded = AudioFrame::empty();
                    if self.decoder.receive_frame(&mut decoded).is_ok() {
                        return Some(self.resample(&decoded));
                    }
                }
                None => {
                    // EOF — flush the decoder for any final buffered frame.
                    self.decoder.send_eof().ok();
                    let mut decoded = AudioFrame::empty();
                    if self.decoder.receive_frame(&mut decoded).is_ok() {
                        return Some(self.resample(&decoded));
                    }
                    return None; // truly drained
                }
            }
        }
    }

    /// Reopen the file and discard chunks up to the loop-in point so the next
    /// decode yields the window's start. Mirrors the video decoder's
    /// `loop_to_start`; the discard cost is bounded by the clip length. Discards
    /// happen in whole chunks, so we may overshoot the in-point by up to one
    /// chunk — the accepted "little loop drift."
    fn loop_to_start(&mut self) -> Result<(), String> {
        self.reopen()?; // resets emitted_frames = 0
        let start = self.start_frame();
        if start > 0 {
            let per_frame = self.out_channels.max(1) as usize;
            let mut discarded = 0u64;
            while discarded < start {
                match self.decode_chunk() {
                    Some(chunk) => discarded += (chunk.len() / per_frame) as u64,
                    // Clip shorter than the requested start (estimate was high):
                    // stop discarding and play from wherever we landed.
                    None => break,
                }
            }
            self.emitted_frames = discarded;
        }
        Ok(())
    }

    /// True when the loop window is narrower than the whole clip. Gates the
    /// windowing so untrimmed clips behave exactly as they did before.
    fn is_trimmed(&self) -> bool {
        self.loop_start > 0.0 || self.loop_end < 1.0
    }

    /// Output-frame index of the loop-in point.
    fn start_frame(&self) -> u64 {
        (self.loop_start * self.total_out_frames as f64).round() as u64
    }

    /// Output-frame index of the loop-out point. Clamped to keep at least one
    /// frame in the window and never exceed the clip length.
    fn end_frame(&self) -> u64 {
        let end = (self.loop_end * self.total_out_frames as f64).round() as u64;
        let lo = self.start_frame() + 1;
        let hi = self.total_out_frames.max(lo);
        end.clamp(lo, hi)
    }

    /// Restrict looping to the window `[start, end]` (fractions of the clip,
    /// `0.0..1.0`), matching `VideoDecoder::set_loop` so audio and video trim to
    /// the same slice. Clamps to range and keeps the out-point at least ~one
    /// frame past the in-point.
    pub fn set_loop(&mut self, start: f32, end: f32) {
        let min_gap = if self.total_out_frames > 0 {
            1.0 / self.total_out_frames as f64
        } else {
            0.01
        };
        let start = (start.clamp(0.0, 1.0) as f64).min(1.0 - min_gap);
        let end = (end.clamp(0.0, 1.0) as f64).clamp(start + min_gap, 1.0);
        self.loop_start = start;
        self.loop_end = end;
    }

    /// Run one decoded frame through the resampler and copy out interleaved f32.
    ///
    /// We pre-allocate the output frame to a generous capacity rather than let
    /// `run()` size it to `input.samples()`. When upsampling (e.g. 44.1→48 kHz)
    /// a same-size output can't hold all converted samples, and ffmpeg buffers
    /// the overflow in an internal FIFO that *grows every call* — an ever-
    /// increasing latency/backlog. Sizing for the converted count (plus margin
    /// for resampler latency) keeps the FIFO drained. After `run()`,
    /// `out.samples()` is the actual number written.
    fn resample(&mut self, frame: &AudioFrame) -> Vec<f32> {
        let cap = (frame.samples() as u64 * self.out_rate as u64 / self.src_rate.max(1) as u64)
            as usize
            + 256;
        let layout = ChannelLayout::default(self.out_channels as i32);
        let mut out = AudioFrame::new(Sample::F32(SampleType::Packed), cap, layout);

        if self.resampler.run(frame, &mut out).is_err() {
            return Vec::new();
        }

        // Packed/interleaved: plane 0 holds every channel. `data(0)` is the raw
        // byte slice (length = linesize, which may be padded), so we bound the
        // read to the valid sample count: samples × channels × 4 bytes/f32.
        let n_bytes = (out.samples() * self.out_channels as usize * 4).min(out.data(0).len());
        out.data(0)[..n_bytes]
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    /// Pull the next packet belonging to our audio stream, skipping other
    /// streams (e.g. video). Returns None at end-of-file.
    fn next_audio_packet(&mut self) -> Option<ffmpeg::Packet> {
        for (stream, packet) in self.input_ctx.packets() {
            if stream.index() == self.stream_index {
                return Some(packet);
            }
        }
        None
    }

    /// Re-open the file from the top to loop playback. Rebuilds both the decoder
    /// and the resampler fresh; the resampler's few buffered latency samples are
    /// discarded, which is a negligible click at the loop point (v1 accepts a
    /// little loop drift — see docs/audio-plan.md §6).
    fn reopen(&mut self) -> Result<(), String> {
        self.input_ctx =
            input(&self.path).map_err(|e| format!("Cannot reopen {}: {e}", self.path))?;

        let stream = self
            .input_ctx
            .streams()
            .best(Type::Audio)
            .ok_or("No audio stream on reopen")?;

        let codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("Codec params on reopen: {e}"))?;
        self.decoder = codec_ctx
            .decoder()
            .audio()
            .map_err(|e| format!("Decoder on reopen: {e}"))?;

        let out_layout = ChannelLayout::default(self.out_channels as i32);
        self.resampler = self
            .decoder
            .resampler(Sample::F32(SampleType::Packed), out_layout, self.out_rate)
            .map_err(|e| format!("Resampler on reopen: {e}"))?;

        self.emitted_frames = 0;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{synth_audio, synth_video};

    /// Opening a synthesized audio fixture succeeds and builds the decoder and
    /// resampler without error.
    #[test]
    fn open_succeeds_on_audio_fixture() {
        eprintln!("audio: AudioDecoder::open succeeds on a synthesized audio fixture");
        let (_dir, path) = synth_audio(440, 0.5);
        assert!(AudioDecoder::open(path.to_str().unwrap(), 48_000, 2).is_ok());
    }

    /// Opening a path that doesn't exist returns an Err naming the failure
    /// ("Cannot open") instead of panicking.
    #[test]
    fn open_errors_on_bogus_path() {
        eprintln!("audio: AudioDecoder::open errors on a missing file path");
        let err = match AudioDecoder::open("/no/such/file.m4a", 48_000, 2) {
            Ok(_) => panic!("expected open to fail on a missing file"),
            Err(e) => e,
        };
        assert!(err.contains("Cannot open"), "unexpected error: {err}");
    }

    /// Opening a video-only clip (no audio track) returns an Err.
    #[test]
    fn open_errors_on_file_without_audio() {
        eprintln!("audio: AudioDecoder::open errors on a file with no audio track");
        // A video-only `testsrc` clip has no audio track to decode.
        let (_dir, path) = synth_video(64, 48, 10, 0.5);
        assert!(AudioDecoder::open(path.to_str().unwrap(), 48_000, 2).is_err());
    }

    /// The first decoded chunk carries samples, is channel-aligned (even length
    /// for interleaved stereo), and contains only finite f32 values.
    #[test]
    fn next_chunk_returns_interleaved_stereo_f32() {
        eprintln!("audio: next_chunk returns finite, channel-aligned interleaved stereo f32");
        let (_dir, path) = synth_audio(440, 0.5);
        let mut dec = AudioDecoder::open(path.to_str().unwrap(), 48_000, 2).expect("open fixture");
        let chunk = dec.next_chunk().expect("decode a chunk");
        assert!(!chunk.is_empty(), "first chunk should carry samples");
        // Interleaved stereo → an even number of samples (L,R,L,R,...).
        assert_eq!(chunk.len() % 2, 0, "stereo chunk must be channel-aligned");
        // Decoded audio is finite.
        assert!(
            chunk.iter().all(|s| s.is_finite()),
            "samples must be finite"
        );
    }

    /// Pulling far more audio than the file holds keeps returning Some, proving
    /// the decoder loops cleanly past EOF by reopening.
    #[test]
    fn next_chunk_loops_cleanly_past_eof() {
        eprintln!("audio: next_chunk keeps yielding chunks by looping past EOF");
        let (_dir, path) = synth_audio(440, 0.3);
        let mut dec = AudioDecoder::open(path.to_str().unwrap(), 48_000, 2).expect("open fixture");
        // Pull far more audio than the 0.3 s file holds: the decoder loops by
        // reopening, so every chunk must still come back as Some.
        for i in 0..200 {
            assert!(
                dec.next_chunk().is_some(),
                "chunk {i} returned None (no clean loop)"
            );
        }
    }

    /// A freshly opened decoder is *not* trimmed: the window spans the whole
    /// file, so the default path is byte-for-byte the pre-trim whole-file loop.
    #[test]
    fn default_window_is_whole_clip() {
        eprintln!("audio: a new decoder loops the whole clip (untrimmed) by default");
        let (_dir, path) = synth_audio(440, 0.5);
        let dec = AudioDecoder::open(path.to_str().unwrap(), 48_000, 2).expect("open fixture");
        assert!(!dec.is_trimmed(), "default window should not be trimmed");
        assert_eq!(dec.start_frame(), 0, "default in-point is frame 0");
        assert_eq!(
            dec.end_frame(),
            dec.total_out_frames,
            "default out-point is the clip end"
        );
    }

    /// `set_loop` clamps out-of-range input to [0,1] and corrects an inverted
    /// pair (end <= start) into a valid window holding at least one frame.
    #[test]
    fn set_loop_clamps_and_orders() {
        eprintln!("audio: set_loop clamps to [0,1] and keeps end strictly after start");
        let (_dir, path) = synth_audio(440, 0.5);
        let mut dec = AudioDecoder::open(path.to_str().unwrap(), 48_000, 2).expect("open fixture");

        // Inverted input: end below start. Window must be corrected, not empty.
        dec.set_loop(0.8, 0.2);
        assert!(
            dec.loop_start <= dec.loop_end,
            "inverted input should be corrected: {} > {}",
            dec.loop_start,
            dec.loop_end
        );
        assert!(
            dec.end_frame() > dec.start_frame(),
            "window must hold at least one frame"
        );

        // Out-of-range input: clamped into [0,1].
        dec.set_loop(-1.0, 5.0);
        assert!(
            (0.0..=1.0).contains(&dec.loop_start),
            "start clamped to [0,1]"
        );
        assert!((0.0..=1.0).contains(&dec.loop_end), "end clamped to [0,1]");
        assert!(dec.loop_end >= dec.loop_start, "end stays >= start");
    }

    /// With a trimmed window (second half of the clip), playback loops *within*
    /// that window: chunks keep coming, the emitted-frame counter wraps back
    /// down at the out-point, and after wrapping it never falls below the
    /// in-point — proof we cycle the slice, not the whole file.
    #[test]
    fn trimmed_window_loops_within_bounds() {
        eprintln!("audio: a trimmed loop window cycles only its slice, not the whole clip");
        let (_dir, path) = synth_audio(440, 0.5);
        let mut dec = AudioDecoder::open(path.to_str().unwrap(), 48_000, 2).expect("open fixture");
        dec.set_loop(0.5, 1.0);
        let start = dec.start_frame();
        assert!(start > 0, "0.5 fraction should land past frame 0");

        // Pull well past the window length so it loops at least once.
        let mut emitted = Vec::new();
        for i in 0..200 {
            assert!(dec.next_chunk().is_some(), "chunk {i} returned None");
            emitted.push(dec.emitted_frames);
        }

        // A wrap happened (counter decreased = it looped rather than running on).
        let first_wrap = emitted
            .windows(2)
            .position(|w| w[1] < w[0])
            .unwrap_or_else(|| panic!("never wrapped: {emitted:?}"));
        // After wrapping, playback stays inside the window (never below the
        // in-point) — proof we loop the slice, not the whole clip.
        assert!(
            emitted[first_wrap + 1..].iter().all(|&c| c >= start),
            "left the loop window after wrap (start={start}): {emitted:?}"
        );
    }
}
