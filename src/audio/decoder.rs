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
        })
    }

    /// Decode and resample the next available chunk of audio, returning it as
    /// interleaved f32 (`[L, R, L, R, ...]`). Loops back to the start at EOF.
    /// Returns None only on an unrecoverable error.
    ///
    /// Same "push packets in, pull frames out" dance as the video decoder: one
    /// packet may yield zero or many frames, so we drain ready frames first and
    /// only feed a new packet when the decoder has none buffered.
    pub fn next_chunk(&mut self) -> Option<Vec<f32>> {
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
                    // EOF — flush the decoder, then reopen to loop.
                    self.decoder.send_eof().ok();
                    let mut decoded = AudioFrame::empty();
                    if self.decoder.receive_frame(&mut decoded).is_ok() {
                        return Some(self.resample(&decoded));
                    }
                    if self.reopen().is_err() {
                        return None;
                    }
                }
            }
        }
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
}
