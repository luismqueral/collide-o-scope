//! Video decoding via ffmpeg.
//!
//! Wraps `ffmpeg-next` to pull one RGBA frame at a time out of a media file.
//! The render loop calls `next_frame()` ~30×/sec; this owns all the messy
//! ffmpeg state (input context, codec, pixel-format scaler) behind a small API.
//!
//! `use ... as ffmpeg;` is a Rust import alias — it renames the long crate name
//! `ffmpeg_next` to `ffmpeg` for the rest of this file. The other `use` lines
//! pull specific types into scope so we can write `Input` instead of the full
//! `ffmpeg_next::format::context::Input` path every time.
use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::{context::Input, input};
use ffmpeg_next::media::Type;
use ffmpeg_next::software::scaling::{context::Context as ScalerContext, flag::Flags};
use ffmpeg_next::util::frame::video::Video as VideoFrame;

/// Owns one open media file and decodes it frame-by-frame.
///
/// In Rust, a `struct` groups related data. Fields are private by default;
/// `pub` exposes them to other modules (here only `width`/`height` are public,
/// so callers can size GPU textures without reaching into ffmpeg internals).
pub struct VideoDecoder {
    path: String,                    // original file path, kept so we can reopen to loop
    input_ctx: Input,                // ffmpeg demuxer: reads packets out of the container
    stream_index: usize,             // which stream in the file is the video track
    decoder: ffmpeg::decoder::Video, // turns compressed packets into raw frames
    scaler: ScalerContext,           // converts the decoder's native pixel format → RGBA
    pub width: u32,
    pub height: u32,
    frame_count: u64,  // how many frames we've yielded since (re)open
    total_frames: u64, // estimated frame count, used only for progress()
    /// True for still images (png/jpg/etc): decode one frame then hold it,
    /// instead of re-opening the file every frame at EOF.
    still: bool,
}

impl VideoDecoder {
    /// Open a file and set up everything needed to decode it.
    ///
    /// Returns `Result<Self, String>`: `Ok(decoder)` on success, or `Err(msg)`
    /// with a human-readable error. The `?` operator below is Rust shorthand:
    /// "if this is an `Err`, return it from this function immediately; otherwise
    /// unwrap the `Ok` value." `map_err(...)` rewrites ffmpeg's error type into
    /// our `String` first so `?` has the right error type to propagate.
    pub fn open(path: &str) -> Result<Self, String> {
        ffmpeg::init().map_err(|e| format!("ffmpeg init failed: {e}"))?;

        let input_ctx = input(&path).map_err(|e| format!("Cannot open {path}: {e}"))?;

        // A container (mp4, mkv…) can hold several streams (video, audio,
        // subtitles). `best(Type::Video)` picks the primary video track.
        // `ok_or(...)` turns the `Option` it returns into a `Result` so `?` can
        // bail out with our message if there's no video stream at all.
        let stream = input_ctx
            .streams()
            .best(Type::Video)
            .ok_or("No video stream found")?;
        let stream_index = stream.index();

        let codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("Codec params: {e}"))?;
        let decoder = codec_ctx
            .decoder()
            .video()
            .map_err(|e| format!("Decoder: {e}"))?;

        let width = decoder.width();
        let height = decoder.height();

        // Estimate total frames from stream duration and frame rate
        let total_frames = {
            let frames = stream.frames() as u64;
            if frames > 0 {
                frames
            } else {
                // Fallback: compute from duration and fps
                let duration_secs =
                    input_ctx.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE);
                let fps = f64::from(stream.avg_frame_rate());
                let estimated = (duration_secs * fps) as u64;
                estimated.max(1)
            }
        };

        // Decoders output frames in formats like YUV420; the GPU wants RGBA.
        // This scaler does that conversion (same in/out size, just a pixel
        // format change). `Flags::BILINEAR` is the interpolation used if it
        // also has to resize — here it doesn't, so it's effectively a colorspace
        // converter.
        let scaler = ScalerContext::get(
            decoder.format(),
            width,
            height,
            ffmpeg::format::Pixel::RGBA,
            width,
            height,
            Flags::BILINEAR,
        )
        .map_err(|e| format!("Scaler: {e}"))?;

        // Still images are 1-frame streams: decode once then hold. GIFs stay
        // animated/looping (not flagged here).
        let still = matches!(
            std::path::Path::new(path)
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase())
                .as_deref(),
            Some("png" | "jpg" | "jpeg" | "bmp" | "webp" | "tiff" | "tif")
        );

        Ok(Self {
            path: path.to_string(),
            input_ctx,
            stream_index,
            decoder,
            scaler,
            width,
            height,
            frame_count: 0,
            total_frames,
            still,
        })
    }

    /// Get the next decoded RGBA frame. Returns None only on unrecoverable error.
    /// Loops back to the start when the file ends.
    pub fn next_frame(&mut self) -> Option<Vec<u8>> {
        // A still image only has one frame — yield it once, then hold (the
        // caller keeps the last uploaded texture when we return None).
        if self.still && self.frame_count >= 1 {
            return None;
        }
        // ffmpeg decoding is a two-step "push packets in, pull frames out" dance:
        // you `send_packet` (compressed data) and then `receive_frame` (raw
        // pixels), but the mapping isn't 1:1 — one packet may yield zero or many
        // frames. So we loop: drain any ready frames, and only feed a new packet
        // when the decoder has none buffered.
        loop {
            // Try to receive already-decoded frames first
            let mut decoded = VideoFrame::empty();
            if self.decoder.receive_frame(&mut decoded).is_ok() {
                self.frame_count += 1;
                return Some(self.scale_frame(&decoded));
            }

            // Feed more packets to the decoder
            match self.next_video_packet() {
                Some(packet) => {
                    if self.decoder.send_packet(&packet).is_err() {
                        continue;
                    }
                    let mut decoded = VideoFrame::empty();
                    if self.decoder.receive_frame(&mut decoded).is_ok() {
                        self.frame_count += 1;
                        return Some(self.scale_frame(&decoded));
                    }
                }
                None => {
                    // EOF — flush decoder then loop the file
                    self.decoder.send_eof().ok();
                    let mut decoded = VideoFrame::empty();
                    if self.decoder.receive_frame(&mut decoded).is_ok() {
                        self.frame_count += 1;
                        return Some(self.scale_frame(&decoded));
                    }
                    // Reopen file to loop
                    if self.reopen().is_err() {
                        return None;
                    }
                }
            }
        }
    }

    /// Returns loop progress as 0.0..1.0
    pub fn progress(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        (self.frame_count % self.total_frames) as f32 / self.total_frames as f32
    }

    /// Run one decoded frame through the RGBA scaler and copy out the bytes.
    /// `data(0)` is plane 0 (the single interleaved RGBA plane); `to_vec()`
    /// copies it into an owned `Vec<u8>` the caller can upload to the GPU.
    fn scale_frame(&mut self, frame: &VideoFrame) -> Vec<u8> {
        let mut rgb_frame = VideoFrame::empty();
        self.scaler.run(frame, &mut rgb_frame).unwrap();
        rgb_frame.data(0).to_vec()
    }

    /// Pull the next packet that belongs to our video stream, skipping packets
    /// from other streams (e.g. audio). Returns `None` at end-of-file.
    fn next_video_packet(&mut self) -> Option<ffmpeg::Packet> {
        for (stream, packet) in self.input_ctx.packets() {
            if stream.index() == self.stream_index {
                return Some(packet);
            }
        }
        None
    }

    /// Re-open the file from the top to loop playback. Simpler (and robust
    /// across all codecs) than seeking back to frame 0; the per-clip cost is
    /// negligible at loop boundaries.
    fn reopen(&mut self) -> Result<(), String> {
        self.input_ctx =
            input(&self.path).map_err(|e| format!("Cannot reopen {}: {e}", self.path))?;

        let stream = self
            .input_ctx
            .streams()
            .best(Type::Video)
            .ok_or("No video stream on reopen")?;

        let codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("Codec params on reopen: {e}"))?;
        self.decoder = codec_ctx
            .decoder()
            .video()
            .map_err(|e| format!("Decoder on reopen: {e}"))?;

        self.frame_count = 0;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{synth_video, write_non_media};

    #[test]
    fn open_reads_dimensions_from_fixture() {
        let (_dir, path) = synth_video(64, 48, 10, 0.5);
        let dec = VideoDecoder::open(path.to_str().unwrap()).expect("open fixture");
        assert_eq!(dec.width, 64);
        assert_eq!(dec.height, 48);
    }

    #[test]
    fn open_errors_on_bogus_path() {
        let err = match VideoDecoder::open("/no/such/file.mp4") {
            Ok(_) => panic!("expected open to fail on a missing file"),
            Err(e) => e,
        };
        assert!(err.contains("Cannot open"), "unexpected error: {err}");
    }

    #[test]
    fn open_errors_on_non_media_file() {
        let (_dir, path) = write_non_media();
        // A text file has no decodable video stream; open must fail rather than
        // hand back a half-built decoder.
        assert!(VideoDecoder::open(path.to_str().unwrap()).is_err());
    }

    #[test]
    fn next_frame_yields_rgba_sized_buffer() {
        let (_dir, path) = synth_video(64, 48, 10, 0.5);
        let mut dec = VideoDecoder::open(path.to_str().unwrap()).expect("open fixture");
        let frame = dec.next_frame().expect("decode one frame");
        // RGBA = 4 bytes per pixel.
        assert_eq!(frame.len(), 64 * 48 * 4);
    }

    #[test]
    fn progress_advances_and_wraps_on_loop() {
        let (_dir, path) = synth_video(64, 48, 10, 0.5); // ~5 frames
        let mut dec = VideoDecoder::open(path.to_str().unwrap()).expect("open fixture");
        assert_eq!(dec.progress(), 0.0, "no frames decoded yet");

        // Decode well past the clip length so it loops at least once.
        let mut seq = Vec::new();
        for _ in 0..30 {
            dec.next_frame().expect("decode frame");
            seq.push(dec.progress());
        }

        // Progress always stays in [0, 1).
        assert!(
            seq.iter().all(|&p| (0.0..1.0).contains(&p)),
            "out of range: {seq:?}"
        );
        // It advanced past zero at some point...
        assert!(
            seq.iter().any(|&p| p > 0.0),
            "progress never advanced: {seq:?}"
        );
        // ...and wrapped back down at a loop boundary (not monotonic).
        assert!(
            seq.windows(2).any(|w| w[1] < w[0]),
            "progress never wrapped (no loop detected): {seq:?}"
        );
    }
}
