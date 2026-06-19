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
    total_frames: u64, // estimated frame count, used for progress() + loop window
    /// True for still images (png/jpg/etc): decode one frame then hold it,
    /// instead of re-opening the file every frame at EOF.
    still: bool,
    /// Loop window as fractions of the clip `0.0..1.0`. Default `0.0..1.0` =
    /// loop the whole file (today's behavior). Set via `set_loop` so a clip can
    /// cycle just a slice of itself (a punchy VJ loop). Stored as f64 to match
    /// the frame-count maths in `start_frame`/`end_frame`.
    loop_start: f64,
    loop_end: f64,
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
            // Default window = whole clip; `set_loop` narrows it later.
            loop_start: 0.0,
            loop_end: 1.0,
        })
    }

    /// Get the next decoded RGBA frame. Returns None only on unrecoverable error.
    /// Loops back to the window's start (the loop-in point, or frame 0 when the
    /// clip isn't trimmed) when it reaches the loop-out point or true EOF.
    pub fn next_frame(&mut self) -> Option<Vec<u8>> {
        // A still image only has one frame — yield it once, then hold (the
        // caller keeps the last uploaded texture when we return None).
        if self.still && self.frame_count >= 1 {
            return None;
        }
        // Trimmed clips loop early, at the loop-out point, rather than running to
        // EOF. Untrimmed clips (the default 0..1 window) skip this and rely on the
        // true-EOF branch below — byte-for-byte the same behavior as before, which
        // matters because `total_frames` is only an *estimate*.
        if self.is_trimmed() && self.frame_count >= self.end_frame() {
            if self.loop_to_start().is_err() {
                return None;
            }
        }
        match self.pull_frame() {
            Some(frame) => {
                self.frame_count += 1;
                Some(self.scale_frame(&frame))
            }
            None => {
                // True EOF — loop back to the window's start and try once more.
                if self.loop_to_start().is_err() {
                    return None;
                }
                let frame = self.pull_frame()?;
                self.frame_count += 1;
                Some(self.scale_frame(&frame))
            }
        }
    }

    /// Pull the next raw decoded frame, or `None` at end-of-file. This is the
    /// ffmpeg "push packets in, pull frames out" dance: `send_packet` feeds
    /// compressed data and `receive_frame` yields raw pixels, but the mapping
    /// isn't 1:1 — one packet may yield zero or many frames. So we loop: drain
    /// any ready frames first, and only feed a new packet when none are buffered.
    /// Unlike `next_frame`, this neither scales, counts, nor loops — callers do
    /// that (so `loop_to_start` can reuse it to fast-forward by discarding).
    fn pull_frame(&mut self) -> Option<VideoFrame> {
        loop {
            // Try to receive an already-decoded frame first.
            let mut decoded = VideoFrame::empty();
            if self.decoder.receive_frame(&mut decoded).is_ok() {
                return Some(decoded);
            }

            // Feed more packets to the decoder.
            match self.next_video_packet() {
                Some(packet) => {
                    if self.decoder.send_packet(&packet).is_err() {
                        continue;
                    }
                    let mut decoded = VideoFrame::empty();
                    if self.decoder.receive_frame(&mut decoded).is_ok() {
                        return Some(decoded);
                    }
                }
                None => {
                    // EOF — flush the decoder for any final buffered frame.
                    self.decoder.send_eof().ok();
                    let mut decoded = VideoFrame::empty();
                    if self.decoder.receive_frame(&mut decoded).is_ok() {
                        return Some(decoded);
                    }
                    return None; // truly drained
                }
            }
        }
    }

    /// Reopen the file and fast-forward to the loop-in point so the next pull
    /// yields the window's first frame. Reopen (vs. keyframe seek) matches
    /// `reopen()`'s codec-robustness rationale; the discard cost is bounded by
    /// the clip length — fine for the short slices this feature targets.
    fn loop_to_start(&mut self) -> Result<(), String> {
        self.reopen()?; // resets frame_count = 0
        let start = self.start_frame();
        if start > 0 {
            for _ in 0..start {
                // Clip shorter than the requested start (estimate was high):
                // stop discarding and just play from wherever we landed.
                if self.pull_frame().is_none() {
                    break;
                }
            }
            self.frame_count = start;
        }
        Ok(())
    }

    /// True when the loop window is narrower than the whole clip. Gates the
    /// early-loop path so untrimmed clips behave exactly as they did before.
    fn is_trimmed(&self) -> bool {
        self.loop_start > 0.0 || self.loop_end < 1.0
    }

    /// Frame index of the loop-in point (the first frame of the window).
    fn start_frame(&self) -> u64 {
        (self.loop_start * self.total_frames as f64).round() as u64
    }

    /// Frame index of the loop-out point (looping happens once `frame_count`
    /// reaches this). Clamped to keep at least one frame in the window and to
    /// never exceed the clip length.
    fn end_frame(&self) -> u64 {
        let end = (self.loop_end * self.total_frames as f64).round() as u64;
        let lo = self.start_frame() + 1;
        let hi = self.total_frames.max(lo);
        end.clamp(lo, hi)
    }

    /// Restrict looping to the window `[start, end]` (fractions of the clip,
    /// `0.0..1.0`). Clamps both to range and guarantees the out-point sits at
    /// least ~one frame after the in-point, so an inverted/degenerate pair still
    /// yields a valid (minimal) window instead of an empty one.
    pub fn set_loop(&mut self, start: f32, end: f32) {
        // One frame as a fraction of the clip; the smallest sensible window.
        let min_gap = if self.total_frames > 0 {
            1.0 / self.total_frames as f64
        } else {
            0.01
        };
        // Keep enough headroom above `start` that `start + min_gap <= 1.0`.
        let start = (start.clamp(0.0, 1.0) as f64).min(1.0 - min_gap);
        let end = (end.clamp(0.0, 1.0) as f64).clamp(start + min_gap, 1.0);
        self.loop_start = start;
        self.loop_end = end;
    }

    /// Returns loop progress as 0.0..1.0 *within the active window*, so the UI
    /// bar still sweeps 0→1 across a trimmed slice. Untrimmed clips reduce to the
    /// original `frame_count % total_frames` behavior.
    pub fn progress(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        let start = self.start_frame();
        let span = self.end_frame().saturating_sub(start);
        if span == 0 {
            return 0.0;
        }
        // frame_count can momentarily sit below `start` right after a reopen;
        // saturating_sub floors the position at 0 rather than wrapping huge.
        let pos = self.frame_count.saturating_sub(start);
        (pos % span) as f32 / span as f32
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

    /// Opening a known fixture reports the clip's true pixel dimensions.
    #[test]
    fn open_reads_dimensions_from_fixture() {
        eprintln!("video: VideoDecoder::open reports the fixture's width and height");
        let (_dir, path) = synth_video(64, 48, 10, 0.5);
        let dec = VideoDecoder::open(path.to_str().unwrap()).expect("open fixture");
        assert_eq!(dec.width, 64);
        assert_eq!(dec.height, 48);
    }

    /// Opening a path that doesn't exist returns an Err naming the failure
    /// ("Cannot open"), rather than panicking or half-initializing a decoder.
    #[test]
    fn open_errors_on_bogus_path() {
        eprintln!("video: VideoDecoder::open errors on a missing file path");
        let err = match VideoDecoder::open("/no/such/file.mp4") {
            Ok(_) => panic!("expected open to fail on a missing file"),
            Err(e) => e,
        };
        assert!(err.contains("Cannot open"), "unexpected error: {err}");
    }

    /// Opening a plain text file (no decodable video stream) returns an Err
    /// instead of a half-built decoder.
    #[test]
    fn open_errors_on_non_media_file() {
        eprintln!("video: VideoDecoder::open errors on a non-media file");
        let (_dir, path) = write_non_media();
        // A text file has no decodable video stream; open must fail rather than
        // hand back a half-built decoder.
        assert!(VideoDecoder::open(path.to_str().unwrap()).is_err());
    }

    /// Decoding one frame yields a buffer of exactly width × height × 4 bytes
    /// (RGBA), confirming the scaler output size.
    #[test]
    fn next_frame_yields_rgba_sized_buffer() {
        eprintln!("video: next_frame returns an RGBA buffer sized to the frame");
        let (_dir, path) = synth_video(64, 48, 10, 0.5);
        let mut dec = VideoDecoder::open(path.to_str().unwrap()).expect("open fixture");
        let frame = dec.next_frame().expect("decode one frame");
        // RGBA = 4 bytes per pixel.
        assert_eq!(frame.len(), 64 * 48 * 4);
    }

    /// progress() starts at 0, stays within [0, 1), advances as frames decode,
    /// and wraps back down at a loop boundary.
    #[test]
    fn progress_advances_and_wraps_on_loop() {
        eprintln!("video: progress stays in [0,1), advances, and wraps on loop");
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

    /// A freshly opened clip is *not* trimmed: the loop window spans the whole
    /// file (start frame 0, end frame = total). This guarantees the default
    /// path is byte-for-byte the pre-trim behavior.
    #[test]
    fn default_window_is_whole_clip() {
        eprintln!("video: a new decoder loops the whole clip (untrimmed) by default");
        let (_dir, path) = synth_video(64, 48, 10, 1.0); // ~10 frames
        let dec = VideoDecoder::open(path.to_str().unwrap()).expect("open fixture");
        assert!(!dec.is_trimmed(), "default window should not be trimmed");
        assert_eq!(dec.start_frame(), 0, "default in-point is frame 0");
        assert_eq!(
            dec.end_frame(),
            dec.total_frames,
            "default out-point is the last frame"
        );
    }

    /// With a trimmed window (second half of the clip), playback loops *within*
    /// that window: after the first wrap, `frame_count` never falls back below
    /// the in-point, and `progress` stays in [0,1).
    #[test]
    fn trimmed_window_loops_within_bounds() {
        eprintln!("video: a trimmed loop window cycles only its slice, not the whole clip");
        let (_dir, path) = synth_video(64, 48, 10, 1.0); // ~10 frames
        let mut dec = VideoDecoder::open(path.to_str().unwrap()).expect("open fixture");
        dec.set_loop(0.5, 1.0);
        let start = dec.start_frame();
        assert!(start > 0, "0.5 fraction should land past frame 0");

        // Decode well past the out-point so the window loops at least once.
        let mut counts = Vec::new();
        let mut progs = Vec::new();
        for _ in 0..40 {
            dec.next_frame().expect("decode frame");
            counts.push(dec.frame_count);
            progs.push(dec.progress());
        }

        // Progress stays in [0, 1).
        assert!(
            progs.iter().all(|&p| (0.0..1.0).contains(&p)),
            "out of range: {progs:?}"
        );
        // A wrap happened (count decreased = it looped rather than running on).
        let first_wrap = counts
            .windows(2)
            .position(|w| w[1] < w[0])
            .unwrap_or_else(|| panic!("never wrapped: {counts:?}"));
        // After wrapping, playback stays inside the window (never below the
        // in-point) — proof we loop the slice, not the whole clip.
        assert!(
            counts[first_wrap + 1..].iter().all(|&c| c >= start),
            "left the loop window after wrap (start={start}): {counts:?}"
        );
    }

    /// `set_loop` clamps out-of-range input to [0,1] and corrects an inverted
    /// pair (end <= start) into a valid window holding at least one frame.
    #[test]
    fn set_loop_clamps_and_orders() {
        eprintln!("video: set_loop clamps to [0,1] and keeps end strictly after start");
        let (_dir, path) = synth_video(64, 48, 10, 1.0); // ~10 frames
        let mut dec = VideoDecoder::open(path.to_str().unwrap()).expect("open fixture");

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
}
