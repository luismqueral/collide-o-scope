//! A single compositing layer: one video clip + its own effect chain.
//!
//! The app holds a stack of `Layer`s. Each owns its decoder and a GPU texture
//! it uploads frames into; the renderer composites them bottom-to-top using each
//! layer's `blend_mode` and `opacity`.
//!
//! `crate::` paths refer to other modules in this same binary (the crate root),
//! the way `./` refers to the current package in JS imports.
use std::collections::HashMap;

use crate::automation::Expr;
use crate::effects::EffectUniforms;
use crate::text::{TextAlign, TextFont, TextSource};
use crate::video::VideoDecoder;

/// How a layer is mixed with everything beneath it. Same maths as Photoshop
/// blend modes / the WebGL version this project migrated from.
///
/// `#[derive(...)]` auto-generates trait implementations: `Copy`/`Clone` make
/// it a trivially-copied value (no ownership juggling), `PartialEq` enables
/// `==` comparisons, `Debug` enables `{:?}` printing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendMode {
    Normal,
    Screen,
    Multiply,
    Difference,
}

// An `impl` block hangs methods/constants off a type. These convert the enum
// to the various forms the rest of the app needs: a number for the shader, a
// pretty label for the UI, and a lowercase id for the WebSocket wire format.
impl BlendMode {
    /// Every variant, in UI order — handy for building dropdowns / cycling.
    pub const ALL: &[BlendMode] = &[
        BlendMode::Normal,
        BlendMode::Screen,
        BlendMode::Multiply,
        BlendMode::Difference,
    ];

    /// Numeric code handed to the WGSL shader (which has no enums, only ints).
    pub fn as_u32(self) -> u32 {
        match self {
            BlendMode::Normal => 0,
            BlendMode::Screen => 1,
            BlendMode::Multiply => 2,
            BlendMode::Difference => 3,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            BlendMode::Normal => "Normal",
            BlendMode::Screen => "Screen",
            BlendMode::Multiply => "Multiply",
            BlendMode::Difference => "Difference",
        }
    }

    /// Lowercase id used on the wire (matches the web UI <option> values).
    pub fn as_str(self) -> &'static str {
        match self {
            BlendMode::Normal => "normal",
            BlendMode::Screen => "screen",
            BlendMode::Multiply => "multiply",
            BlendMode::Difference => "difference",
        }
    }
}

/// Maximum frames to decode in one catch-up burst. Prevents a long stall (e.g. a
/// slow VHS tick) from fast-forwarding the footage on the next tick.
const MAX_CATCHUP_FRAMES: u32 = 4;

/// Where a layer's pixels come from. Every variant ultimately feeds the same
/// `upload_frame` → GPU texture → FX-chain path, so the rest of the engine
/// (effects, blend, opacity, transform, automation) is identical regardless of
/// kind. A tagged enum (one field + one tag) replaces the old
/// `Option<VideoDecoder>` + `audio_only: bool` pair, and the compiler now forces
/// every consumer to handle each kind exhaustively.
pub enum LayerSource {
    /// Video **or** still image — the decoder handles both (images set its
    /// `still` flag and hold one frame). This is the only kind that decodes
    /// per-tick footage.
    Clip(VideoDecoder),
    /// A title card: CPU-rasterized glyphs. Rasterizes lazily (only when
    /// `dirty`), then the persisted texture is animated purely by
    /// transform/opacity automation — no per-frame re-rasterize.
    Text(TextSource),
    /// Audio-only clip (no video stream). Skipped in the visual composite; the
    /// mixer thread drives its audio. Was the old `decoder: None` case.
    AudioOnly,
}

/// One clip in the stack, bundling its pixel source, GPU texture, transport
/// state, per-layer effects, and any parameter automations.
pub struct Layer {
    /// Stable identifier assigned by `App::add_layer`, used by the web UI to
    /// track cards across reorders. Survives MoveLayer/RemoveLayer.
    pub id: u64,
    pub filename: String,
    /// Where this layer's pixels come from (clip, text, or audio-only).
    pub source: LayerSource,
    pub texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,
    pub opacity: f32,
    pub blend_mode: BlendMode,
    pub paused: bool,
    pub visible: bool,
    pub effects: EffectUniforms,
    pub width: u32,
    pub height: u32,
    // Transport
    pub speed: f32,             // 0.25..4.0 playback multiplier (1.0 = normal)
    pub fps: f32,               // target decode FPS (e.g. 30.0)
    pub frame_accumulator: f32, // seconds of footage owed, drained by advance()
    // Parameter automation
    pub automations: HashMap<String, Expr>, // param name → compiled expression
    pub automation_errors: HashMap<String, String>, // param name → parse error
    // Per-layer audio (mute/volume/pan); the mixer holds the playing source and
    // is the audible authority — this mirror exists for snapshots/patches.
    pub audio: crate::audio::AudioParams,
}

impl Layer {
    pub fn new(path: &str, device: &wgpu::Device) -> Result<Self, String> {
        // Audio-only clips have no video stream: skip the (failing) decoder open
        // and use a 1×1 placeholder texture. They're filtered out of the visual
        // composite, so this texture is never actually sampled.
        let audio_only = is_audio_file(std::path::Path::new(path));
        let (source, width, height) = if audio_only {
            (LayerSource::AudioOnly, 1u32, 1u32)
        } else {
            let dec = VideoDecoder::open(path)?;
            let (w, h) = (dec.width, dec.height);
            (LayerSource::Clip(dec), w, h)
        };

        let (texture, texture_view) = create_layer_texture(device, width, height);

        // Extract just the filename from the path (for display in the UI).
        // `file_name()` returns an `Option` (a path could end in `..`), so we
        // `map` the success case and `unwrap_or_else` to fall back to the full
        // path if there's no final component. `to_string_lossy` handles paths
        // that aren't valid UTF-8 by substituting replacement characters.
        let filename = std::path::Path::new(path)
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());

        let mut effects = EffectUniforms::default();
        effects.resolution = [width as f32, height as f32];

        Ok(Self {
            id: 0,
            filename,
            source,
            texture,
            texture_view,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            paused: false,
            visible: true,
            effects,
            width,
            height,
            speed: 1.0,
            fps: 30.0,
            frame_accumulator: 0.0,
            automations: HashMap::new(),
            automation_errors: HashMap::new(),
            audio: crate::audio::AudioParams::default(),
        })
    }

    /// Build a title-card layer. The text rasterizes into a fixed `canvas_w ×
    /// canvas_h` RGBA texture (chosen at creation, like a clip's file-derived
    /// dimensions); the existing transform/fit then places and scales it like
    /// any other layer. No file, so the patch persists the text/style instead
    /// of a `filename` (see `patch::TextConfig`).
    #[allow(clippy::too_many_arguments)]
    pub fn new_text(
        text: String,
        font: TextFont,
        size_px: f32,
        color: [f32; 3],
        align: TextAlign,
        canvas_w: u32,
        canvas_h: u32,
        device: &wgpu::Device,
    ) -> Self {
        // Card title for the UI: first non-empty line, capped; else "Text".
        let filename = text
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .map(|l| l.chars().take(24).collect::<String>())
            .unwrap_or_else(|| "Text".to_string());

        let source = LayerSource::Text(TextSource::new(
            text, font, size_px, color, align, canvas_w, canvas_h,
        ));
        let (width, height) = (canvas_w.max(1), canvas_h.max(1));
        let (texture, texture_view) = create_layer_texture(device, width, height);

        let mut effects = EffectUniforms::default();
        effects.resolution = [width as f32, height as f32];

        Self {
            id: 0,
            filename,
            source,
            texture,
            texture_view,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            paused: false,
            visible: true,
            effects,
            width,
            height,
            speed: 1.0,
            fps: 30.0,
            frame_accumulator: 0.0,
            automations: HashMap::new(),
            automation_errors: HashMap::new(),
            audio: crate::audio::AudioParams::default(),
        }
    }

    /// True for audio-only layers (no video stream): skipped in the visual
    /// composite, still mixed for audio. Replaces the old `audio_only` field.
    pub fn is_audio_only(&self) -> bool {
        matches!(self.source, LayerSource::AudioOnly)
    }

    /// Playback position within the source, 0..1. Clips report decoder progress;
    /// text/audio-only have no transport, so they report 0.
    pub fn progress(&self) -> f32 {
        match &self.source {
            LayerSource::Clip(decoder) => decoder.progress(),
            LayerSource::Text(_) | LayerSource::AudioOnly => 0.0,
        }
    }

    /// The text source if this is a title-card layer, else `None`. Lets the
    /// patch + web snapshot read a text layer's content and style.
    pub fn text_source(&self) -> Option<&TextSource> {
        match &self.source {
            LayerSource::Text(t) => Some(t),
            _ => None,
        }
    }

    /// Mutable access to the text source for in-place edits (re-set text/style
    /// and mark `dirty` so `advance` re-rasterizes next tick). `None` for
    /// non-text layers.
    pub fn text_source_mut(&mut self) -> Option<&mut TextSource> {
        match &mut self.source {
            LayerSource::Text(t) => Some(t),
            _ => None,
        }
    }

    /// Advance this layer's footage by `dt` real seconds (scaled by `speed`).
    /// Decodes/skips as many frames as elapsed time dictates, uploading only the
    /// last one (intermediate frames are skipped on screen). This keeps footage on
    /// the same wall-clock as the animated `time` uniform, so they stay in sync even
    /// when a tick runs long (e.g. the blocking VHS readback). Mirrors the export
    /// accumulator in render_export.rs, but capped for live use.
    pub fn advance(&mut self, dt: f32, queue: &wgpu::Queue) {
        // Each source kind decides what (if anything) to upload this tick. We
        // compute the RGBA buffer here, then upload after the match so the
        // `&mut self.source` borrow is released before `upload_frame(&self)`.
        let frame: Option<Vec<u8>> = match &mut self.source {
            // Clip: pull as many frames as elapsed time owes, showing only the
            // last (intermediate frames are skipped on screen). This keeps
            // footage on the same wall-clock as the animated `time` uniform.
            LayerSource::Clip(decoder) => {
                self.frame_accumulator += dt * self.speed;
                let interval = 1.0 / self.fps; // seconds per source frame
                let mut n = (self.frame_accumulator / interval).floor() as u32;
                if n == 0 {
                    return;
                }
                if n > MAX_CATCHUP_FRAMES {
                    // Discard the backlog so a long stall doesn't fast-forward.
                    n = MAX_CATCHUP_FRAMES;
                    self.frame_accumulator = 0.0;
                } else {
                    self.frame_accumulator -= n as f32 * interval;
                }
                let mut last: Option<Vec<u8>> = None;
                for _ in 0..n {
                    if let Some(frame) = decoder.next_frame() {
                        last = Some(frame);
                    }
                }
                last
            }
            // Text: rasterize once when dirty (create/edit), then never again —
            // the texture is the cache and motion comes from automation. Clear
            // the flag so subsequent ticks are no-ops.
            LayerSource::Text(text) => {
                if text.dirty {
                    text.dirty = false;
                    Some(text.rasterize())
                } else {
                    None
                }
            }
            // Audio-only: no video; the mixer thread drives audio. Nothing here.
            LayerSource::AudioOnly => None,
        };

        if let Some(frame) = frame {
            self.upload_frame(queue, &frame);
        }
    }

    /// Copy a decoded RGBA frame from CPU memory up to this layer's GPU texture.
    /// `bytes_per_row = 4 * width` because RGBA8 is 4 bytes per pixel; wgpu needs
    /// the row stride to interpret the flat byte slice as a 2D image.
    pub fn upload_frame(&self, queue: &wgpu::Queue, rgba_data: &[u8]) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * self.width),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Allocate the RGBA GPU texture a layer uploads frames into, plus its default
/// view. Shared by clip and text layers so the descriptor lives in one place.
/// wgpu uses "descriptor" structs (a settings bag) for resource creation.
/// `COPY_DST` = we'll write CPU pixels into it; `TEXTURE_BINDING` = the shader
/// can sample it. `Rgba8UnormSrgb` matches the decoder's / rasterizer's RGBA8
/// output and applies sRGB gamma correctly when sampled. Audio-only layers get
/// a 1×1 of this (never bound — see the render_layers filter).
fn create_layer_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Layer Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // A "view" is the handle shaders actually bind to (it can reinterpret a
    // texture's format/mips); the default view is the whole texture as-is.
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, texture_view)
}

/// Valid media file extensions accepted by the library + drag-and-drop.
pub fn is_supported_media(path: &std::path::Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => matches!(
            ext.to_lowercase().as_str(),
            // video
            "mp4" | "webm" | "mov" | "avi" | "mkv"
            // images (held single frame) + animated gif (loops)
            | "gif" | "png" | "jpg" | "jpeg" | "bmp" | "webp" | "tiff" | "tif"
            // audio-only (music bed / sound source — no video, see is_audio_file)
            | "m4a" | "mp3" | "wav" | "aac" | "ogg" | "opus" | "flac" | "aiff" | "aif"
        ),
        None => false,
    }
}

/// True for audio-only file extensions. These become layers with no video
/// decoder (audio is driven entirely by the mixer thread), so they're skipped
/// in the visual composite while keeping full per-layer audio controls.
pub fn is_audio_file(path: &std::path::Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => matches!(
            ext.to_lowercase().as_str(),
            "m4a" | "mp3" | "wav" | "aac" | "ogg" | "opus" | "flac" | "aiff" | "aif"
        ),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// `BlendMode` maps 1:1 to contiguous shader codes 0..4 and distinct labels/wire ids.
    #[test]
    fn blend_mode_bijection_over_all() {
        eprintln!("layers: BlendMode round-trips to contiguous codes, labels, and wire ids");
        // as_u32 must be the 0..N index into ALL, and the string/label forms
        // must round-trip 1:1 (the wire format + UI depend on this).
        for (i, &mode) in BlendMode::ALL.iter().enumerate() {
            assert_eq!(mode.as_u32() as usize, i, "as_u32 should match ALL index");
        }
        // Codes are distinct and contiguous 0..4.
        let codes: Vec<u32> = BlendMode::ALL.iter().map(|m| m.as_u32()).collect();
        assert_eq!(codes, vec![0, 1, 2, 3]);
        // Labels / wire ids are distinct.
        let labels: Vec<&str> = BlendMode::ALL.iter().map(|m| m.label()).collect();
        let ids: Vec<&str> = BlendMode::ALL.iter().map(|m| m.as_str()).collect();
        assert_eq!(labels, vec!["Normal", "Screen", "Multiply", "Difference"]);
        assert_eq!(ids, vec!["normal", "screen", "multiply", "difference"]);
    }

    /// `is_supported_media` accepts every known video, image, and audio extension.
    #[test]
    fn supported_media_accepts_known_extensions() {
        eprintln!("layers: is_supported_media accepts all known media extensions");
        for ext in [
            "mp4", "webm", "mov", "avi", "mkv", // video
            "gif", "png", "jpg", "jpeg", "bmp", "webp", "tiff", "tif", // images
            "m4a", "mp3", "wav", "aac", "ogg", "opus", "flac", "aiff", "aif", // audio
        ] {
            let p = format!("clip.{ext}");
            assert!(
                is_supported_media(Path::new(&p)),
                "{ext} should be supported"
            );
        }
    }

    /// `is_supported_media` matches extensions regardless of letter case.
    #[test]
    fn supported_media_is_case_insensitive() {
        eprintln!("layers: is_supported_media is case-insensitive on extensions");
        assert!(is_supported_media(Path::new("CLIP.MP4")));
        assert!(is_supported_media(Path::new("Song.Mp3")));
    }

    /// `is_supported_media` rejects unknown and extensionless paths.
    #[test]
    fn supported_media_rejects_unknown_and_extensionless() {
        eprintln!("layers: is_supported_media rejects unknown and extensionless paths");
        assert!(!is_supported_media(Path::new("notes.txt")));
        assert!(!is_supported_media(Path::new("archive.zip")));
        assert!(!is_supported_media(Path::new("README"))); // no extension
        assert!(!is_supported_media(Path::new("")));
    }

    /// `is_audio_file` flags audio extensions while video/image stay non-audio media.
    #[test]
    fn audio_file_detection_splits_from_video() {
        eprintln!("layers: is_audio_file separates audio extensions from video/image media");
        // Audio extensions are audio files...
        for ext in [
            "m4a", "mp3", "wav", "aac", "ogg", "opus", "flac", "aiff", "aif",
        ] {
            let p = format!("track.{ext}");
            assert!(is_audio_file(Path::new(&p)), "{ext} should be audio");
        }
        // ...video/image extensions are not (even though they're supported media).
        for ext in ["mp4", "webm", "mov", "png", "gif"] {
            let p = format!("clip.{ext}");
            assert!(!is_audio_file(Path::new(&p)), "{ext} should NOT be audio");
            assert!(is_supported_media(Path::new(&p)));
        }
        assert!(!is_audio_file(Path::new("noext")));
    }
}
