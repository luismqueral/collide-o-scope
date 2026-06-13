use std::collections::HashMap;

use crate::automation::Expr;
use crate::effects::EffectUniforms;
use crate::video::VideoDecoder;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendMode {
    Normal,
    Screen,
    Multiply,
    Difference,
}

impl BlendMode {
    pub const ALL: &[BlendMode] = &[
        BlendMode::Normal,
        BlendMode::Screen,
        BlendMode::Multiply,
        BlendMode::Difference,
    ];

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

pub struct Layer {
    /// Stable identifier assigned by `App::add_layer`, used by the web UI to
    /// track cards across reorders. Survives MoveLayer/RemoveLayer.
    pub id: u64,
    pub filename: String,
    pub decoder: VideoDecoder,
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
}

impl Layer {
    pub fn new(
        path: &str,
        device: &wgpu::Device,
    ) -> Result<Self, String> {
        let decoder = VideoDecoder::open(path)?;
        let width = decoder.width;
        let height = decoder.height;

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
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Extract just the filename from the path
        let filename = std::path::Path::new(path)
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());

        let mut effects = EffectUniforms::default();
        effects.resolution = [width as f32, height as f32];

        Ok(Self {
            id: 0,
            filename,
            decoder,
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
        })
    }

    /// Advance this layer's footage by `dt` real seconds (scaled by `speed`).
    /// Decodes/skips as many frames as elapsed time dictates, uploading only the
    /// last one (intermediate frames are skipped on screen). This keeps footage on
    /// the same wall-clock as the animated `time` uniform, so they stay in sync even
    /// when a tick runs long (e.g. the blocking VHS readback). Mirrors the export
    /// accumulator in render_export.rs, but capped for live use.
    pub fn advance(&mut self, dt: f32, queue: &wgpu::Queue) {
        self.frame_accumulator += dt * self.speed;
        let interval = 1.0 / self.fps; // seconds per source frame
        let mut n = (self.frame_accumulator / interval).floor() as u32;
        if n == 0 {
            return;
        }
        if n > MAX_CATCHUP_FRAMES {
            // Discard the backlog so a long stall doesn't fast-forward next tick.
            n = MAX_CATCHUP_FRAMES;
            self.frame_accumulator = 0.0;
        } else {
            self.frame_accumulator -= n as f32 * interval;
        }
        // Decode n frames; only the last is shown (skips intermediate frames).
        let mut last: Option<Vec<u8>> = None;
        for _ in 0..n {
            if let Some(frame) = self.decoder.next_frame() {
                last = Some(frame);
            }
        }
        if let Some(frame) = last {
            self.upload_frame(queue, &frame);
        }
    }

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

/// Valid media file extensions accepted by the library + drag-and-drop.
pub fn is_supported_media(path: &std::path::Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => matches!(
            ext.to_lowercase().as_str(),
            // video
            "mp4" | "webm" | "mov" | "avi" | "mkv"
            // images (held single frame) + animated gif (loops)
            | "gif" | "png" | "jpg" | "jpeg" | "bmp" | "webp" | "tiff" | "tif"
        ),
        None => false,
    }
}
