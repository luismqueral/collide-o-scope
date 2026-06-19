//! Text (title-card) layer source: CPU glyph rasterization to an RGBA buffer.
//!
//! A text layer produces pixels exactly like a video layer does — a flat RGBA
//! byte buffer uploaded to a GPU texture via `Layer::upload_frame`. Once those
//! pixels exist, the text inherits the entire per-layer effect chain, blend
//! modes, opacity, transform, and automation for free (the FX shader only ever
//! sees pixels, never glyphs). So all this module does is turn a string + style
//! into that buffer; everything downstream is shared with the video path.
//!
//! Rasterization is lazy: it runs only when `dirty` is set (on create/edit).
//! The GPU texture is the cache — motion (fades, scrolls, beat-pulses) comes
//! from transform/opacity automation on the persisted texture, not from
//! re-rasterizing every frame.

use ab_glyph::{point, Font, FontRef, Glyph, GlyphId, PxScale, ScaleFont};

/// Clamp bounds for the font size in pixels: keeps the rasterizer from being
/// asked for a sub-pixel nothing or an absurdly large allocation.
const MIN_SIZE_PX: f32 = 4.0;
const MAX_SIZE_PX: f32 = 512.0;

/// Which vendored font a text layer draws with. IBM Plex ships under the SIL
/// Open Font License and is already embedded for egui; we reuse the same bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextFont {
    Sans,
    Mono,
}

impl TextFont {
    /// The embedded OTF bytes for this font (compiled into the binary, so the
    /// build stays self-contained — same approach as the egui font setup).
    pub fn bytes(self) -> &'static [u8] {
        match self {
            TextFont::Sans => include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/fonts/IBMPlexSans-Regular.otf"
            )),
            TextFont::Mono => include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/fonts/IBMPlexMono-Regular.otf"
            )),
        }
    }

    /// Wire/patch identifier for this font.
    pub fn as_str(self) -> &'static str {
        match self {
            TextFont::Sans => "sans",
            TextFont::Mono => "mono",
        }
    }

    /// Parse a wire/patch string back into a font, defaulting to Sans.
    pub fn from_wire(s: &str) -> Self {
        match s {
            "mono" => TextFont::Mono,
            _ => TextFont::Sans,
        }
    }
}

/// Horizontal alignment of each line within the canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlign {
    Left,
    Center,
    Right,
}

impl TextAlign {
    pub fn as_str(self) -> &'static str {
        match self {
            TextAlign::Left => "left",
            TextAlign::Center => "center",
            TextAlign::Right => "right",
        }
    }

    /// Parse a wire/patch string back into an alignment, defaulting to Left.
    pub fn from_wire(s: &str) -> Self {
        match s {
            "center" => TextAlign::Center,
            "right" => TextAlign::Right,
            _ => TextAlign::Left,
        }
    }
}

/// All the state a text layer needs to produce its pixels.
pub struct TextSource {
    pub text: String,
    pub font: TextFont,
    pub size_px: f32,
    /// sRGB 0..1, matching how the rest of the app stores colors (hex on the wire).
    pub color: [f32; 3],
    pub align: TextAlign,
    /// Fixed rasterization canvas. Chosen once at creation (the layer's GPU
    /// texture is allocated to match); the existing transform/fit then places
    /// and scales it like any other layer.
    pub canvas_w: u32,
    pub canvas_h: u32,
    /// Set on create/edit; cleared by the layer after the next rasterize+upload.
    pub dirty: bool,
}

impl TextSource {
    pub fn new(
        text: String,
        font: TextFont,
        size_px: f32,
        color: [f32; 3],
        align: TextAlign,
        canvas_w: u32,
        canvas_h: u32,
    ) -> Self {
        Self {
            text,
            font,
            size_px: size_px.clamp(MIN_SIZE_PX, MAX_SIZE_PX),
            color,
            align,
            canvas_w: canvas_w.max(1),
            canvas_h: canvas_h.max(1),
            dirty: true,
        }
    }

    /// Render the current text/style into a fresh `canvas_w × canvas_h` RGBA
    /// buffer: a transparent background (alpha 0) with antialiased glyphs in
    /// `color` whose alpha is the glyph coverage. Straight (non-premultiplied)
    /// alpha — the composite shader blends using `overlay.a` as the weight, so
    /// transparent pixels contribute nothing and lower layers show through.
    pub fn rasterize(&self) -> Vec<u8> {
        let w = self.canvas_w as usize;
        let h = self.canvas_h as usize;
        let mut buf = vec![0u8; w * h * 4]; // transparent background

        // Parse the embedded font. The bytes are 'static, so the FontRef may
        // borrow them for the life of this call.
        let font = match FontRef::try_from_slice(self.font.bytes()) {
            Ok(f) => f,
            // Unparseable font → blank buffer. Shouldn't happen for the vendored
            // OTFs, but we never want a panic in the per-frame path.
            Err(_) => return buf,
        };

        let size = self.size_px.clamp(MIN_SIZE_PX, MAX_SIZE_PX);
        let scale = PxScale::from(size);
        let scaled = font.as_scaled(scale);
        let ascent = scaled.ascent();
        let line_height = ascent - scaled.descent() + scaled.line_gap();

        let [cr, cg, cb] = self.color;
        let color = [
            (cr.clamp(0.0, 1.0) * 255.0) as u8,
            (cg.clamp(0.0, 1.0) * 255.0) as u8,
            (cb.clamp(0.0, 1.0) * 255.0) as u8,
        ];

        // A small inset so glyphs don't kiss the canvas edge.
        let pad = size * 0.25;

        // Vertically center the whole block of lines within the canvas so a
        // title card sits in the middle of the frame (horizontal centering is
        // per-line, via `align`). Without this the text was pinned to the top
        // (baseline = pad + ascent), which read as "stuck at the top" on a full
        // frame. Clamp the block top to `pad` so a block taller than the canvas
        // still starts on-canvas instead of spilling off the top.
        let n_lines = self.text.split('\n').count().max(1) as f32;
        let block_height = n_lines * line_height;
        let block_top = (((self.canvas_h as f32) - block_height) * 0.5).max(pad);

        // Each newline-separated line gets its own baseline, laid out top-down.
        for (line_idx, line) in self.text.split('\n').enumerate() {
            // First pass: total advance width of the line (for alignment),
            // summing per-glyph advance plus kerning between neighbors.
            let line_width = line_advance(&font, &scaled, line);

            let start_x = match self.align {
                TextAlign::Left => pad,
                TextAlign::Center => ((self.canvas_w as f32) - line_width) * 0.5,
                TextAlign::Right => (self.canvas_w as f32) - line_width - pad,
            };
            let baseline_y = block_top + ascent + line_idx as f32 * line_height;

            // Second pass: place + draw each glyph along the baseline.
            let mut pen_x = start_x;
            let mut prev: Option<GlyphId> = None;
            for ch in line.chars() {
                let id = font.glyph_id(ch);
                if let Some(p) = prev {
                    pen_x += scaled.kern(p, id);
                }
                let glyph: Glyph = id.with_scale_and_position(scale, point(pen_x, baseline_y));
                pen_x += scaled.h_advance(id);
                prev = Some(id);

                if let Some(outlined) = font.outline_glyph(glyph) {
                    let bounds = outlined.px_bounds();
                    outlined.draw(|gx, gy, coverage| {
                        let px = bounds.min.x as i32 + gx as i32;
                        let py = bounds.min.y as i32 + gy as i32;
                        if px < 0 || py < 0 || px as usize >= w || py as usize >= h {
                            return; // clipped by the canvas
                        }
                        let idx = (py as usize * w + px as usize) * 4;
                        // Single-color text: keep the strongest coverage where
                        // glyphs overlap, avoiding dark seams from over-blending.
                        let a = (coverage.clamp(0.0, 1.0) * 255.0) as u8;
                        if a >= buf[idx + 3] {
                            buf[idx] = color[0];
                            buf[idx + 1] = color[1];
                            buf[idx + 2] = color[2];
                            buf[idx + 3] = a;
                        }
                    });
                }
            }
        }

        buf
    }
}

/// Total pen advance for one line: per-glyph horizontal advance plus the
/// kerning adjustment between each adjacent pair.
fn line_advance<F: Font, SF: ScaleFont<F>>(font: &FontRef, scaled: &SF, line: &str) -> f32 {
    let mut width = 0.0f32;
    let mut prev: Option<GlyphId> = None;
    for ch in line.chars() {
        let id = font.glyph_id(ch);
        if let Some(p) = prev {
            width += scaled.kern(p, id);
        }
        width += scaled.h_advance(id);
        prev = Some(id);
    }
    width
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rasterize always returns a buffer sized exactly to the canvas (RGBA).
    #[test]
    fn rasterize_returns_canvas_sized_rgba() {
        eprintln!("text: rasterize returns a canvas_w*canvas_h*4 RGBA buffer");
        let src = TextSource::new(
            "Hi".into(),
            TextFont::Sans,
            32.0,
            [1.0, 1.0, 1.0],
            TextAlign::Left,
            128,
            64,
        );
        let buf = src.rasterize();
        assert_eq!(buf.len(), 128 * 64 * 4);
    }

    /// Empty text leaves the canvas fully transparent (every alpha byte 0).
    #[test]
    fn empty_text_is_fully_transparent() {
        eprintln!("text: empty string rasterizes to a fully transparent canvas");
        let src = TextSource::new(
            String::new(),
            TextFont::Sans,
            32.0,
            [1.0, 1.0, 1.0],
            TextAlign::Left,
            64,
            32,
        );
        let buf = src.rasterize();
        assert!(
            buf.iter().skip(3).step_by(4).all(|&a| a == 0),
            "expected all-transparent canvas for empty text"
        );
    }

    /// Non-empty text marks some pixels opaque, and opaque pixels carry the
    /// requested color (here pure red).
    #[test]
    fn drawn_text_uses_requested_color() {
        eprintln!("text: drawn glyphs are opaque and carry the requested color");
        let src = TextSource::new(
            "A".into(),
            TextFont::Mono,
            48.0,
            [1.0, 0.0, 0.0],
            TextAlign::Center,
            96,
            96,
        );
        let buf = src.rasterize();
        // At least one fully-or-partly opaque pixel exists.
        let lit: Vec<usize> = (0..buf.len() / 4).filter(|&i| buf[i * 4 + 3] > 0).collect();
        assert!(!lit.is_empty(), "expected some glyph coverage");
        // Every lit pixel is red (R=255, G=0, B=0) — the fill color.
        for &i in &lit {
            assert_eq!(buf[i * 4], 255, "red channel");
            assert_eq!(buf[i * 4 + 1], 0, "green channel");
            assert_eq!(buf[i * 4 + 2], 0, "blue channel");
        }
    }

    /// A single line of text is vertically centered in the canvas: the lit
    /// region's midpoint sits near the canvas midline (not pinned to the top).
    /// Guards the `block_top` centering math in `rasterize`.
    #[test]
    fn single_line_is_vertically_centered() {
        eprintln!("text: a single line's lit region is centered on the canvas midline");
        let cw = 256u32;
        let ch = 256u32;
        let src = TextSource::new(
            "Ag".into(), // ascender + descender for a representative glyph span
            TextFont::Sans,
            48.0,
            [1.0, 1.0, 1.0],
            TextAlign::Center,
            cw,
            ch,
        );
        let buf = src.rasterize();
        let w = cw as usize;
        let h = ch as usize;
        let (mut miny, mut maxy) = (h, 0usize);
        for y in 0..h {
            for x in 0..w {
                if buf[(y * w + x) * 4 + 3] > 0 {
                    miny = miny.min(y);
                    maxy = maxy.max(y);
                }
            }
        }
        assert!(maxy >= miny, "expected some lit pixels");
        let lit_mid = (miny + maxy) as f32 * 0.5;
        let canvas_mid = ch as f32 * 0.5;
        // Allow a generous slack: glyph bounds aren't symmetric about the
        // baseline (ascenders/descenders differ), but the block should land
        // near the middle — nowhere near the old top-anchored position.
        assert!(
            (lit_mid - canvas_mid).abs() < ch as f32 * 0.15,
            "lit midpoint {lit_mid} should be near canvas midline {canvas_mid}"
        );
    }

    /// Font + alignment identifiers round-trip through their wire strings.
    #[test]
    fn font_and_align_round_trip_wire_strings() {
        eprintln!("text: TextFont/TextAlign round-trip through wire strings");
        for f in [TextFont::Sans, TextFont::Mono] {
            assert_eq!(TextFont::from_wire(f.as_str()), f);
        }
        for a in [TextAlign::Left, TextAlign::Center, TextAlign::Right] {
            assert_eq!(TextAlign::from_wire(a.as_str()), a);
        }
        // Unknown strings fall back to the safe defaults.
        assert_eq!(TextFont::from_wire("???"), TextFont::Sans);
        assert_eq!(TextAlign::from_wire("???"), TextAlign::Left);
    }
}
