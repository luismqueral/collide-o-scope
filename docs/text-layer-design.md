# Text Layer (Title Cards) — Design Investigation

How to add text/title cards as a first-class layer, so they inherit the full effect
chain, blend modes, opacity, transform, and automation that video/image layers already
have. Grounded in how layers actually source pixels today.

---

## 1. Why text must be rasterized

The per-layer effects run as a **GPU fragment shader that samples a texture** — it only
ever sees RGBA *pixels*. Warp, glitch, RGB-split, and feedback all work by reading
neighboring pixels; the shader has no concept of "text" or vector glyphs. So to put
effects on text, the text has to become pixels first.

The alternative — drawing glyphs procedurally in the shader (SDF/MSDF text) — is possible
but much harder, needs a glyph atlas anyway, and is overkill here. **CPU-rasterize-to-
texture is the clear path.**

## 2. The seam: a layer's only contract is `upload_frame`

A video layer's entire contract with the engine is one method:

```
upload_frame(&self, queue, rgba_data: &[u8])   // src/layers/mod.rs:221
```

The decoder hands it a flat RGBA byte buffer; that's written to the layer's GPU texture
(`src/layers/mod.rs:84`); then the per-layer FX chain runs on that texture. **A text layer
just needs to produce the same RGBA buffer and call the same `upload_frame`.** Downstream
it is indistinguishable from a video layer — it inherits every effect, blend mode,
opacity, transform, and automation with **zero shader changes**.

## 3. Today there are only *two* sources, not three

Images are **not** a separate source. A `Layer` holds `decoder: Option<VideoDecoder>`
(`src/layers/mod.rs:83`), and `VideoDecoder` already handles stills internally via its
`still` flag (`src/video/decoder.rs`) — png/jpg decode one frame and hold. So today:

- `Some(VideoDecoder)` → video **or** image
- `None` → audio-only (1×1 placeholder texture, skipped in the composite)

Everything funnels through `advance()` (`src/layers/mod.rs:187`), which pulls from the
decoder and calls `upload_frame`. That is the seam text plugs into.

## 4. The split: turn the optional decoder into a source enum

Replace the optional decoder field with a tagged source (a discriminated union — one
field, one tag per source kind):

```
// replaces `decoder: Option<VideoDecoder>` on Layer
enum LayerSource {
    Clip(VideoDecoder),   // video + image (still flag), exactly as today
    Text(TextSource),     // { text, font, size, color, align, canvas, dirty, cached_rgba }
    AudioOnly,            // the old `None` case, now explicit
}
```

`advance()` becomes a `match`:

- **`Clip`** → today's body verbatim (frame accumulator, fps, catch-up, `upload_frame`).
- **`Text`** → if `dirty`, rasterize → `upload_frame`, clear the flag; else do nothing.
  The texture persists, and **motion comes from transform/opacity automation, not from
  re-rasterizing** — so fades, scrolls, and beat-pulses are free and deterministic.
- **`AudioOnly`** → early return, as now.

Adding a future source = one more variant + one more arm; the compiler forces exhaustive
handling everywhere. (A `trait FrameSource` is the alternative, but text isn't
"frame-paced" like video, so the enum fits cleaner and is easier to debug.)

## 5. The one genuinely new decision: canvas size

A video/image layer gets its texture dimensions from the file (`src/layers/mod.rs:117`).
Text has **no inherent resolution**, so a text layer must *choose* one — e.g. match the
output canvas, or a fixed high-res — rasterize into it, then let the existing transform/fit
place and scale it. This is the only new concept the source model introduces.

## 6. Rasterization tooling

A Rust glyph rasterizer fills the RGBA buffer:

- `ab_glyph` / `fontdue` — simple single-font draw; easiest, great for basic title cards.
- `cosmic-text` — full layout (wrapping, alignment, multi-font/emoji); more power, more
  weight. Better for richer typography on social cards.

Fonts are already vendored and embedded for egui (`assets/fonts/`, `include_bytes!`), so a
text layer reuses that approach rather than reading from the system.

## 7. How invasive — the honest map

**Untouched (the win):** texture, `texture_view`, `upload_frame`, the entire FX shader,
blend modes, opacity, transform/fit, automation. Text inherits all of it for free.

**Localized to a few match arms** — every spot that currently reaches into `.decoder`:

- `advance()` — `src/layers/mod.rs:187`
- export frame-pull — `src/render_export.rs:624`
- progress readout — `src/main.rs:1204` (text reports 0 / N/A)
- reload-clip swap — `src/main.rs:304`

**The real work is at two edges** (the same edges *any* layer feature touches):

- **Patch serde** (`src/patch/mod.rs`) — a text layer has no `filename`, so it serializes
  its text/style/canvas instead and reconstructs on load (the "recreate sources on load"
  path, `src/main.rs:1090`).
- **Web contract** (`src/web/state.rs`) — new `WebAction`s to create/edit a text layer +
  `LayerSnapshot` fields to round-trip it to the panel.

**Verdict:** shallow in the hot rendering path (a few match arms, zero shader changes),
with two predictable edges — patch persistence and the web actions. The pixels are the
easy part; the plumbing to *create and save* a text layer is where the effort sits.

## 8. Overlap with preview-plugins

A title/intro card is itself a "share with context" feature, so it overlaps the
preview-plugins idea:

- An **engine/export-baked** title card needs this rasterize-to-texture path (so it lands
  in the MP4).
- A **browser-rendered** preview skin could instead own the title text directly (HTML text
  is trivial there) — no rasterization needed, but it only exists in that preview view.

So the answer to "where do preview-plugins render" partly decides whether title cards need
section 1–7 at all, or whether simple cards can live in the preview layer.
