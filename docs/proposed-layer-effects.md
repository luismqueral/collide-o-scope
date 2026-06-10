# Proposed Layer Effects: Warping, Pixel Shifting, Chroma Key

A design investigation into new per-layer effects, focused on the three families
you flagged: **warping**, **pixel shifting**, and **chroma keying**. Grounded in how
the current pipeline actually works, with concrete parameter proposals.

---

## 1. How the effect pipeline works today (so we know what's cheap)

There is **one** effects shader, `src/shaders/effects.wgsl`, driven by **one** uniform
struct, `EffectUniforms` (`src/effects/params.rs`). The renderer runs it twice:

- **Per layer** — `Renderer::render_layers` (`src/renderer/state.rs:351`) uploads each
  layer's `layer.effects` and draws the layer's video texture through the effects
  pipeline into a scratch texture, then composites that onto the accumulator with the
  layer's blend mode (`src/shaders/composite.wgsl`).
- **Master** — `Renderer::render_master_effects` runs the *same* shader once over the
  final composite.

**Consequence #1 — adding a field to `EffectUniforms` gives you the effect at BOTH
levels for free.** The Rust/WGSL plumbing is shared; the only difference is which
sliders the web UI chooses to render per-layer vs. in the master column. This is the
single most important fact for scoping: a warp/shift/key added to the struct is
immediately usable as a per-layer effect *and* a master effect.

**Consequence #2 — warps are nearly free.** A UV warp just changes where we sample the
texture, exactly like the existing `apply_breathing()` (`effects.wgsl:130`) and pixelate
(`effects.wgsl:236`), which mutate `sample_uv` *before* `textureSample`. Wave, swirl,
bulge, and noise-displacement all follow that same pattern — a few ALU ops per pixel, no
extra passes, no extra textures.

**Consequence #3 — chroma key fits the compositor as-is.** `composite.wgsl` already mixes
by `uniforms.opacity * overlay.a` and writes `max(base.a, overlay.a * opacity)`. So if the
effects shader writes `a < 1` where a pixel matches the key color, the keyed pixels show
the layer beneath with no compositor changes — across *all* blend modes.

### The wiring path for any new per-layer effect

Adding one effect touches these points (use an existing param like `posterize` as a
template — grep finds every site):

1. `src/effects/params.rs` — new field(s) in `EffectUniforms` + its `Default`.
2. `src/shaders/effects.wgsl` — mirror the field(s) in the `Uniforms` struct **in the
   same order/packing**, then add the effect logic.
3. `src/web/state.rs` — new field(s) in `LayerSnapshot` (+ `EffectsSnapshot` if you also
   want a master slider) and the `apply_param` / `apply_to_uniforms` arms.
4. `src/main.rs` — `WebAction::SetLayerParam` match arm with a `clamp` (`main.rs:225`),
   and the `LayerSnapshot` builder in `push_web_state` (`main.rs:438`).
5. `src/patch/mod.rs` — `LayerPatch` field + default + `from`/`apply_to_uniforms` +
   serialize + `param_meta` so it round-trips through saved patches **and** offline render
   export (export replays a captured `PatchState`).
6. `static/index.html` (master rows only) / `static/app.js` `createLayerCard` (per-layer
   rows) + `static/style.css`.

### Uniform packing constraint (read before adding fields)

`EffectUniforms` is `#[repr(C)]`, 16-byte aligned, currently **96 bytes = 6 × vec4**, and
the WGSL `Uniforms` struct must match field-for-field. The last vec4 is:

```
// vec4 #6
color_drift: f32,
_pad: [f32; 3],   // ← exactly 3 free f32 slots right now
```

So there are **3 spare floats** before you must add a 7th `vec4` (and shrink `_pad`).
The three families below need far more than 3 params, so we'll be extending the struct —
that's fine, just keep Rust and WGSL byte-identical (this is the #1 source of garbage-
output bugs in wgpu). Booleans travel as `f32` (0.0/1.0), matching the existing `invert`
convention.

A suggested layout once all three families land (illustrative — implement incrementally):

```
// vec4 #7  warp:   wave_amp, wave_freq, wave_speed, wave_axis
// vec4 #8  warp:   swirl_angle, swirl_radius, bulge_strength, bulge_radius
// vec4 #9  shift:  slice_intensity, slice_height, slice_prob, slice_speed
// vec4 #10 shift:  block_size, block_intensity, block_prob, _pad
// vec4 #11 chroma: chroma_enable, chroma_threshold, chroma_smoothness, chroma_spill
// vec4 #12 chroma: chroma_color.r, chroma_color.g, chroma_color.b, _pad
```

---

## 2. Warping

UV-displacement effects. All mutate `sample_uv` near the top of `fs_main` (after
breathing, before/around pixelate), reusing `uniforms.time` for animation and the
existing `perlin_noise()` helper for organic motion. Cheap and very "VJ-friendly".

### 2a. Wave / ripple  *(recommended first)*

Sinusoidal displacement — the classic underwater/heat-haze wobble.

| Param | Range | Default | Meaning |
|---|---|---|---|
| `wave_amp` | 0 .. 0.10 | 0 | displacement amplitude in UV units (0.1 ≈ 10% of frame) |
| `wave_freq` | 0 .. 50 | 8 | wave cycles across the frame |
| `wave_speed` | 0 .. 10 | 1 | scroll speed (multiplies `time`) |
| `wave_axis` | 0/1/2 | 0 | 0 = horizontal, 1 = vertical, 2 = both (select, not a slider) |

```wgsl
if uniforms.wave_amp > 0.0 {
    let t = uniforms.time * uniforms.wave_speed;
    if uniforms.wave_axis != 1.0 { sample_uv.x += sin(uv.y * uniforms.wave_freq + t) * uniforms.wave_amp; }
    if uniforms.wave_axis != 0.0 { sample_uv.y += sin(uv.x * uniforms.wave_freq + t) * uniforms.wave_amp; }
}
```

### 2b. Swirl / twirl  *(recommended first)*

Rotation around the center that falls off with radius — a vortex.

| Param | Range | Default | Meaning |
|---|---|---|---|
| `swirl_angle` | -720 .. 720 | 0 | max rotation (degrees) at the center |
| `swirl_radius` | 0 .. 1 | 0.5 | radius of influence (UV; 0.5 ≈ half-frame) |

```wgsl
if abs(uniforms.swirl_angle) > 0.1 {
    let c = sample_uv - 0.5;
    let d = length(c);
    let falloff = smoothstep(uniforms.swirl_radius, 0.0, d);
    let a = uniforms.swirl_angle * 0.01745 * falloff;
    let cs = cos(a); let sn = sin(a);
    sample_uv = vec2f(c.x*cs - c.y*sn, c.x*sn + c.y*cs) + 0.5;
}
```

### 2c. Bulge / pinch  *(recommended first)*

Radial zoom: positive bulges out (fisheye), negative pinches in.

| Param | Range | Default | Meaning |
|---|---|---|---|
| `bulge_strength` | -1 .. 1 | 0 | signed magnitude (+ bulge / − pinch) |
| `bulge_radius` | 0 .. 1 | 0.5 | extent of the lens |

```wgsl
if abs(uniforms.bulge_strength) > 0.001 {
    let c = sample_uv - 0.5;
    let d = length(c) / uniforms.bulge_radius;
    let scale = 1.0 + uniforms.bulge_strength * (1.0 - clamp(d, 0.0, 1.0));
    sample_uv = c / scale + 0.5;
}
```

### 2d. Turbulence (noise displacement)  *(stretch)*

Organic, smoky distortion driven by the existing `perlin_noise()`.

| Param | Range | Default | Meaning |
|---|---|---|---|
| `turb_amount` | 0 .. 0.10 | 0 | displacement amplitude (UV) |
| `turb_scale` | 1 .. 20 | 4 | noise frequency |
| `turb_speed` | 0 .. 10 | 1 | evolution speed |

> **Edge behavior (all warps):** the sampler uses default address mode (clamp-to-edge),
> so UVs pushed past `[0,1]` smear the border pixels. That reads fine for subtle amounts;
> for heavy warps consider a `mirror-repeat` sampler variant if the smear looks bad.

---

## 3. Pixel shifting / glitch

Quantize UV into bands or blocks and jump them with a hash keyed on `floor(time * speed)`
so the pattern reseeds in discrete steps (the same trick the grain uses,
`effects.wgsl:95`). These are *displacement* glitches — cheap, single-pass.

### 3a. Slice shift (scanline-band displacement)  *(recommended first)*

Horizontal bands jump sideways — VHS tracking-tear / signal-break look.

| Param | Range | Default | Meaning |
|---|---|---|---|
| `slice_intensity` | 0 .. 1 | 0 | max horizontal shift (fraction of width) |
| `slice_height` | 1 .. 128 | 16 | band thickness in pixels |
| `slice_prob` | 0 .. 1 | 0.3 | fraction of bands that shift each step |
| `slice_speed` | 0 .. 30 | 8 | reseed rate (steps/sec) |

```wgsl
if uniforms.slice_intensity > 0.0 {
    let seed = floor(uniforms.time * uniforms.slice_speed);
    let band = floor(uv.y * uniforms.resolution.y / uniforms.slice_height);
    let r = hash(vec2f(band, seed));
    if r > 1.0 - uniforms.slice_prob {
        sample_uv.x += (hash(vec2f(band, seed + 7.0)) - 0.5) * uniforms.slice_intensity;
    }
}
```

### 3b. Block mosh (rectangular block displacement)  *(stretch)*

2D block grid, random per-block offset — datamosh-adjacent.

| Param | Range | Default | Meaning |
|---|---|---|---|
| `block_size` | 4 .. 128 | 32 | block edge in pixels |
| `block_intensity` | 0 .. 1 | 0 | max offset (fraction of frame) |
| `block_prob` | 0 .. 1 | 0.2 | fraction of blocks displaced |

### 3c. Extend `rgb_split` to 2D instead of a new effect  *(recommended)*

We already have per-layer `rgb_split` (horizontal-only chromatic shift,
`effects.wgsl:253`). Rather than add a separate "channel displace" effect, the cheapest win
is to generalize it: add a `rgb_split_y` companion and optionally a `rgb_split_angle`, so
the split can run vertically/diagonally and animate. Reuses the existing slider + handler.

| Param | Range | Default | Meaning |
|---|---|---|---|
| `rgb_split_y` | 0 .. 30 | 0 | vertical channel offset (px), pairs with existing `rgb_split` |

### 3d. Pixel sort — *out of scope (noted for completeness)*

True pixel sorting orders pixels along each scanline by luminance. That's inherently
**multi-pass / compute** (a fragment shader can't sort its row in one pass), so it doesn't
fit the current single-pass `effects.wgsl` model. Park it; revisit only if we add a compute
pipeline.

---

## 4. Chroma key

Make a key color transparent so lower layers show through. Fits the existing compositor
with **no pipeline change** — the effects pass just needs to write `color.a`.

| Param | Range | Default | Meaning |
|---|---|---|---|
| `chroma_enable` | bool | off | master switch (0/1 as f32) |
| `chroma_color` | RGB | (0,1,0) | key color — **color picker**, not a slider |
| `chroma_threshold` | 0 .. 1 | 0.4 | how close to the key color counts as "keyed" |
| `chroma_smoothness` | 0 .. 1 | 0.1 | soft edge / feather width past the threshold |
| `chroma_spill` | 0 .. 1 | 0 | suppress residual key tint on fringes |

Applied at the very end of `fs_main`, modifying alpha (and optionally desaturating spill):

```wgsl
if uniforms.chroma_enable > 0.5 {
    // distance is more stable in chroma than raw RGB; HSL hue/sat works with our helpers
    let key_hsl = rgb_to_hsl(uniforms.chroma_color);
    let px_hsl  = rgb_to_hsl(rgb);
    var dh = abs(px_hsl.x - key_hsl.x); dh = min(dh, 1.0 - dh);      // hue wraps
    let dist = length(vec2f(dh * 2.0, px_hsl.y - key_hsl.y));        // hue + sat distance
    let a = smoothstep(uniforms.chroma_threshold,
                       uniforms.chroma_threshold + uniforms.chroma_smoothness + 0.001,
                       dist);
    color.a = color.a * a;
    // optional spill: pull toward grey where we're near the key hue
    rgb = mix(vec3f(dot(rgb, vec3f(0.299,0.587,0.114))), rgb, mix(1.0, a, uniforms.chroma_spill));
}
return vec4f(clamp(rgb, vec3f(0.0), vec3f(1.0)), color.a);
```

### Chroma-key gotchas (important)

- **Only reveals layers *beneath*.** In `render_layers`, the bottom-most visible layer
  (`i == 0`) is copied straight to the accumulator and never goes through the compositor
  (`state.rs:436`), so keying it just turns keyed areas **black**. Keying is meaningful on
  upper layers. Worth a one-line hint in the UI.
- **Color space.** Layer textures are `Rgba8UnormSrgb`, so the sampler returns
  **linear** RGB in-shader. The key color arrives from the UI as sRGB 0–255 — convert it
  to linear before comparing (or do the comparison in HSL as sketched, which is more
  forgiving). Mismatched spaces = a key that "almost" works.
- **HSL vs YCbCr.** The HSL helpers already in the shader are good enough to start.
  Professional keyers use YCbCr (chroma-plane distance); if green/blue screens look noisy,
  upgrading the metric to YCbCr is the natural follow-up.
- **Premultiplied alpha.** The compositor mixes by `overlay.a` and never divides, so
  straight (non-premultiplied) alpha is correct — no extra handling needed.

---

## 5. Recommendation & phasing

1. **Phase 1 — Warps (highest value / lowest risk).** Wave, swirl, bulge. Pure UV math,
   no new passes, instantly useful live, and proves out the "extend the struct → get it
   per-layer + master" path end-to-end. ~6 new floats (extends to vec4 #7–#8).
2. **Phase 2 — Chroma key.** Single best "new capability" (real layering, not just look).
   No compositor change. Needs a **color-picker UI control** — the only param here that
   isn't a slider/toggle, so it's a small frontend addition.
3. **Phase 3 — Pixel shifting.** Slice shift first (best look-per-effort), then 2D
   `rgb_split`. Block mosh as a stretch. Skip pixel sort (needs compute).

### Cross-cutting UI note

Per-layer effect rows are built in `createLayerCard` (`static/app.js`), grouped into
COLOR / DIGITAL fx-groups (the recent "Randomize sliders" button already iterates
`input[type=range]` in a group — these new sliders get randomization for free). A natural
home: a new **WARP** group and a **KEY** group per layer, plus optional master-column
sliders for the warps. The chroma color picker is the one control that needs a new input
type; everything else is the existing slider/toggle/select pattern.
