# collide-o-scope

A native VDJ (video DJ) tool for live visual performance. Plays and layers video
clips with real-time GPU effects, driven from a browser-based control panel.
Built as a personal project to learn Rust, migrating from a previous Svelte/WebGL
implementation.

> Heads up: this is a personal project built primarily for my own use. It's a
> learning exercise and almost certainly buggy — shared as-is, not a polished
> product.

## What it does

- **Multi-layer compositing** — stack clips with blend modes (Normal, Screen,
  Multiply, Difference), per-layer opacity, speed, FPS, and transport.
- **Two-tier effects** — a per-layer effect chain *and* a master output bus, all
  running as a single GPU fragment shader (no pipeline switching).
- **Per-layer fit mode** — Stretch / Fit (letterbox) / Fill (crop) so a portrait
  clip on a landscape canvas no longer has to squash.
- **Analog/VHS simulation** — film grain, vignette, color drift, breathing, plus a
  full NTSC/VHS signal pass (head switching, tracking noise, snow, chroma loss…).
- **Web control panel** — all parameters live in the browser (a transposed parameter
  *matrix* by default, or the legacy "classic" panel via `--classic`); the wgpu window
  just shows the output.
- **Parameter automation** — bind any continuous param to a live math expression
  (oscillators, deterministic noise, beat-synced `sin(beat*tau)`…), evaluated every
  frame so the live preview and the exported video match exactly.
- **Tap tempo** — tap (or type) a BPM so beat-synced automations lock to the music.
- **Patches** — save/load the whole state (master + every layer + VHS) as YAML.
- **Offline render** — export the current composition to MP4, live or fully headless.
- **Library browser** — thumbnails + hover-scrub previews; drag-and-drop files/folders.

## Architecture

Two halves share state and run concurrently:

1. **Render engine** (`winit` + `wgpu`) — owns the window, decodes video, runs the
   effect/composite shaders on the GPU, and drives frame timing (~30 FPS).
2. **Web control panel** (`axum` + `tokio` WebSocket on `127.0.0.1:3030`) — the
   browser sends `WebAction` commands; the render loop drains them each frame,
   evaluates any parameter automations, and broadcasts an `AppSnapshot` back so the
   UI stays in sync.

```
src/
├── main.rs            — winit event loop, app state, frame timing, action handling, CLI
├── renderer/state.rs  — wgpu setup, pipelines (effects + composite), texture upload
├── video/decoder.rs   — ffmpeg frame extraction, YUV→RGBA, looping
├── effects/params.rs  — EffectUniforms (the GPU uniform buffer; 17 × vec4 = 272 bytes)
├── automation.rs      — compiled math-expression params (fasteval), evaluated per frame
├── layers/mod.rs      — Layer struct, BlendMode, supported-format check
├── ntsc/              — ntsc-rs VHS post-processing (CPU)
├── patch/             — save/load state as YAML + in-window YAML editor
├── render_export.rs   — offline MP4 export (live + headless)
├── input/keyboard.rs  — key → action mapping
├── web/
│   ├── server.rs      — axum routes + WebSocket
│   └── state.rs       — WebState, AppSnapshot, WebAction (the browser↔engine contract)
└── shaders/
    ├── fullscreen.wgsl — vertex shader (fullscreen triangle, no VBO)
    └── effects.wgsl    — combined effect + composite fragment shader
```

**Frame pipeline (per layer, then master):**

```
decode → per-layer FX (effects.wgsl) → composite onto accumulator (blend mode)
       → master FX → NTSC/VHS (CPU, half-res live / full-res export) → display
```

Compositing uses three textures (accumulator/output, current-layer, temp) plus a
feedback texture for datamosh-style trails.

### Effects

**Master bus** — pixelate, RGB split, hue, saturation, brightness, contrast,
posterize, invert, film grain (4 algorithms, optional color), vignette, color
drift, breathing (scale/rotation/position).

**Per-layer** — color (hue/sat/bright/contrast), digital (pixelate/RGB
split/posterize/invert), warp (wave, swirl, bulge), chroma key (threshold,
smoothness, spill, optional solid-color background fill), pixel-shift/glitch
(slice, block, jitter, chroma shift, datamosh), feedback (persistence, zoom,
rotate, luma key, chroma, additive), and transform (position, scale, **fit mode**).

**NTSC/VHS** — tape speed, chroma loss, edge wave, head switching, tracking noise,
snow, composite/luma/chroma noise, luma smear, composite sharpening.

## Build

Requires system ffmpeg (the `ffmpeg-next` version must match it; currently v8):

```sh
brew install ffmpeg
cargo build
```

## Run

Launching opens the render window **and** auto-opens the control panel at
`http://127.0.0.1:3030`.

```sh
# Folder → becomes the library
cargo run -- path/to/clips/

# Single file → also uses its parent folder as the library
cargo run -- path/to/clip.mp4

# No args → uses ./library/ if present; otherwise drag-and-drop onto the window
cargo run
```

Library scanning is **single-level** (not recursive). Supported inputs: `mp4`,
`webm`, `mov`, `avi`, `mkv`, and images `gif` (animated loops), `png`, `jpg`,
`jpeg`, `bmp`, `webp`, `tiff`/`tif`.

## Patches

Save/load the full state as YAML. Patches live in a `patches/` folder next to the
library folder (i.e. the project root when running `cargo run -- library/`), falling
back to `./patches`. Manage them from the control panel or the in-window YAML editor.

## Automation

Any continuous parameter can be driven by a math expression instead of a fixed
value. Expressions are compiled once ([`fasteval`](https://crates.io/crates/fasteval))
and evaluated every frame against the same clock the exporter uses, so live preview
and rendered output stay identical. Available vocabulary (see `src/automation.rs`):

- **Time / tempo** — `t` (elapsed seconds), `beat` (beats since the last tap
  downbeat), `bpm`, plus `pi` / `tau`.
- **Oscillators** (1 Hz, output −1..1) — `tri`, `saw`, `square`, `pulse`.
- **Procedural** (deterministic value-noise, not true randomness) — `fbm`, `hold`,
  `wiggle(freq, amp)`, `noise(seed)`.
- **Shaping** — `clamp`, `lerp`, `smoothstep`, plus fasteval built-ins (`sin`, `cos`,
  `abs`, `min`, `max`, `floor`, `sqrt`, `^` for power, …).

Example: `0.5 + 0.5*sin(beat*tau)` pulses a 0..1 param once per beat.

## Headless render

Render a saved patch straight to MP4 with no window or web server:

```sh
cargo run -- render --patch patches/my-patch.yaml --library library/ \
    [--out out.mp4] [--duration 10] [--fps 30] [--res 1280x720]
```

Defaults: `--out experiments/headless-output/out.mp4`, `1280x720`, `30` fps, `10` s.

## Testing

Tests run with [cargo-nextest](https://nexte.st/) (`curl -LsSf https://get.nexte.st/latest/mac | tar zxf - -C ~/.cargo/bin`):

```sh
cargo nextest run                    # fast hardware-free unit + ffmpeg integration tests
cargo nextest run --run-ignored all  # also runs the GPU export E2E (needs a real GPU)
cargo nextest run --no-capture       # also prints each test's one-line description as it runs
cargo nextest list                   # list every test (grouped by module) without running
```

GPU-dependent tests are marked `#[ignore]` so the default run (and CI) stays
hardware-free; ffmpeg tests synthesize their own tiny fixtures via the system
`ffmpeg`. There are no doctests yet (those would run via `cargo test --doc`).

Every test carries a `///` doc comment saying what it verifies, and prints the
same as a one-line description (visible with `--no-capture`). The nextest
`default` profile uses `status-level = "all"`, so the output lists every test —
pass or skip — as a checklist of what's covered.

## Controls (render window)

| Key | Action |
|-----|--------|
| Space | Pause/resume |
| F | Toggle fullscreen |
| P / Shift+P | Increase / decrease pixelate |
| G / Shift+G | Increase / decrease RGB split |
| 0 | Reset master effects |
| Ctrl+E | Toggle in-window YAML editor |
| Ctrl+S | Save patch |
| Ctrl+O | Load patch |
| Esc | Quit |

Most performance control happens in the browser panel; these are quick in-window
shortcuts.

## Stack

- **wgpu** — GPU rendering (Metal on macOS, Vulkan, DX12)
- **winit** — windowing and input
- **ffmpeg-next** — video decoding
- **ntsc-rs** — analog/VHS signal simulation
- **axum** + **tokio** — web control panel + WebSocket
- **egui** — in-window UI (YAML editor)
- **serde** / **serde_yaml** — patch persistence
- **bytemuck** — zero-cost GPU uniform casting

## Known gotchas

- `ffmpeg-next` must match the system ffmpeg version (v8 for ffmpeg 8.1).
- `EffectUniforms` is tightly packed at 272 bytes (17 × vec4); the Rust struct and
  the WGSL `Uniforms` must stay field-for-field identical or bytemuck `Pod` fails.
- Static panel assets (`static/*.js`, `*.css`, `*.html`) are embedded into the binary
  at compile time via `include_str!`, so editing them requires a `cargo build` +
  restart + hard browser reload — there is no separate web dev server.
- A param is only automatable if it has an arm in `EffectUniforms::set_by_name`
  (`effects/params.rs`); that match is the single write path the per-frame automation
  eval uses, so a missing key silently no-ops.
- NTSC/VHS runs on the CPU, so it processes at half resolution in the live preview
  (full resolution on export) to keep frame time down.
- The `block` crate emits future-incompat warnings (upstream, harmless).
