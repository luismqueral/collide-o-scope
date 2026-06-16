# Collide-o-Scope

Native Rust VDJ (video DJ) performance tool. Plays and layers video clips with
real-time GPU effects, driven from a browser-based control panel. The wgpu window
just shows the output; all performance control happens in the browser. Built as a
personal project to learn Rust, migrating from a previous Svelte/WebGL implementation.

## Stack

- **winit** — window management (fullscreen, input events)
- **wgpu** — GPU rendering via Metal (macOS) / Vulkan / DX12
- **ffmpeg-next** — video decoding (requires system `brew install ffmpeg`)
- **ntsc-rs** — analog/VHS signal simulation (CPU)
- **axum** + **tokio** — web control panel + WebSocket on `127.0.0.1:3030`
- **egui** — in-window UI (YAML patch editor)
- **fasteval** — compiled math expressions for parameter automation
- **serde** / **serde_yaml** — patch persistence
- **bytemuck** — zero-cost casting for GPU uniform buffers

Future: midir (MIDI controller), cpal (audio).

## Module layout

```
src/
├── main.rs            — winit event loop, app state, frame timing, action handling, CLI
├── renderer/state.rs  — wgpu setup, pipelines (effects + composite), texture upload
├── video/decoder.rs   — ffmpeg frame extraction, YUV→RGBA, looping
├── effects/params.rs  — EffectUniforms (the GPU uniform buffer; 17 × vec4 = 272 bytes)
├── automation.rs      — compiled math-expression params (fasteval), evaluated per frame
├── layers/mod.rs      — Layer struct, BlendMode, supported-format check
├── ntsc/              — ntsc-rs VHS post-processing (CPU)
├── patch/             — save/load state as YAML + in-window egui YAML editor
├── render_export.rs   — offline MP4 export (live + headless)
├── input/keyboard.rs  — key → action mapping (quick in-window shortcuts only)
├── web/
│   ├── server.rs       — axum routes + WebSocket
│   ├── state.rs        — WebState, AppSnapshot, WebAction (browser↔engine contract)
│   └── static_files.rs — static/*.{html,js,css} embedded at compile time (include_str!)
└── shaders/
    ├── fullscreen.wgsl — vertex shader (fullscreen triangle, no VBO)
    └── effects.wgsl    — combined effect + composite fragment shader

static/                — vanilla JS/CSS/HTML panel (matrix.js, classic, style.css). No build step.
```

## Build & run

Requires system ffmpeg (the `ffmpeg-next` version must match it; currently v8):

```sh
brew install ffmpeg
cargo build

# Folder → becomes the library; single file → uses its parent folder
cargo run -- videos/
cargo run -- videos/some-file.mp4

# Legacy "classic" control panel instead of the default matrix view
cargo run -- --classic videos/

# Headless render of a saved patch straight to MP4 (no window / no web server)
cargo run -- render --patch patches/my-patch.yaml --library videos/ \
    [--out out.mp4] [--duration 10] [--fps 30] [--res 1280x720]
```

Launching opens the render window **and** auto-opens the panel at `http://127.0.0.1:3030`.

## Frame pipeline

```
decode → per-layer FX (effects.wgsl) → composite onto accumulator (blend mode)
       → master FX → NTSC/VHS (CPU, half-res live / full-res export) → display
```

Each frame the render loop also: drains queued `WebAction`s, evaluates any parameter
automations into `EffectUniforms` (via `set_by_name`), uploads the uniform buffer,
renders, and broadcasts an `AppSnapshot` back to the browser so the UI stays in sync.

## Keyboard controls (render window — quick shortcuts only)

- Space — pause/resume
- F — toggle fullscreen
- P / Shift+P — increase / decrease pixelate
- G / Shift+G — increase / decrease RGB split
- 0 — reset master effects
- Ctrl+E — toggle in-window YAML editor
- Ctrl+S / Ctrl+O — save / load patch
- Escape — quit

Most performance control happens in the browser panel.

## Architecture notes

- **Browser↔engine contract lives in `web/state.rs`:** the browser sends `WebAction`
  (a `#[serde(tag = "action")]` enum) over the WebSocket → the render loop drains the
  `state.actions` queue each frame → applies them → pushes an `AppSnapshot` back.
- **Two-tier effects:** a per-layer effect chain *and* a master output bus, all in a
  single combined GPU fragment shader (uniform-driven, no pipeline switching).
- **Automation single write path:** a param is only automatable if it has an arm in
  `EffectUniforms::set_by_name` (`effects/params.rs`). That match is the one path the
  per-frame automation eval uses (both the layer and master loops in `main.rs`), so a
  missing key silently no-ops. Non-numeric params (invert, enums, colors) are excluded.
- **Live/offline parity:** automations are evaluated against the same clock the exporter
  uses, so the live preview and the rendered MP4 match exactly.
- Fullscreen triangle drawn with 3 vertices — UVs computed from `vertex_index`.
- Frame timing advances on a held clock (~30 FPS); the decoder loops automatically at EOF.

## Known gotchas

- **No web build step:** `static/*.{js,css,html}` are embedded into the binary at compile
  time via `include_str!` (`web/static_files.rs`), so editing them requires `cargo build`
  + restart + a hard browser reload. There is no separate web dev server. (Pure-Rust
  changes only need a restart.)
- `EffectUniforms` is tightly packed at 272 bytes (17 × vec4); the Rust struct and the
  WGSL `Uniforms` must stay field-for-field identical or bytemuck `Pod` fails.
- `ffmpeg-next` must match the system ffmpeg version (v8 for ffmpeg 8.1).
- NTSC/VHS runs on the CPU, so it processes at half resolution in the live preview (full
  resolution on export) to keep frame time down.
- The `block` crate emits future-incompat warnings (upstream issue, harmless).
