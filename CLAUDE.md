# Collide-o-Scope

Native Rust VDJ (video DJ) performance tool. Plays video clips with real-time GPU effects for live visual performance.

## Stack

- **winit** — window management (fullscreen, input events)
- **wgpu** — GPU rendering via Metal (macOS) / Vulkan / DX12
- **ffmpeg-next** — video decoding (requires system `brew install ffmpeg`)
- **bytemuck** — zero-cost casting for GPU uniform buffers

Future: ntsc-rs (analog VHS effects), midir (MIDI controller), cpal (audio)

## Module layout

```
src/
├── main.rs           — winit event loop, frame timing, app state
├── renderer/state.rs — wgpu setup, pipelines, texture upload, render()
├── video/decoder.rs  — ffmpeg frame extraction, YUV→RGBA, looping
├── effects/params.rs — EffectUniforms struct, parameter adjustments
├── input/keyboard.rs — key→action mapping
└── shaders/
    ├── fullscreen.wgsl — vertex shader (fullscreen triangle, no VBO)
    └── effects.wgsl    — combined pixelate + RGB split fragment shader
```

## Build & run

```sh
cargo build
cargo run -- videos/some-file.mp4
```

## Keyboard controls

- P / Shift+P — increase / decrease pixelate
- G / Shift+G — increase / decrease RGB split
- 0 — reset all effects
- Space — pause/resume
- F — toggle fullscreen
- Escape — quit

## Architecture notes

- Single combined effect shader (uniform-driven, no pipeline switching)
- Fullscreen triangle drawn with 3 vertices — UVs computed from vertex_index
- Video decoded synchronously (no threading needed for single 720p stream at 30fps)
- Frame timing: only advances frame when 33ms elapsed since last render
- Decoder loops automatically at EOF

## Known gotchas

- `ffmpeg-next` version must match system ffmpeg (currently v8 for ffmpeg 8.1)
- wgpu v25 removed the trace_path arg from `request_device`
- The `block` crate emits future-incompat warnings (upstream issue, harmless)
