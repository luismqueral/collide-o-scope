# collide-o-scope

A native VDJ (video DJ) tool for live visual performance. Plays and layers video clips with real-time GPU effects. Built as a personal project to learn Rust, migrating from a previous Svelte/WebGL implementation.

## What it does

- Multi-layer video compositing with blend modes (normal, screen, multiply, difference)
- Per-layer and master effects bus with full analog effect suite
- Per-layer transport controls (play/pause, speed, FPS)
- Library browser for managing source clips
- Drag-and-drop support for files and folders

## Effects

**Digital:** pixelate, RGB split, hue shift, saturation, brightness, contrast, posterize, invert

**Analog** (ported from legacy WebGL version): film grain (4 noise algorithms), vignette, color drift (per-frame random chromatic aberration), breathing (zoom/rotation/position micro-distortion)

All effects run as a single combined fragment shader on the GPU — no pipeline switching.

## Stack

- **wgpu** — GPU rendering (Metal on macOS, Vulkan, DX12)
- **winit** — windowing and input
- **ffmpeg-next** — video decoding
- **egui** — immediate-mode UI
- **bytemuck** — zero-cost GPU buffer casting

## Build

Requires system ffmpeg:

```sh
brew install ffmpeg
cargo build
```

## Run

```sh
# Open with a video file (parent folder becomes library)
cargo run -- path/to/clip.mp4

# Open with a folder (browse clips in library panel)
cargo run -- path/to/clips/

# No args — drag and drop files/folders onto the window
cargo run
```

## Controls

| Key | Action |
|-----|--------|
| Space | Pause/resume selected layer |
| F | Toggle fullscreen |
| P / Shift+P | Increase/decrease pixelate |
| G / Shift+G | Increase/decrease RGB split |
| 0 | Reset effects |
| Esc | Quit |

## Status

Early stage / learning project. Things that work:

- [x] Multi-layer compositing with 3-texture ping-pong
- [x] Per-layer effects + master effects bus
- [x] Analog effects (grain, breathing, vignette, color drift)
- [x] Library browser with folder scanning
- [x] Drag-and-drop (files and folders)
- [x] Layer visibility toggle, reordering, blend modes
- [x] Per-layer speed/FPS control

Things I want to add:

- [ ] Thumbnails in layer list and library
- [ ] Drag-to-reorder layers (currently grip handle + delta)
- [ ] MIDI controller mapping
- [ ] Audio reactivity
- [ ] ntsc-rs integration for full VHS signal simulation
- [ ] Shift/morph/glitch effects (from legacy WebGL version)
- [ ] Feedback/trails effect
- [ ] Preset save/load
