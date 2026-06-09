# VDJ Project — Conversation Context

## Project Overview

Building a custom live VDJ (video DJ) performance tool for playing clips from a friend's band. Currently exists as a browser-based WebGL/Svelte implementation. The goal is to migrate toward a native Rust architecture for better performance.

---

## Effects Library

**ntsc-rs** (`github.com/ntsc-rs/ntsc-rs`) — a serious Rust implementation of NTSC/VHS analog video signal simulation. CPU-bound, sequential signal processing. The anchor of the effects pipeline.

---

## Architecture Decision: Native Rust

**Why not browser:**
- ntsc-rs is CPU-heavy and needs native threads — WASM constrains this severely
- Multi-layer compositing with per-layer effects benefits from direct GPU access
- Web Audio API has inherent latency; native audio (cpal) supports real-time priority
- WASM has memory limits; multiple video streams need headroom

**Chosen stack:**
- **winit** — native window management
- **wgpu** — cross-platform GPU (Vulkan/Metal/DX12); used for warping, morphing, compositing
- **ntsc-rs** — integrated directly for VHS/analog effects
- **cpal** — cross-platform audio with real-time thread priority
- **egui** — immediate-mode control UI (side panel in same window)

**Output surface:** Native winit + wgpu fullscreen window on the performance display. No browser in the signal path.

---

## Rust Development Workflow

**VS Code setup:**
- `rust-analyzer` extension — autocomplete, inline types, error highlighting
- `CodeLLDB` — debugging
- Claude Code alongside for agentic assistance

**Dev loop:**
```
edit → cargo build → run → see result → repeat
```
- `cargo-watch` for auto-rebuild on save
- Shaders (WGSL) can hot-reload at runtime without recompiling Rust
- No CSS; visual output is whatever wgpu draws

**Project file organization** mirrors JS/Python — files map to functionality, folders map to modules. `mod.rs` declares a module, `use` brings things into scope.

**The borrow checker:** Rust's main learning curve. Compiler enforces memory safety at compile time. Errors are detailed and usually self-explaining. If it compiles, it almost certainly works correctly.

---

## LLM-Assisted Rust Development

**The good:** The compiler acts as a guardrail — it immediately catches bad AI output. Tighter feedback loop than Python/JS where bad code silently ships.

**The risks:**
- LLMs trained mostly on high-level languages; wgpu/winit have less training coverage
- Common AI mistakes: wrong lifetime annotations, excessive `.clone()`, non-idiomatic ownership patterns
- Agentic sessions can drift architecturally if not reviewed

**Mitigation for a personal project:**
- This is not released software — correctness matters, polish doesn't
- Let Claude Code run, verify it compiles and performs, move on
- Go in scoped chunks (one feature at a time), not "build the whole app"

---

## Tooling & MCPs

**Key MCPs for Rust development:**
- **Context7** — injects live, version-specific crate docs into prompts; prevents hallucinated APIs. Install first. Check `context7.com` for wgpu/winit coverage; submit via "Add Library" if missing.
- **Rust MCP Server** — bridges Claude Code with local Rust environment; runs `cargo build`, `cargo clippy`, tests in the loop
- **rust-docs-mcp-server** — Rust-specific doc access

**CLAUDE.md file** (in project root): persistent context Claude Code reads every session. Include:
- What the project is
- Key crates and why
- Architectural conventions
- Known solved gotchas

**Other tips:**
- Pin crate versions in `Cargo.toml` once working (wgpu moves fast)
- Feed `sotrh.github.io/learn-wgpu` sections directly into prompts for wgpu questions
- `DeepWiki` (`deepwiki.com/gfx-rs/wgpu`) has wgpu indexed and navigable
- Run `cargo clippy` after AI-generated code to catch non-idiomatic patterns

---

## Next Step

Review the existing Svelte/WebGL implementation at `~/projects/collide-o-scope/projects/live` to understand current feature set, then build a step-by-step migration plan to the Rust architecture.

**To proceed:** Either paste key source files here, or open a Claude Code session pointed at the project directory for a full read-through.
