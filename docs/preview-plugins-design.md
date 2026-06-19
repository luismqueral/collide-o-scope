# Preview-Plugins / Widgets / Phone-Shareable Recording — Design Investigation

How to build alternate "preview" views — interactive controls and live visualizers —
that stack with the video output into phone-native shareable clips, recorded with less
ceremony than OBS. This captures a design walkthrough; nothing here is built yet.

---

## 1. The architectural crux

The render engine (winit + wgpu) owns the window **and the video pixels**. The browser
panel (axum + tokio WebSocket on `127.0.0.1:3030`) already receives a full `AppSnapshot`
~30×/sec — every effect value, automation, `bpm`, `beat`, `meter`, hue, layer state
(`src/web/state.rs:27`) — and sends `WebAction`s back to drive the engine
(`src/web/state.rs:348`).

So the browser has the complete live **state**, but **not the video pixels**. That single
fact drives every option below. Anything that only needs *state* (a moving dial, a meter,
a value readout) is trivial in the browser today. Anything that needs the *video* in the
browser requires shipping pixels across the socket.

## 2. The vision: modular widgets, not one monolithic "preview app"

Instead of one bespoke preview application per idea, build **one widget runtime** and many
small widgets. A widget:

- reads `AppSnapshot` and renders to its own DOM/canvas each tick;
- is optionally **bound to params** — read-only, write-only, or both.

"Visualizer" (read-only, e.g. a moving meter) and "controller" (interactive, e.g. a VST-style
dial that emits `SetParam`/`SetAutomation`) are the **same shape**; the binding decides. A
dial can do both: reflect automation when idle, take over when grabbed.

New experiment = new widget file + one registry entry. This is the iteration loop, and the
reason for a runtime rather than N hand-built views ("there will be a LOT of experiments").

### Management model (names worth keeping)

- **Widget (a type):** smallest unit. Contract = `render(snapshot, el)` + `bindings`
  (snapshot paths it reads; `WebAction`s it writes).
- **Registry:** a `{ typeName: factory }` map. Adding an experiment = register one factory.
- **Instance:** a placed, configured widget — `{ type, config, position }`. Many instances
  of few types (three dials → three different params).
- **Stage:** a saved *view* — ordered instances + layout (video region size, video
  above/below). A stage is a view *over* any patch, so it is its own artifact, not folded
  into the patch YAML.
- **Record button (nav, global):** fires the record handshake (see §5). Not per-widget.

### Two render hazards (already bitten this panel before)

1. **Render must be idempotent.** The snapshot fires ~30×/sec; a widget that rebuilds DOM
   each tick flickers and eats interaction. Update values in place, never recreate nodes.
2. **Interactive widgets must not clobber the value being dragged.** A both-ways dial must
   suppress the incoming snapshot while grabbed (an `isInteracting` guard) or it fights the
   user and appears to "freeze." Bake this into the widget contract.

## 3. Aspect ratio: "1:1 stacked" is NOT a standard vertical ratio

Two equal 1:1 squares stacked = **1080×2160 = 1:2** (0.50) — *taller/narrower* than the
phone-native targets:

| Target | Pixels | Aspect |
|---|---|---|
| 9:16 (Reels/TikTok/Shorts) | 1080×1920 | 0.5625 |
| 4:5 (IG feed portrait) | 1080×1350 | 0.80 |
| two 1:1 stacked | 1080×2160 | 0.50 |

So 1:1 + 1:1 *overshoots* 9:16 (side bars / crop when posted). To land on 9:16 with a square
video the regions must be **unequal** — e.g. video 1080×1080 + widgets 1080×840 = 1080×1920.

**Recommendation:** make the record stage a fixed **1080×1920 (9:16)** target; the video
region is a chosen height (default 1080 square), widgets take the remainder. "Video above or
below" just flips their Y order — *the only layout customization*, as specified.

## 4. Two ways to produce the stacked clip

### Option A — single canvas (WYSIWYG, needs the frame-pipe)

Compose video + widgets onto **one** `<canvas>` = the record stage, and `captureStream()` +
`MediaRecorder` records exactly that. The stage *is* the final frame. But getting live video
onto that canvas needs the **frame-pipe**: GPU readback (`readback_composite` /
`readback_half`, `src/renderer/state.rs:1016`/`1024`) → downscale → JPEG/MJPEG → WebSocket →
draw on canvas. Lossy, laggy, CPU-hungry — but sync and WYSIWYG are free because there is one
stream.

### Option B — record separately, stitch after (easier to build)

The engine **already** exports video+audio to MP4 (`src/render_export.rs`, the `render`
subcommand, live + headless). So the video half is solved with no pixels crossing the socket:

1. **Video region** → engine export → MP4 (audio baked in). *Exists today.*
2. **Widget region** → browser `MediaRecorder` → WebM (silent, pure browser).
3. **Stitch** → one ffmpeg `vstack` + take audio from the engine MP4 → 1080×1920 MP4.

Audio "just comes along" with the engine export — no separate tap, no mux step.

## 5. The record handshake (Option B)

The two takes must be the **same take** — same start, duration, fps. The engine drives both:
"start" opens the live export **and** tells the browser to start `MediaRecorder`; "stop" ends
both; ffmpeg stacks + trims to the shorter. Two regimes:

- **Pure-automation take:** deterministic via live/offline parity — reproducible, could even
  re-render offline.
- **Live performance (twisting dials):** one-shot; both must record simultaneously.

## 6. Audio plan

A captured `<canvas>` is **silent** — master audio lives in the engine (cpal master bus), not
the browser. Two ways to get sound into the clip:

- **Mux after (recommended):** engine export already contains the master audio; the final
  ffmpeg step just keeps it. Bit-exact, no resampling drift, ffmpeg already a hard dep.
- **Pipe audio to the browser:** stream master audio as a `MediaStreamTrack`, combine with
  the canvas video track in one `MediaStream`. Single artifact, but new real-time transport
  + A/V drift risk. More moving parts.

Either way, sync to the *same start signal* — recording truly begins when `MediaRecorder`
fires, not on click, so the engine should key its audio/export window off the same event.

## 7. Cons / open risks (the honest list)

The crux tradeoff: **Option B is easier to *build*; Option A has better recording *fidelity*.**

1. **Two independent live takes drift.** Different clocks/threads; `MediaRecorder` has
   nondeterministic startup latency and emits *variable* frame rate; a throttled tab drops
   frames. ffmpeg can trim total length but not fix *interior* drift, so widgets progressively
   lag the video on long takes. Worst for the *interactive* (non-reproducible) take — the
   exciting path is the fragile path.
2. **`MediaRecorder` needs a canvas; widgets want DOM.** `captureStream()` is canvas-only.
   So either every widget renders to canvas (fights the simple idempotent-DOM model), or
   DOM→canvas rasterize per frame (slow/flaky on fonts/CSS), or screen-capture (OBS-lite,
   prompts, wrong resolution).
3. **Not WYSIWYG.** You don't see the real stacked 9:16 until ffmpeg runs — you perform blind
   to the final framing. (Option A *is* the final frame.)
4. **May not actually beat OBS on raw overhead.** Two encoders (engine H.264 + browser VP9) +
   live GPU preview + web server on one box. OBS does one hardware composite+encode. We win on
   integration/convenience, maybe not CPU/GPU.
5. **Interaction latency shows in the stitch.** Dial → `WebAction` → engine applies next frame
   → export captures it, but the widget records its motion instantly → dial moves slightly
   *before* the video reacts. Visible on tight beat-synced moves.
6. **"Cheap experiments" fights the build loop.** `static/*` is embedded via `include_str!`
   (`src/web/static_files.rs`), so every widget edit = `cargo build` + restart + hard reload.
   Mitigable by serving `static/` from disk in dev, but that's a separate change.

**The deciding question** is not "which is easier to code" but **how much manual, beat-tight
interaction happens live.** Loose/automation-driven takes → Option B's drift is invisible,
ship it. Played like an instrument with tight manual moves → Option A's frame-pipe starts
earning its complexity.

## 8. Optional refinement

Instead of video-recording the widgets, log the `AppSnapshot` stream during the take and
replay it through the widget renderer offline → encode. Decouples widget framerate/quality
from the live session entirely. A later optimization; `MediaRecorder` on the live canvas is
the simple start.

## 9. Relationship to title cards

A title/intro card is also a "share with context" feature, so it overlaps here (see
`docs/text-layer-design.md`): an **engine/export-baked** card needs the rasterize-to-texture
path so it lands in the MP4; a **browser-rendered** card could instead be just another widget
on the stage (HTML text is trivial there) — but then it only exists in the preview view, not
the exported video.
