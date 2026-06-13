# Feature Backlog

A grab-bag of features/improvements to pick up later. Each is grounded in how the
code works today (file:line references) with a proposed approach, the points it
touches, and open questions. Roughly ordered easy → involved, not by priority.

| # | Feature | Area | Rough effort |
|---|---------|------|--------------|
| 1 | Label above the Layers section | frontend only | trivial |
| 2 | Collapsible left column | frontend only | small |
| 3 | Recategorize effects | frontend (+ a little wiring) | small |
| 4 | Improve the Master Controls section | frontend (design pass) | small–medium |
| 5 | Media library folders | backend + frontend | medium |
| 6 | Alternate effect param inputs (XY pad, etc.) | frontend (+ pattern) | medium |
| 7 | Non-throttled / settable master framerate | core render loop | medium–large |
| 8 | Push the datamosh / feedback "obliteration" family | shader (cheap, feedback already wired) | small per idea |

---

## 1. Label above the Layers section

**Current state.** The center column (`static/index.html` ~33–43) has no header — it's
just `layers-list` + `layers-empty`. Every other column already has a lowercase
`.panel-title`: "patches", "media library" (left), "Master Effects" (right).

**Goal.** Add a "layers" `panel-title` above the center column for visual consistency.

**Approach.** Add an `<h3 class="panel-title">layers</h3>` at the top of `.col-center`
(or wrapping `.layers-section`). Confirm the existing `.panel-title` style reads well
centered over the layer cards; nudge spacing in `static/style.css` if needed.

**Touch points.** `static/index.html`, possibly `static/style.css`.

**Open question.** Should the label sit inside the scroll area or stay pinned as the
column scrolls? (Right column's title is pinned above `fx-scroll`.)

---

## 2. Collapsible left column

**Current state.** `.col-left` holds the patches + media library sections
(`index.html` 13–30). Layout is the 3-column `.columns` grid in `static/style.css`.

**Goal.** Let the left column collapse to reclaim width for layers / the larger UI,
then restore it.

**Approach.** A small toggle (chevron) button that toggles a `.collapsed` class on
`.col-left`. CSS transitions the column width to a thin rail (or 0) and hides the
section bodies. Persist the state in `localStorage` so it survives reloads (the panel
already uses localStorage-style client state patterns). Consider collapsing to a thin
strip with an expand affordance rather than fully removing it.

**Touch points.** `static/index.html` (toggle button), `static/app.js` (toggle handler
+ persistence), `static/style.css` (`.columns` grid template + `.col-left.collapsed`).

**Open questions.** Collapse to 0 vs. a thin rail? Does the center/right reflow to fill
the space (grid `fr` units) or stay fixed? Same treatment eventually for the right
column?

---

## 3. Recategorize effects

**Current state.** Master FX groups live in `index.html` as `.fx-group` blocks with
`data-group` ids: **OUTPUT**, **DIGITAL**, **ANALOG**, **MOTION**, **VHS**, plus the
**Render** group. Per-group randomize/reset buttons key off `data-group` and iterate
the params inside that group (`static/app.js`). Per-layer cards build their own grouping
in `createLayerCard`. The grouping is purely a *frontend* concern — the backend
`EffectUniforms` is one flat struct; group membership is just which rows sit under which
header.

**Goal.** Re-organize which params live under which category (and possibly rename/add
categories) so related controls are grouped more intuitively.

**Approach.** Decide the new taxonomy first (this is the real work — see open question),
then move `.param-row` blocks between `.fx-group`s in `index.html` and mirror the same
grouping in `createLayerCard` (app.js) so per-layer and master stay consistent.
Randomize/reset keep working as long as each row keeps its `data-param` and lands under
the right `data-group`.

**Touch points.** `static/index.html` (master groups), `static/app.js` (`createLayerCard`
per-layer groups), maybe `static/style.css`.

**Open question.** What's the target taxonomy? Current split is roughly
digital/analog/motion/vhs. Candidate reframings: by *purpose* (color / geometry / noise /
time) or by *signal era* (digital vs. analog) — needs a decision before moving rows.

---

## 4. Improve the Master Controls section

**Current state.** Right column header is "Master Effects" with a transport row
(`index.html` 47–52): `btn-play-all` (pause/play), `btn-stop` (titled "Reset FX", ↺
glyph), and `ws-status` — a **red dot that is actually the WebSocket connection
indicator**, not a record light. Screenshot shows the title centered above a pause
button, a reset button, and the red dot, which reads ambiguously (the dot looks like a
record indicator; "Master Effects" sits oddly above transport controls).

**Goal.** Make this section clearer and better organized.

**Sub-ideas to consider (need a design pass / direction):**
- Disambiguate the `ws-status` dot — move it out of the transport row, shrink it, or
  give it an explicit "connected/disconnected" affordance so it doesn't read as "record".
- Clarify play/pause vs. "Reset FX" (the ↺ button resets effects, not playback — the
  tooltip says so but the icon is ambiguous).
- Reconsider the "Master Effects" title vs. the transport: maybe a dedicated transport
  bar separate from the FX title, or label the transport explicitly.
- Possible future: a real record/export affordance distinct from connection status.

**Touch points.** `static/index.html` (transport row + title), `static/style.css`,
`static/app.js` (`syncTransport`, ws-status updates).

**Open question.** What's the intended mental model — is this "global transport" or
"master FX controls"? That decision drives the layout.

---

## 5. Media library folders

**Current state.** `.library-grid` is a flat grid; `syncLibrary(files)`
(`static/app.js:973`) renders a flat list of files scanned from `library_folder`.
Double-click a tile to add it as a layer. The backend currently sends a flat file list.

**Goal.** Support nested folders in the media library (browse, expand/collapse, add
files from within folders).

**Approach.** Backend: emit relative paths (or a small tree structure) instead of a flat
filename list — likely a recursive scan of `library_folder` filtered by
`is_supported_media` (`src/layers/mod.rs:183`). Frontend: render collapsible folder nodes
and file tiles; keep double-click-to-add working for files at any depth. Persist
expanded/collapsed folder state client-side.

**Touch points.** `src/web/state.rs` (library message shape — flat → tree/paths), the
folder scan that populates it (`src/main.rs` / web layer), `static/app.js` (`syncLibrary`
→ tree render), `static/style.css`, `static/index.html` (`#library-grid`).

**Open questions.** Recursive scan depth (full tree vs. one level at a time)? Tree view
vs. breadcrumb navigation? Watch for folder changes on disk, or rescan on demand?

---

## 6. Alternate effect param inputs (XY pad, etc.)

**Current state.** All params are `.param-row` variants: a range slider + value span,
`select-row` (dropdown), or `toggle-row` (checkbox). Slider input listeners send
`SetParam` / `SetLayerParam` actions over the WebSocket (`static/app.js`). See the
existing `docs/parameter-automation.md` for related automation/modulation thinking.

**Goal.** Explore richer input widgets — e.g. an **XY pad** that drives two correlated
params at once (RGB split × hue, the two breathe axes, a 2D warp), plus possibly numeric
entry, drag-to-scrub values, or LFO/modulation hooks.

**Approach.** Add a new control type alongside `param-row` (e.g. `xy-pad` bound to two
`data-param`s). On drag, map the pad's x/y to each param's min/max and emit the existing
SetParam actions — **no backend change** (it's still two scalar params). Build it once as
a reusable widget so it works in both the master column (`index.html`) and per-layer
cards (`createLayerCard`).

**Touch points.** `static/app.js` (new widget + param wiring), `static/index.html` /
`createLayerCard` (markup), `static/style.css`. Backend untouched if it decomposes to
existing scalar params.

**Open questions.** Which param pairs are worth an XY pad? Should the pad show a live
dot / trail? Does this fold into the parameter-automation plan (LFOs driving the same
params)?

---

## 7. Non-throttled / settable master framerate

**Current state.** The render loop is **hard-capped at 30fps**: `FRAME_DURATION =
1000/TARGET_FPS` with `TARGET_FPS = 30` (`src/main.rs:30–31`), and `RedrawRequested`
only proceeds once `now - last_frame_time >= FRAME_DURATION` (`main.rs:1233`). The
surface present mode is `Fifo` (vsync) (`src/renderer/state.rs:104`). `master_fps` does
**not** raise the loop rate — it only gates *content* stutter on the fixed 30-tick grid:
`stride = round(30 / master_fps)` (`main.rs:1242–1243`), which is why the UI exposes only
30 / 15 / 10 / 7.5 / 6 (the rates that land evenly on the 30-grid).

**Goal.** Allow the master to run faster than 30 (e.g. 60) or uncapped, and/or set an
arbitrary master framerate rather than only the 30/k presets.

**Approach.** Make the loop's target configurable instead of the `TARGET_FPS = 30`
constant. Key consequences to handle:
- The stride/stutter math assumes a 30-tick grid; raising the loop rate means
  recomputing stride against the new target and rethinking which presets to expose.
- `present_mode: Fifo` caps presentation to display refresh — uncapped needs `Mailbox`
  or `Immediate` (with the usual tearing/perf tradeoffs).
- Interaction with the blocking VHS half-res readback (it stalls the tick); higher target
  rates make that stall more visible. The catch-up decode already keeps footage in sync,
  but perf headroom matters.

**Touch points.** `src/main.rs` (timing constants 30–31, redraw gate + stride logic
1233–1243), `src/renderer/state.rs` (`present_mode`), `src/web/state.rs`
(`SetMasterFramerate` clamp), the MOTION "Framerate" `<select>` in `index.html` (presets),
`static/app.js`.

**Open questions.** Is the ask "run the live preview at 60" or "expose an *uncapped*
mode", or both? Should arbitrary framerates be allowed (free stutter rates) or stay on a
preset grid? How does this relate to *export* fps (already independent in
`ExportConfig.fps`)?

---

## 8. Push the datamosh / feedback "obliteration" family

**How datamosh works today.** It's a **frame-feedback** effect — the same trick as the
classic JS canvas "draw the previous frame back onto itself with some alpha" loop, but
on the GPU.

1. **Capture.** At the start of every frame, the renderer copies *last* frame's final
   output into a dedicated `feedback_texture` before this frame's passes overwrite it
   (`src/renderer/state.rs:575–588`). It's bound into the effects shader as `prev_tex`
   (binding 2, `effects.wgsl:74`).
2. **Displace ("motion vectors").** The **Block mosh** glitch (`effects.wgsl:339–351`)
   chops the frame into a grid (`block_size`), and for cells whose random hash fires
   (`block_prob`, reseeded at `block_speed`) it offsets `sample_uv` by `block_intensity`
   and sets a flag `mosh_amt = 1.0`. These jumps are the fake "motion vectors".
3. **Bleed the past ("P-frame bloom").** Only where a block fired
   (`datamosh > 0 && mosh_amt > 0`, `effects.wgsl:402–409`), the shader samples
   **prev_tex** at the displaced UV and `mix(color, prev, datamosh)`. Because last
   frame already held the smear, trails **accumulate** frame-over-frame, bounded by the
   mix so they fade instead of blowing out.

That's a faithful approximation of real datamoshing (drop keyframes → later frames' motion
vectors get applied to the wrong reference → blocks drag): random block displacement plays
the role of the motion vectors, and sampling the previous output plays the role of the
stale reference frame.

**Key usage fact:** `datamosh` does nothing on its own — it's gated by `mosh_amt`, so
**Block Amt must be > 0** for any smear to appear. All params are per-layer, in the
**SHIFT** group (`static/app.js:795–868`).

**Why this is a cheap thing to expand.** The feedback texture (`prev_tex`) is already
captured, bound, and wired through both the live renderer *and* the export path. Any new
effect that reads `prev_tex` is essentially free — no new passes, no new textures, no Rust
plumbing beyond a uniform field (see `docs/proposed-layer-effects.md` for the per-field
wiring path + the `EffectUniforms` packing constraint).

**Directions to explore (each is a small, mostly-shader change):**
- **Ungated persistence / video feedback.** Add a "persistence" param that bleeds
  `prev_tex` across the *whole* frame, not just fired blocks (drop the `mosh_amt` gate for
  that term). Classic infinite-trail / echo look.
- **Feedback transform (zoom/rotate).** Sample `prev_tex` at a scaled/rotated/offset UV
  instead of `sample_uv` → droste infinite-zoom "tunnel" and spiral smears.
- **Freeze / bloom-out.** Stop advancing the source frame while still applying
  displacement + feedback → the image dissolves into pure motion (true datamosh
  "bloom"). Needs a hold flag on the layer's decode.
- **Keyed feedback.** Gate the bleed by luma or motion so only bright / moving regions
  smear while static areas stay sharp.
- **Channel-desync feedback.** Feed R/G/B back at different mix rates for trailing color
  ghosts (pairs naturally with `shift_chroma`).
- **Additive accumulation.** `color + prev * k` (clamped) instead of `mix` → blooming
  light-trail obliteration.

**Touch points (typical).** `src/effects/params.rs` (+field, mind packing),
`src/shaders/effects.wgsl` (mirror field + logic near the existing datamosh block),
`src/web/state.rs` + `src/main.rs` + `src/patch/mod.rs` (param plumbing/clamp/round-trip),
`static/app.js` `createLayerCard` SHIFT group (+ slider). Backend render loop and feedback
capture are already done.

**Open questions.** Per-layer only, or also a master-level feedback? Should
persistence/echo be its own param or fold into `datamosh`? How much of this should be one
"obliteration" macro vs. discrete sliders?

---

## Notes

- Items 1–4 are mostly frontend and can be done independently / quickly.
- Items 5–7 cross the Rust ↔ web boundary; check `src/web/state.rs` message shapes and
  the `static/app.js` sync functions together.
- For any new effect param/category work, `docs/proposed-layer-effects.md` documents the
  full per-effect wiring path (params.rs → effects.wgsl → web/state.rs → main.rs →
  patch/mod.rs → frontend) and the `EffectUniforms` packing constraint.
