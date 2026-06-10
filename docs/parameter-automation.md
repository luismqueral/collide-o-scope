# Parameter Automation — Design Report

> **Status:** design / proposal. No code has been written for this feature yet.
> This document is the spec we'd build against, with the emphasis the request
> demanded: *how the UI works*, because the UX here is genuinely nuanced.

## 1. What we're building

Give every numeric parameter the ability to be driven by a small math
expression — like After Effects expressions — instead of a fixed value. You
type the expression **directly into the param's value field** in the web panel.

- Type `12` into a param → it's a static value (exactly today's behaviour).
- Type `sin(t)` into the same field → the param is now **automated**: a formula
  is evaluated every frame and drives the value, and the slider becomes a
  **live read-out** that animates on its own.

This is the natural generalization of the `breathe_*` and `color_drift`
parameters, which are already hardcoded time-driven oscillators baked into the
shader. Automation lets the *user* author that kind of motion on any param,
without touching Rust.

### Why this fits the architecture cleanly

One fact makes the whole feature possible: **a single `time` value already
flows through both render paths, and they agree.**

- **Live:** `src/main.rs:944-948` sets `time = self.start_time.elapsed().as_secs_f32()`
  on `master_effects` and every `layer.effects` once per frame.
- **Offline render:** `src/render_export.rs:494` sets a deterministic
  `time = frame_num * frame_interval` (where `frame_interval = 1.0 / fps`).

If automation formulas are evaluated against this same `t`, then **what you see
live is exactly what renders to disk** — no drift, no "preview doesn't match
export" surprises. This parity is the reason we evaluate in Rust, not JS (see
§7).

---

## 2. The core interaction (the heart of the UX)

Today each param row is a CSS grid of three cells (`static/style.css:547`):

```
[ label ........ ][ slider .................... ][ value ]
  --label-width        1fr                        --value-width
```

The `value` cell is a **read-only `<span class="value">`** — it just displays
the number (`static/index.html:63`, formatted by `formatValue` at
`static/app.js:906`).

**The proposal: make that `value` cell editable.** It becomes the single entry
point for automation. No separate "expression editor", no modal, no new panel.
The thing that currently *shows* the value becomes the thing you *type into*.

### The interaction, step by step

1. **Click the value** (`12.00`). It turns into a text input, selects its
   contents, and shows the current literal number — ready to overwrite.
2. **Type a number** (`8`) and press Enter/blur → static value. Identical to
   dragging the slider to 8. This must stay frictionless; most edits are still
   "just set a number".
3. **Type an expression** (`sin(t) * 10 + 10`) and press Enter/blur → the param
   becomes automated. The row enters its **automated state** (§3): a small `ƒ`
   marker appears, and the slider starts animating on its own.
4. **To remove automation:** click the value again (it shows the *expression
   text* now, not a number, so you can edit it), clear it or type a plain
   number, press Enter. Back to static.

### Why the value field, and not the slider?

- A slider can express *one number*. A formula is text. The value cell is the
  only place in the row that's already textual.
- It keeps automation **discoverable but unobtrusive** — the panel looks
  identical until you decide to type a formula. No extra chrome for the 90% case.
- It mirrors AE, where you alt-click a stopwatch and type into the value. Here
  the value field *is* the affordance; one fewer click.

### Editing affordances (specifics)

- **Enter / blur** commits. **Escape** cancels and restores the prior text.
- While the input is focused, the **server must not overwrite it** — exactly the
  same guard the sliders already use: `syncEffects` only writes a row when
  `document.activeElement !== slider` (`static/app.js:283`). We extend that
  guard to the value input: *skip the row if its value input is focused.*
- The value cell is small (`--value-width: 48px`). For typing a formula it
  should **expand on focus** — either grow to span the slider+value columns
  (`grid-column: 2 / -1`, the trick `.select-row select` already uses at
  `static/style.css:642`), or float a wider input over the row. Expanding
  in-grid is simpler and consistent with the existing select pattern.

---

## 3. Visual states of a param row

Four states. Each must be instantly legible at a glance across ~30 params.

| State | What it looks like |
|-------|--------------------|
| **Static** (default) | Today's row. Slider draggable, value shows a number. |
| **Automated** | A `ƒ` glyph appears (where? see below). The value cell shows the number *being produced this frame*, ticking live. The slider thumb animates on its own. |
| **Editing** | Value cell is a focused text input showing the literal text (number or expression). Slider is dimmed/ignored. Server pushes are suppressed for this row. |
| **Error** | Expression failed to parse/eval. Value cell text turns red (reuse the `#e55` error red already used by `.export-status.error`, `.layer-remove-btn:hover`). The param holds its last good value. Hovering shows the parser message via `title`. |

### Where the `ƒ` marker goes

The codebase already has a per-group icon-button convention: the dice
`group-rand` / `layer-fx-rand` buttons sit in group headers
(`static/index.html:58`, `static/app.js:341`). For a **per-param** marker we
have less room. Two viable options:

- **(A) Replace the value text's left padding with a `ƒ` prefix** inside the
  value cell: `ƒ 18.4`. Cheapest, no layout change, reads as "this number is
  computed". Recommended.
- **(B) A tiny toggle glyph in the label cell**, left of the label. More
  visible but eats label width (labels already ellipsis-truncate at
  `static/style.css:558`).

Recommend **(A)**: a `.value.automated::before { content: 'ƒ'; }` styled in the
accent colour (`--accent`), so an automated row visibly differs from a static
one without adding a DOM node. The accent colour also signals "interactive /
special", matching the slider thumb and focus borders.

### The slider while automated — an open decision

This is the single trickiest UX call. When a param is automated, what does its
slider *do*?

- **Option 1 — Frozen live read-out.** The thumb animates to show the current
  value but is **not draggable** (or dragging is ignored). The formula owns the
  value completely. Simplest mental model: "automated = hands off".
- **Option 2 — Draggable baseline.** Dragging sets a manual value `x` that the
  formula can reference (`x + sin(t)` → wiggle *around* wherever you park the
  slider). This is powerful and very AE-like (expressions reference the
  property's own value), but means the slider both *reads* (animated) and
  *writes* (baseline) — confusing unless we visually separate "baseline tick"
  from "live thumb".

Recommendation: **ship Option 1 first** (frozen read-out, automation fully owns
the value). Add Option 2 only if the `x` variable (§4) proves desirable in
practice. The two decisions are linked: Option 2 only matters if `x` exists.

---

## 4. Expression syntax & variables

Keep it small, math-first, and AE-flavoured so it's familiar.

### Variables

- **`t`** — elapsed seconds (float). The primary clock. Maps directly to the
  existing `time` uniform. **Always available.**
- **`x`** *(optional, tied to the slider decision above)* — the param's own
  manual/slider value, enabling *relative* automation like `x + sin(t)*5`.
  Only meaningful if we adopt slider Option 2.

Recommendation: **`t` only for v1.** It covers the overwhelming majority of
"make it move" cases and avoids the slider-ownership ambiguity. Reserve `x` for
a fast-follow once the core loop is proven.

### Functions & constants

A standard math set plus a few AE-style helpers:

| Category | Provided |
|----------|----------|
| Trig / wave | `sin cos tan` and convenience oscillators `tri(t)` (triangle), `saw(t)`, `square(t)` |
| Math | `abs min max floor ceil round sqrt pow exp log` |
| Shaping | `clamp(v,lo,hi)`, `lerp(a,b,k)`, `smoothstep(lo,hi,v)` |
| Motion helpers | `wiggle(freq, amp)` — band-limited noise around 0; `noise(seed)` |
| Constants | `pi`, `tau` (= 2π) |

`wiggle`/`noise` are the AE crowd-pleasers. **Critical:** they must be
**deterministic** — seeded from `(t, param-name)` — so the same frame produces
the same value live and in offline render. A naive `rand()` would make exports
differ from preview and break the whole parity guarantee.

### Range behaviour

Formulas can overshoot a param's documented range; that's fine. Every value
already passes through `apply_to_uniforms`, which **clamps each field to its
range** (`src/web/state.rs:284-300`, e.g. `pixelate.clamp(1.0, 32.0)`). So
`sin(t)*1000` on a 0–1 param just pins to the rails — no special handling
needed, no crashes.

### Precedent worth citing

- **After Effects expressions** (JS-based; `time`, `wiggle()`, `value`) — the
  mental model we're emulating.
- **`fasteval`** (Rust crate) — the recommended evaluator (§7).
- TouchDesigner / VVVV CHOP-style expressions and Blender driver expressions are
  the same idea in other tools; reassuring that "type math into the value" is a
  well-trodden UX.

---

## 5. Number-vs-formula detection (no mode switch)

The UI shouldn't make the user declare "this is a formula". We infer it:

```
on commit(text):
    if text.trim() parses as f32  -> static value  (set_param, today's path)
    else                          -> automation    (set_automation)
    if text.trim() is empty       -> clear automation, keep last value
```

`Number.parseFloat` won't do (it accepts `12abc`); use a strict check
(`/^-?\d*\.?\d+$/` or `Number(text)` + `Number.isFinite`). A plain number must
*never* be misread as a one-term expression, or we'd needlessly mark trivial
edits as automated.

This keeps the common case (type a number) byte-for-byte identical to current
behaviour and routes only real expressions down the new path.

---

## 6. How the slider animates live (the satisfying part)

This already works — we just feed it. The reactive loop is:

1. Render loop computes the automated value each frame (§8) and writes it into
   the uniform (e.g. `master_effects.pixelate_size`).
2. `push_web_state()` (`src/main.rs:933`) serializes the uniforms into the
   `AppSnapshot` and broadcasts it to all WebSocket clients — **every frame,
   ~30fps**, exactly as it does now.
3. In the browser, `syncEffects` (`static/app.js:272-299`) finds the row by
   `data-param` and writes `slider.value` + the value text — **but only if the
   slider isn't the active element** (line 283).

So for an automated param, the server is already streaming the live value and
the slider is already wired to follow it. **The thumb animating on its own is a
free side-effect of the existing 30fps snapshot sync.** We don't build an
animation system in JS; Rust is the clock and the browser is a dumb mirror.

One refinement: when automated, we *want* the value text to update even though
the user "owns" the param. The focus guard still applies (don't fight an open
editor), but a non-focused automated row should animate. The current guard
(`activeElement !== slider`) already permits this.

> This is consistent with the memory note that flicker/interaction bugs in this
> panel come from non-idempotent DOM churn, not external scripts. Automation
> adds **no new per-frame DOM creation** — it only writes `value`/`textContent`
> on existing nodes, the same idempotent path `syncEffects` already uses.

---

## 7. Why evaluate in Rust, not JavaScript

- **Offline render has no browser.** `render_export.rs` runs headless. If
  formulas lived in JS, exports couldn't be automated. Evaluating in the render
  loop means **the same code path drives live preview and disk export**, and
  parity (§1) holds automatically.
- **Determinism & seeding** for `wiggle`/`noise` live in one place.
- **Performance:** ~30 master params + per-layer params, evaluated once per
  frame, is trivial for a compiled expression.

### Recommended crate: `fasteval`

- Parse/compile **once** when the expression is set; evaluate cheaply every
  frame.
- Supports a **variable namespace** (a closure resolving `t`, and custom funcs
  like `wiggle`).
- Ships the standard math builtins; we add the AE helpers as custom callbacks.

Alternatives: `evalexpr` (ergonomic, slightly heavier per-eval), `meval`
(simpler, fewer features). `fasteval`'s compile-then-eval split is the best fit
for "evaluate this same formula 30× a second".

---

## 8. Data flow & where state lives

### New WebActions (mirrors the patch actions we just added in `state.rs`)

```rust
// src/web/state.rs — alongside SetParam / SetLayerParam
SetAutomation      { param: String, expr: String },          // master param
ClearAutomation    { param: String },
SetLayerAutomation { index: usize, param: String, expr: String },
ClearLayerAutomation { index: usize, param: String },
```

(Or fold "clear" into "set with empty expr". Two explicit variants read more
clearly.)

### Storage

- An `App` field, e.g. `master_automations: HashMap<String, CompiledExpr>` and a
  per-layer equivalent. Keyed by param name (the same `data-param` string the UI
  already uses everywhere).
- `SetAutomation` parses/compiles the expr; on parse error, store the error
  string and surface it in the snapshot so the row can show the **error state**
  (§3).

### Per-frame evaluation (one new block in the render loop)

Right where `time` is set today (`src/main.rs:944-948`), after assigning
`elapsed`:

```
for (param, expr) in &master_automations:
    let v = expr.eval(t = elapsed)        // fasteval
    master_effects.set_by_name(param, v)  // then existing clamp applies
// same for each layer's automations against layer.effects
```

This runs *before* the uniform upload, so the computed values are what render
and what `push_web_state` broadcasts.

### Snapshot additions (so the browser knows what's automated)

`AppSnapshot` / `EffectsSnapshot` already carry the live numeric values (that's
what animates the slider). We add a small map so the UI can render the `ƒ`
marker and error state without guessing:

```rust
// e.g. on AppSnapshot
#[serde(default)] pub automations: HashMap<String, String>,        // param -> expr
#[serde(default)] pub automation_errors: HashMap<String, String>,  // param -> message
```

The browser uses `automations` to decide which rows show `ƒ` and what text to
put back in the value field when you click to edit; `automation_errors` drives
the red error state + `title` tooltip.

---

## 9. Persistence & offline-render parity

`PatchState` (`src/patch/mod.rs`) already captures master/layers/ntsc and
round-trips to YAML. Add automations to it:

```rust
// PatchState
#[serde(default)] pub master_automations: HashMap<String, String>,
// and per-LayerConfig: #[serde(default)] pub automations: HashMap<String,String>
```

Consequences, all positive:

- **Patches store motion, not just a frozen pose.** Saving a patch with
  `pixelate = floor(saw(t)*8)+1` reloads as that *moving* effect.
- **Offline export replays automation for free.** Because `render_export.rs`
  feeds the same deterministic `t` and (will) run the same eval block, a
  rendered clip reproduces the live look frame-for-frame.
- `#[serde(default)]` keeps **old patch files loadable** (absent map = no
  automation).

---

## 10. Per-layer parameters

Everything above applies to the per-layer FX rows too. They're built in
`renderLayerCard` (`static/app.js:540-636`) with the same
`.param-row[data-param]` structure and the same `.value` span (e.g.
`static/app.js:583`), and synced by `updateLayerCard` (`:668`) using the same
focus-guarded write. So the editable-value-cell interaction, the `ƒ` marker, and
the live-animation behaviour are **identical** — they just route through the
`SetLayerAutomation { index, param, expr }` action and read from the per-layer
snapshot map. No separate UX to design.

Toggles/selects (`invert`, `blend_mode`, `grain_algo`) are **not** automatable
in v1 — automation is numeric only. Their rows keep no editable value cell.

---

## 11. Edge cases & failure UX

- **Parse error on commit:** keep the last good value, mark the row red, put the
  message in `title`. Never throw away the user's text — clicking the cell again
  shows the broken expression so they can fix it.
- **`NaN` / `inf` at runtime** (e.g. `log(t-5)` before t=5): treat as "hold last
  good value" for that frame; don't write NaN into a uniform. Optionally flash
  the error colour.
- **Division by zero / undefined var:** parser/eval error → error state.
- **Empty field:** clears automation, param keeps its current number.
- **Performance ceiling:** compiled expressions are cheap, but cap expression
  length and document that `wiggle/noise` are the only stochastic funcs (seeded).
- **Focus fights:** the existing `activeElement` guard prevents the 30fps stream
  from yanking text out from under the cursor; we must extend it to the value
  input, not just the slider.

---

## 12. Suggested phasing

1. **Spike the eval core.** Add `fasteval`, a `CompiledExpr` wrapper, the
   `HashMap` fields, and the per-frame eval block in `main.rs`. Hardcode one
   automation in code and confirm the slider animates live (validates §6 with
   zero UI work).
2. **Wire the WebActions + snapshot maps.** `SetAutomation`/`ClearAutomation`,
   broadcast `automations`/`automation_errors`.
3. **Editable value cell.** Click-to-edit, number-vs-formula detection, focus
   guard, `ƒ` marker, error state. Master params only.
4. **Per-layer parity.** Same interaction through layer actions.
5. **Persistence.** Add maps to `PatchState`; verify save→load→export round-trip.
6. **AE helpers + determinism.** `wiggle/noise/tri/saw/square`, seeded; confirm
   live and offline frames match.
7. *(optional)* The `x` variable + draggable baseline (slider Option 2, §3).

---

## 13. Open decisions to confirm before building

1. **Variables:** `t` only (recommended for v1), or also `x` = slider value?
2. **Slider while automated:** frozen live read-out (recommended), or draggable
   baseline?
3. **`ƒ` marker placement:** value-cell prefix (recommended) vs label-cell glyph?
4. **Clear semantics:** empty-string-clears vs an explicit `ClearAutomation`
   action (recommended for clarity)?

Each is a small, reversible call — but decisions 1 and 2 are linked and shape
the syntax surface, so worth settling first.

---

## Appendix A — Which effects benefit most

Not every param rewards automation equally. Ranges below are the documented
limits from `EffectUniforms` / the UI (`static/index.html`). Formulas assume the
`t`-only + helper-function set from §4.

### Top crowd-pleasers (continuous motion reads instantly)

| Param | Range | Formula | Effect |
|-------|-------|---------|--------|
| `hue_shift` | −180..180 | `t*60` or `sin(t)*180` | Endless rainbow cycle / sweep-and-return. The one everyone notices. |
| `rgb_split` | 0..30 | `abs(sin(t*2))*20` | Chromatic aberration that *throbs*. Pairs with bass. |
| `opacity` (layer) | 0..1 | `sin(t)*0.5+0.5` / `square(t*4)` | Slow crossfade or hard strobe. **The big one for layering** — turns a static stack into a composition. |
| `vignette` | 0..1.5 | `sin(t)*0.6+0.6` | Breathing iris / heartbeat. Cheap, very "alive". |

### Rhythmic / beat-synced (`square`, `saw`, `tri`, `floor`)

| Param | Range | Formula | Effect |
|-------|-------|---------|--------|
| `brightness` | −1..1 | `square(t*4)*0.5` | Hard flash-frames; classic drop move. |
| `pixelate` | 1..32 | `floor(saw(t)*16)+1` | Stepped "zoom to blocks" sweep. Quantized reads far better stepped than smooth. |
| `posterize` | 0..16 | `floor(tri(t)*16)` | Banding that ratchets in. |
| `speed` (layer) | 0.25..4 | `1+sin(t)*0.5` / `square` | Modulates the **source footage**, not just the look. Unusually expressive. |

### Atmospheric / organic (`wiggle`)

| Param | Range | Formula | Effect |
|-------|-------|---------|--------|
| `grain_intensity` | 0..0.3 | `wiggle(2, 0.1)` | Swelling film flicker; builds tension. |
| `saturation` | −1..1 | `sin(t*0.3)` | Slow grey↔vivid swells. |
| `contrast` | −1..1 | `sin(t*0.2)*0.4` | "The room is breathing." |

### Where VHS comes alive (static analog look becomes a *performer*)

Drive these with `square`/`wiggle` so the tape "acts up" in **bursts** rather
than constantly:

| Param | Formula | Effect |
|-------|---------|--------|
| `head_switching_shift` | `square(t*0.5)*60` | Periodic glitch hits. |
| `tracking_noise_wave` / `tracking_noise_snow` | `wiggle(1, …)` | Intermittent tracking wobble + static. |
| `snow_intensity` | `square(t*0.3)*0.4` | Bursts of snow. |
| `edge_wave_intensity` | `abs(sin(t))*15` | Wobbling edges. |

### Skip in v1

Booleans and selects aren't numeric, so they're out: `invert`, `color_grain`,
`blend_mode`, `grain_algo`, `tape_speed`. And `grain_size` / `breathe_*` /
`color_drift` are too subtle to read as motion — note `breathe_*` and
`color_drift` are *already* hardcoded LFOs.

### The real unlock: shared `t`, related frequencies

The payoff isn't any single param — it's driving several off the **same `t`**
with harmonically related rates: everything pulsing at `t*2`, hue cruising at
`t*30`, vignette breathing at `t*0.5`. It reads as one coherent *musical*
motion instead of random wiggling. And because formulas are deterministic
(§9), that whole "song" replays identically in offline render — exactly what
the procedural-generation goal needs.
