# Automation Editor — Design Doc

**Status:** Draft, for review · 2026-06-10
**Scope:** Frontend only (no Rust changes). Replaces the read-only Formula
Cookbook with an interactive modulator builder.

---

## 1. Goal

Today you author automations by *typing a formula* into a param's value cell.
The Formula Cookbook helps you discover formulas but only lets you **apply a
fixed one** — you can't tune it. This editor turns that surface into a guided
**modulator builder**: a small rack of knobs (shape, rate, depth, …) that
compiles to a formula live, with:

- **Presets as starting points** (the old cookbook cards) that *seed the knobs*
  instead of applying verbatim.
- A **raw-formula escape hatch** for power use (the text field you have today).
- A live preview that rides the tapped tempo.

It is opened **per parameter**: each automatable param row grows a small `ƒ`
launcher to the left of its value, and clicking it opens the modal **locked to
that param** (no scope/param pickers — the row already knows whether it's a
master or layer param). The global cookbook button goes away.

This solves three problems: discoverability (you see what each control does),
tunability (drag a knob instead of editing algebra), and consistency (every
result is range-scaled to the target param, building on the work just landed).

---

## 2. Ground truth: how automation works today

So the builder is designed against what the engine can actually run.

- **Backend** (`src/automation.rs`): a param may hold a compiled `fasteval`
  `Expr`, evaluated each frame as `eval(t, beat, bpm)`. The **same `t`** is used
  by the live loop and the offline exporter, so **preview == export**.
- **Inputs:** `t` (elapsed seconds), `beat` (beats since the last tap downbeat),
  `bpm` (current tempo). Constants `pi`, `tau`.
- **Helpers:**
  - Oscillators `tri(x)`, `saw(x)`, `square(x)` — **period 1, output −1..1**.
  - Shaping `clamp(v,lo,hi)`, `lerp(a,b,t)`, `smoothstep(lo,hi,v)`.
  - Motion `wiggle(freq,amp)`, `noise(seed)` — deterministic value-noise.
  - `fasteval` built-ins: `sin`, `cos`, `abs`, `floor`, `min`, `max`, `sqrt`,
    `^` (pow), etc.
- **Automatable params (14, identical for master & per-layer)** — each clamped
  to a fixed range (`set_by_name` in `effects/params.rs`, mirrored as
  `PARAM_RANGE` in `app.js`):

  | param | range | param | range |
  |---|---|---|---|
  | pixelate | 1 … 32 | grain_intensity | 0 … 0.3 |
  | rgb_split | 0 … 30 | grain_size | 1 … 4 |
  | hue_shift | −180 … 180 | vignette | 0 … 1.5 |
  | saturation | −1 … 1 | color_drift | 0 … 0.02 |
  | brightness | −1 … 1 | breathe_scale | 0 … 0.05 |
  | contrast | −1 … 1 | breathe_rotation | 0 … 2 |
  | posterize | 0 … 16 | breathe_position | 0 … 0.02 |

- **Range-aware scaling (just added):** a normalized shape — `'uni'` (0..1) or
  `'bi'` (−1..1) — is rescaled to the param's `[lo,hi]` by `scaledExprFor`. The
  builder generalizes this: it emits a normalized shape, then range-scales it.
- **Install / remove:** `set_automation` / `set_layer_automation` and
  `clear_automation` / `clear_layer_automation` actions. (Emptying a value cell
  now removes the formula — bug fixed alongside this doc.)

**Key constraint:** the engine only runs a **closed-form expression string**.
The builder must therefore *generate a formula*, not a drawn curve. That is
exactly what a parametric modulator does — and why no backend change is needed.

---

## 3. The modulator model

A **modulator** is a fixed set of controls that compile deterministically to one
expression. Mental model: a synth LFO, or After Effects expression controls.

Shapes fall into **two families**:

- **Periodic (looping):** sine, triangle, saw, square — repeat exactly every
  cycle. These are the "modulator" feel.
- **Random (non-looping):** smooth-random, stepped-random — these *wander* and
  never repeat (see §3.4). This is the "truly random, not just a looping
  modulator" behavior; it's the same control rack, just a different shape.

### 3.1 Controls

| Control | Values | Role |
|---|---|---|
| **Shape** | sine, triangle, saw up, saw down, square, smooth-random, stepped-random | the waveform |
| **Sync** | Beat-synced ↔ Free | choose the time base |
| **Rate** | beat mode: musical division (4 bars … 1/8); free mode: period in seconds | how fast it cycles (random shapes: how often it wanders/re-rolls) |
| **Depth** | 0–100 % | fraction of the param's range it sweeps |
| **Center** | 0–100 % | where the midpoint sits within the range |
| **Phase** | 0–360° | offset within the cycle (random shapes: see **Seed**) |
| **Invert** | on/off | flip the waveform |
| **Seed** | integer (random shapes only) | picks *which* random sequence — re-roll for a different wander without changing rate |

### 3.2 How controls become a formula

Define the phase input:

```
x = timebase * rate + phase01
        timebase = beat   (Beat-synced)   |  t   (Free)
        rate     = 1/division (beats)     |  1/period (seconds)
        phase01  = degrees / 360
```

Pick the bipolar shape `S(x)` (−1..1):

| Shape | `S(x)` |
|---|---|
| sine | `sin(x*tau)` |
| triangle | `tri(x)` |
| saw up | `saw(x)` |
| saw down | `0-saw(x)` |
| square | `square(x)` |
| smooth-random | `wiggle(rate, 1)` (uses `t`; see §3.4) |
| stepped-random | `noise(floor(x))` |

Apply invert (`S → 0-S`), then fold to 0..1 with **depth** and **center**:

```
out01 = clamp(center + S(x) * depth/2, 0, 1)
```

Finally range-scale to the param (reusing the existing rule):

```
final = lo + out01 * (hi - lo)
```

### 3.3 Worked examples (these reproduce current cookbook cards)

- **Beat pulse** on `rgb_split`: sine, Beat-synced, 1/beat, depth 100 %, center
  50 %, phase 0 → `(clamp(0.5+sin(beat*tau)*0.5,0,1))*30`.
- **Off-beat strobe** on `brightness`: square, Beat-synced, 1 per 2 beats →
  `-1+(clamp(0.5+square(beat/2)*0.5,0,1))*2`.
- **Slow drift** on `hue_shift`: sine, Free, period ~21 s, depth 100 % →
  `-180+(clamp(0.5+sin(t*0.3)*0.5,0,1))*360`.

The generator **simplifies** for readability: drop `+phase01` when 0, skip the
`clamp`/center wrapper when depth = 100 % and center = 50 %, omit the range
wrapper when the param is already 0..1.

### 3.4 Random & non-looping shapes

The user wants automations that *wander* — "truly random, not just a looping
modulator." This is the random shape family, and it falls out of the engine's
existing value-noise helpers **with no backend change**.

- **smooth-random (`wiggle`)** — organic, continuous drift (a hand-held-camera
  wobble). Rate maps to its `freq` arg. It reads `t` (not `beat`), so
  "beat-synced smooth-random" uses `wiggle(bpm/60 * div, 1)` to track tempo.
- **stepped-random (`noise`)** — re-rolls to a new held value on each grid step:
  `noise(floor(timebase*rate) + seed)` (musical when beat-synced).

**Why these are non-looping (the key point for this request).** Unlike the
periodic shapes — `sin`, `tri`, `saw`, `square` all repeat every cycle —
`wiggle`/`noise` index an **ever-advancing position** along `t`. As `t` climbs,
they keep reading fresh noise cells, so the output **never repeats** (within
float precision: effectively forever for a performance or a render). That is
exactly the "doesn't feel like a loop" quality the user is after — and it
requires only picking the random shape, nothing new in Rust.

**"Random" vs "deterministic" — they're not opposites here.** `wiggle`/`noise`
are *deterministic*: the same `t` always yields the same value (they're
hash-based value-noise, not an RNG — see `src/automation.rs`). That is a
**feature, not a limitation**:

- It is what makes **preview == export** — a rendered clip wanders identically
  every time you render it, so loops are seamless and re-renders are stable.
- It still *looks* random and non-repeating to the eye, because the wander is
  aperiodic over the timescales we use.

So "non-looping" (what the user wants) and "deterministic" (what export needs)
both hold at once. We are **not** introducing a live RNG/`Math.random()` source:
that would make each render different and break the parity guarantee that the
whole automation system is built on.

**The Seed control** gives back the *variety* you'd expect from randomness
without sacrificing determinism: it picks **which** fixed sequence plays. Re-roll
Seed for a different wander at the same rate; the chosen seed is baked into the
formula, so that specific wander reproduces on export. Implemented as an additive
offset into the noise domain:

- stepped-random: `noise(floor(timebase*rate) + seed)`.
- smooth-random: offset wiggle's phase by the seed so it samples a different
  stretch of noise — `wiggle(freq, 1)` shifted by `seed` (documented as
  approximate in preview, same caveat as today's noise preview).

If a future pass genuinely needs *unbounded, never-seeded* randomness for live
(non-rendered) use, that's a separate backend feature (a per-session RNG var
that is **disabled / frozen during export**) — explicitly out of scope here and
called out in §6.

---

## 4. UI / layout

### 4.1 Entry point — the per-param `ƒ` launcher

Every **automatable** param row (the 14 in §2, on both master and each layer)
gains a small `ƒ` button immediately to the **left of its value cell**:

```
  RGB Split   ▔▔▔▔▔●▔▔▔▔   ƒ  12.0
                          └ launcher (opens the editor for THIS param)
```

- It sits between the slider and the value readout — `.param-row`'s grid gains
  a narrow `auto` column before the value.
- Shown on automatable rows only (NTSC params, toggles, selects are excluded —
  same list as `AUTOMATABLE`). Dim by default; **highlighted (accent) when an
  automation is active** on that param, so it doubles as the "this is animated"
  indicator (it supersedes today's `ƒ ` text marker on the value).
- Clicking it opens the modal **pre-targeted** to that row's scope + param. The
  row knows its scope: master rows → master; layer rows → that layer index.
- Clicking the value cell to type a raw formula still works exactly as today —
  the launcher is an *additional*, guided path, not a replacement.

The global cookbook button (`#btn-cookbook` in the transport row) is **removed**
— the per-param launcher is now the sole entry.

### 4.2 Modal layout

Reuse the `#cookbook` modal shell, but the header shows a **fixed context
title** (the param it was opened for) instead of scope/param dropdowns:

```
┌─ Automate · Master › RGB Split ───────────────────────────────  ✕ ┐
│ Preset:  ( Beat pulse ) ( Strobe ) ( Slow drift ) ( Wiggle ) … ▾    │
├─────────────────────────────────────────────────────────────────────┤
│  Shape  [∿ sine][△ tri][⊿ saw▲][⊿ saw▼][⊓ sq][≈ rnd][▥ step]        │
│  Sync   ( Beat ▸ Free )      Rate   [ 1 / beat ▾ ]                   │
│  Depth  ▔▔▔▔▔●▔▔  80%        Center ▔▔▔●▔▔▔▔  50%                    │
│  Phase  ▔●▔▔▔▔▔▔   0°        Invert [ ]                              │
│ ┌─ preview (rides tap tempo) ───────────────────────────────────┐   │
│ │  max ─────────────────────────────────────────────────         │   │
│ │        ╱‾‾╲      ╱‾‾╲      ╱‾‾╲     ● playhead                  │   │
│ │  min ─╱────╲────╱────╲────╱────╲──────────────────────         │   │
│ └────────────────────────────────────────────────────────────────┘   │
│  ƒ  -180+(clamp(0.5+sin(beat*tau)*0.4,0,1))*360     [ Edit raw ]     │
├─────────────────────────────────────────────────────────────────────┤
│                                   [ Remove ]  [ Copy ]  [ Apply ]     │
└─────────────────────────────────────────────────────────────────────┘
```

- **Context title** — e.g. `Master › RGB Split` or `Layer 2 › Vignette`. The
  param is **locked** for the session of the modal; to edit a different one,
  close and click that row's `ƒ`.
- **Preset strip** seeds all knobs from a named control-set, then you tweak.
- **Shape-dependent controls** — for periodic shapes the slot in the bottom-left
  is **Phase** (0–360°); selecting a random shape (smooth-/stepped-random) swaps
  it for **Seed** (re-roll, §3.4), since phase is meaningless for a wander. The
  rest of the rack is unchanged.
- **Preview** reuses the current `drawCurve` infra: plot over ~4 units (4 beats
  when synced, 4 s when free), draw min/max guide lines + midline, moving
  playhead aligned to the live `beat`, value dot. Auto-scaled Y as today.
- **Generated formula** line shows the live, range-scaled expression. **Edit
  raw** reveals a text input prefilled with it — editing there detaches from the
  knobs (escape hatch).
- **Apply** → `set_automation` (master) or `set_layer_automation` (layer) for
  the locked target. **Remove** → the matching `clear_*`. **Copy** → clipboard.

Any control change recomputes the formula + preview immediately. (There is no
in-modal param/scope switching — that's fixed by the launcher you clicked.)

---

## 5. Editing an existing automation

When the modal opens for a param that already has a formula, ideally restore the
knobs. Reverse-parsing arbitrary expressions is out of scope. Pragmatic rule:

- If the installed expression **matches a builder-generated pattern**, restore
  the controls.
- Otherwise open in **raw mode** showing the text (knobs disabled until the user
  picks a shape, which takes over).

Documented limitation, not a blocker.

---

## 6. Code impact

**Frontend only.**

- `static/index.html` — add the per-param `ƒ` launcher to each automatable row
  (§4.1) and remove the global `#btn-cookbook`. Replace the cookbook card grid
  inside `#cookbook` with the builder markup (shape buttons incl.
  smooth-/stepped-random, sync/rate/depth/center/phase/invert, **Seed** for
  random shapes, preview canvas, formula line + Edit-raw, footer buttons).
  Replace the scope/param dropdowns with the fixed **context title** (§4.2).
- `static/app.js` — add `buildExpr(controls, param)` (the §3 generator,
  including the random-shape + Seed branches) and the preset → control-set
  table; wire the `ƒ` launchers → open the modal locked to that row's
  scope+param; wire control events → recompute + preview; keep one preview rAF
  while open. **Reuse:** `scaledExprFor` logic, `AUTOMATABLE`, `PARAM_RANGE`, JS
  mirror helpers (`tri/saw/square/wiggle/noise/…`), `sendAction`, `drawCurve`.
- `static/style.css` — `ƒ` launcher (dim default / accent when active), builder
  control styling (shape button row, knob sliders, preview frame, formula line).
- **No Rust changes.** `set_automation` / `set_layer_automation` / `clear_*`,
  `Expr::new`, `set_by_name` clamping, and the eval loop are all reused as-is.
  The random shapes ride the existing deterministic `wiggle`/`noise` helpers —
  see §3.4.

Optional **future** backend helpers (not in this pass): square **pulse width**,
waveform **skew/ease**, **stacked** modulators (sum/multiply), and a genuine
**live-only RNG** variable (a non-deterministic random source that is frozen to
seeded value-noise during export, preserving §2's preview == export guarantee).

---

## 7. Verification plan

1. Knob changes update the formula line **and** preview live; preview Y
   auto-scales; playhead rides tap tempo for beat-synced shapes.
2. Clicking a row's `ƒ` opens the modal **locked to that scope+param** (title
   reads e.g. `Layer 2 › Vignette`); **Apply** installs it; cell shows `ƒ` and
   the launcher highlights; effect moves.
3. **Remove** clears it; cell returns to a normal value; launcher dims again.
4. **Preset** seeds the knobs; tweaking then Apply installs the tweaked formula.
5. **Edit raw** lets you install a hand-written expression; bad input shows the
   existing parse-error styling.
6. Re-scaling: the same shape on two params (e.g. hue vs color_drift) fills each
   one's full range proportionately.
7. **Random shapes:** smooth-/stepped-random visibly *wander without looping*
   over a long preview; changing **Seed** produces a different wander at the
   same rate.
8. **Export parity:** a beat-synced automation renders at the set tempo; a
   random-shape automation renders the **same** wander every time (seed baked
   in), confirming determinism holds.
9. Emptying a value cell still removes the formula (regression check).
