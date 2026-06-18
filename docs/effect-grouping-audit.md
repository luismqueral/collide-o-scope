# Effect Grouping Audit + Reorganization Options

An audit of how parameters are grouped in the control panel today (LAYER columns
and the MAIN/master column), the problems with the current scheme, and a set of
**concrete options** for reorganizing — each with trade-offs, scope, and risk —
so we can pick a direction before the independent-audio-clip phase.

Grounded in the current code: `static/matrix-schema.js` is the single source of
truth for the matrix view's groups/order; `static/matrix.js` renders it (with a
layer-only reorder via `layerGroupOrder()` and a master loop that uses schema
order). The classic view duplicates the order in `static/app.js` +
`static/index.html`. Backend reset keys are mapped in `matrix.js resetGroupKey()`
→ `reset_layer_group` arms in `main.rs`.

---

## 1. Current state

### LAYER column (per-layer; current order after the recent reorder)

| # | Group | Params (count) | Underlying concern |
|---|---|---|---|
| 1 | LAYER | clip, opacity, speed, fps, blend, visible, paused (7) | source **+** transport **+** composite (mixed) |
| 2 | AUDIO | mute, volume, pan (3) | audio mix |
| 3 | AUDIO FX | eq ×3, delay ×3 (6) | audio fx |
| 4 | COLOR | hue, sat, bright, contrast (4) | color grade |
| 5 | DIGITAL | pixelate, rgb split, posterize, invert (4) | *digital* artifacts |
| 6 | WARP | wave ×4, swirl ×2, bulge ×2 (8) | geometric distortion |
| 7 | KEY | chroma key ×5, bg ×2 (7) | compositing |
| 8 | SHIFT | slice ×5, block ×4, chroma, jitter ×2, datamosh (13) | glitch grab-bag |
| 9 | FEEDBACK | persist, zoom, rotate, luma, chroma, additive (6) | temporal |
| 10 | TRANSFORM | x, y, scale, fit (4) | geometric placement |

### MAIN column (master/global)

OUTPUT (dims, framerate) → COLOR (4) → DIGITAL (4) → ANALOG (grain ×4, vignette,
drift = 6) → MOTION (breathe ×3) → AUDIO (master vol, limiter = 2) → VHS/NTSC (~19).

---

## 2. Problems

1. **Two competing taxonomies.** Most groups are organized by *pipeline
   stage/domain* (COLOR, WARP, KEY, FEEDBACK), but degradation is organized by
   *aesthetic character* (DIGITAL vs ANALOG vs VHS). "Make it look broken" is
   spread across three places; distortion is split by mechanism. No single axis.
2. **Order ≠ signal flow.** Pipeline is decode → geometry/color/stylize →
   **blend composite** → master → VHS. Group order is loosely historical.
   TRANSFORM (placement, a *fundamental* property) is last; BLEND (the composite
   step) is buried at the top inside LAYER.
3. **Geometry is scattered.** TRANSFORM (rigid: move/scale) and WARP (non-rigid:
   wave/swirl/bulge) both answer "where do the pixels go" but sit at opposite ends.
4. **SHIFT is overloaded.** 13 params, ~5 distinct effects (slice / block /
   jitter / datamosh / chroma-shift) in one group — biggest, least coherent.
5. **LAYER conflates 3 concerns:** source (clip), transport (speed/fps/paused),
   compositing (opacity/blend/visible).
6. **Layer↔Main asymmetry, no stated rule.** COLOR/DIGITAL are per-layer *and*
   master; grain/vignette is master-only; ANALOG and VHS (both analog
   degradation) are separated by MOTION + AUDIO.
7. **Only audio gets a base/FX split.** Video families are flat; audio is AUDIO +
   AUDIO FX. Defensible, but currently inconsistent rather than deliberate.

---

## 3. Cross-cutting decisions (every option must answer these)

- **D1 — BLEND/opacity placement.** Signal-flow-correct position is the composite
  stage (near the bottom); habit/frequency-of-use wants it at the top. Pragmatic
  default: keep it bundled in LAYER at the top unless we go two-tier (Option 4),
  which can split a COMPOSITE section cleanly.
- **D2 — Keep or flatten AUDIO / AUDIO FX.** Keep the split (matches the
  mixer mental model + our recent collapse-by-default), or flatten to one AUDIO
  group for consistency with the flat video families.
- **D3 — Symmetry rule.** Adopt: *a family sits in the same relative slot in both
  columns; master-only families keep their slot.* So one map works everywhere.
- **D4 — Single source of truth.** Fold the classic view's hardcoded order into
  the same schema so the two views can't drift. Worth doing alongside any reorg
  (and cheap to do as part of Option 1).

---

## 4. Options

### Option 0 — Status quo
Keep as-is. Listed for comparison.

### Option 1 — Signal-flow reorder (minimal)
Pure reorder of existing groups to follow the pipeline. **No renames, no merges,
no membership changes.**

- **LAYER:** LAYER → AUDIO → AUDIO FX → **TRANSFORM → WARP** → COLOR → DIGITAL →
  SHIFT → KEY → FEEDBACK
  (placement+distortion = geometry first, then grade, then stylize/glitch, then
  key composite, then temporal trails).
- **MAIN:** OUTPUT → **AUDIO** (moved up) → COLOR → DIGITAL → MOTION → **ANALOG →
  VHS/NTSC** (degradation clustered at the end; VHS last = final stage).
- **Delivers:** the master-AUDIO-toward-top move you originally asked for.
- **Pros:** immediate clarity; fully reversible; no backend changes.
- **Cons:** doesn't fix the grab-bags (SHIFT) or the mixed taxonomy; LAYER still
  bundles 3 concerns.
- **Effort/Risk:** **Low / Low.** Files: `matrix-schema.js` order (+ `matrix.js`
  layer reorder already exists), classic `app.js`/`index.html` order. No Rust.

### Option 2 — Reorder + tame the grab-bags (medium)
Option 1, plus structural cleanup of the two worst groups:

- **Merge TRANSFORM + WARP → GEOMETRY** (Move params, then Distort params; can use
  visual sub-dividers).
- **Rename SHIFT → GLITCH**, with sub-dividers: Slice / Block / Jitter / Datamosh /
  Chroma.
- **LAYER:** LAYER → AUDIO → AUDIO FX → **GEOMETRY** → COLOR → DIGITAL →
  **GLITCH** → KEY → FEEDBACK. **MAIN:** as Option 1.
- **Pros:** kills the biggest coherence problems; names map to user intent.
- **Cons:** renames/merges ripple into the backend.
- **Effort/Risk:** **Medium / Medium.** Touches `resetGroupKey()`, `reset_layer_group`
  arms in `main.rs`, patch `param_meta` group labels, both views.

### Option 3 — Unify the degradation axis (taxonomy fix)
Introduce a **DEGRADE** family and stop splitting by analog-vs-digital character.

- **MAIN:** OUTPUT → AUDIO → COLOR → MOTION → **DEGRADE** { digital · film
  (grain/vignette/drift) · tape (VHS/NTSC) }.
- **LAYER:** …COLOR → **DEGRADE** { digital · glitch } → KEY → FEEDBACK.
- **Pros:** one consistent axis; "make it look broken" lives in one place.
- **Cons:** VHS is ~19 params — nesting it readably really wants the two-tier
  rendering from Option 4, otherwise DEGRADE becomes a huge flat block.
- **Effort/Risk:** **Medium-High / Medium.** Best done *with* Option 4, not alone.

### Option 4 — Two-tier sections (maximal IA)
Add a **supergroup/section** tier above today's groups; collapse at the section
level (synergizes with the collapse-by-default we just shipped).

- **LAYER sections:** INPUT {source, playback} · AUDIO {mix, fx} · GEOMETRY
  {transform, warp} · COLOR · STYLIZE {digital, glitch} · COMPOSITE {key, feedback,
  blend}.
- **MAIN sections:** OUTPUT · AUDIO · GRADE {color, motion} · DEGRADE {digital,
  analog, vhs}.
- **Bonus:** cleanly splits LAYER's bundled concerns (source / playback / composite)
  — resolves D1 and Problem 5.
- **Pros:** best navigation for a long list; absorbs Options 2 & 3 as sub-structure.
- **Cons:** by far the most code.
- **Effort/Risk:** **High / Medium-High.** New nested-header rendering in
  `matrix.js`, schema gains `section` membership, two-level collapse logic, classic
  view rework, backend reset keys for any renames.

---

## 5. Comparison

| Option | Fixes order | Fixes grab-bags | Fixes taxonomy | Splits LAYER | Backend change | Effort | Risk |
|---|---|---|---|---|---|---|---|
| 0 Status quo | – | – | – | – | none | – | – |
| 1 Signal-flow reorder | ✅ | – | – | – | none | Low | Low |
| 2 + tame grab-bags | ✅ | ✅ | partial | – | yes | Med | Med |
| 3 Unify degrade | partial | – | ✅ | – | yes | Med-High | Med |
| 4 Two-tier sections | ✅ | ✅ | ✅ | ✅ | yes | High | Med-High |

---

## 6. Recommendation

Sequence rather than pick one:

1. **Ship Option 1 now** — reversible, zero backend risk, and it delivers the
   master-AUDIO-up move you wanted. Fold the classic view into the same schema (D4)
   while we're in there.
2. **Then Option 2** once the order feels right — break up SHIFT→GLITCH and
   merge GEOMETRY; these are the highest-value coherence fixes.
3. **Consider Option 4 only if** the flat list still feels unwieldy after 1+2.
   Treat Option 3's DEGRADE idea as *sub-labeling inside Option 4*, not a
   standalone change (the VHS param count makes it awkward without sections).

Open questions to settle before building: **D1** (blend placement) and **D2**
(keep vs flatten AUDIO/AUDIO FX).
