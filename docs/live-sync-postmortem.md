# Frozen Sliders — a live-sync post-mortem

> **Status:** post-mortem. The two bugs described here are fixed; this is the
> story of finding them, kept so the next "why won't the panel update?" hunt
> starts from the right place.

## 1. The symptom

After the automation-editor work, the web panel stopped reflecting live values:

- An automated param (e.g. a layer's **speed**) animated in the native preview
  window, but the corresponding slider/value in the browser sat frozen.
- Pressing a layer **Reset** didn't visibly change the panel either.
- **Refreshing the page fixed it** — values snapped to correct — but there was
  no continuous live feedback after that.

The maddening part: every cheap explanation was wrong.

## 2. What we ruled out (and why each was tempting)

| Hypothesis | Why it looked right | Why it was wrong |
|------------|--------------------|------------------|
| **Window occlusion** | Broadcasting is coupled to `RedrawRequested`; an occluded macOS window *could* stop redrawing. | User confirmed the native automation animates whether the window is front or back — so the render loop (and `push_web_state`) is running regardless. |
| **Stale browser cache** | `app.js` is `include_str!`-embedded at compile time; a stale tab is plausible. | Hard-reload (Cmd+Shift+R) changed nothing. |
| **`type:"state"` gate fails on broadcasts** | `ws.onmessage` only syncs when `msg.type === 'state'`. | Initial connect and broadcast serialize the **same** `AppSnapshot` (`main.rs` `push_web_state`: `*app = snapshot.clone()` then `tx.send(serde_json::to_string(&snapshot))`). Byte-identical shape; both carry `type:"state"`. |
| **A stuck `dragging` flag** | `syncLayers` early-returns on `if (dragging) return;`. | `dragging` is only set by grabbing a `.layer-grip` and is cleared on `pointerup`. The user was automating, not reordering. |
| **A per-frame rAF clobber** | The editor added a `requestAnimationFrame` preview loop. | `drawPreview` bails unless the editor is open and only writes to the preview canvas / ghost / knobs — never the live param sliders. |

The decisive evidence came from the user's own console: `ws.onmessage` fired
~30/s (`readyState === 1`), **no parse errors**, counter climbing into the
hundreds. So the data arrived, the handler ran, and the `sync*` functions
executed every frame **without throwing** — yet the DOM didn't move. That
narrows it to: *the sync write targets the wrong node, or the value it writes
never actually changes.* Two independent bugs, one per branch of that "or".

---

## 3. Bug A — `syncEffects` wrote into a layer's slider, not the master's

`syncEffects` located each row with a **global** query:

```js
const row = document.querySelector(`.param-row[data-param="${param}"]`);
```

But layer cards reuse the **exact same** `data-param` names as the master panel
(`pixelate`, `rgb_split`, `hue_shift`, …), and in the DOM `#layers-list`
(`index.html:35`) sits **before** `#master-fx` (`index.html:53`). So once any
layer exists, `document.querySelector` returns the **first layer card's** row
and `syncEffects` writes the master value *there* — the real master slider is
never touched.

`document.querySelector` returns the first match in document order. Put the
duplicate earlier in the DOM and a global selector silently retargets.

This is why it was a *regression*: with zero layers loaded there's no duplicate,
so it "worked"; the bug only appears once a layer card is on screen.

**Fix** — scope the lookup to the master container (`app.js`, `syncEffects`):

```js
const masterFx = document.getElementById('master-fx');
if (!masterFx) return;
// …
const row = masterFx.querySelector(`.param-row[data-param="${param}"]`);
```

All master `data-param` rows live inside `#master-fx`; layer cards live in
`#layers-list`. Scoping excludes the duplicates. (`updateLayerCard` was already
correct — it scopes to `card.querySelectorAll(...)`.)

---

## 4. Bug B — layer `speed`/`opacity`/`fps` automation was a silent no-op

The per-frame layer evaluation routed every automated param through the effect
uniforms:

```rust
for (param, expr) in &layer.automations {
    layer.effects.set_by_name(param, expr.eval(elapsed, beat, bpm));
}
```

But `EffectUniforms::set_by_name` (`effects/params.rs`) only knows the ~14
effect params and drops anything else on the floor:

```rust
match param {
    "pixelate" => self.pixelate_size = v.clamp(1.0, 32.0),
    // … 13 more …
    _ => {}   // <- speed / opacity / fps land here and vanish
}
```

`speed`, `opacity`, and `fps` are **layer-level fields** (`layer.speed`, etc.),
*not* effect uniforms. So automating them changed nothing, and the snapshot kept
serializing the unchanged `speed: l.speed` — a permanently frozen slider, even
though the broadcast was flowing fine.

(Manual drags worked because `SetLayerParam` *does* special-case these three to
the layer fields. Only the automation path forgot to.)

**Fix** — route the layer-level fields directly in the eval loop (`main.rs`):

```rust
for (param, expr) in &layer.automations {
    let v = expr.eval(elapsed, beat, bpm);
    match param.as_str() {
        "opacity" => layer.opacity = v.clamp(0.0, 1.0),
        "speed"   => layer.speed   = v.clamp(0.25, 4.0),
        "fps"     => layer.fps     = v.clamp(1.0, 60.0),
        _ => layer.effects.set_by_name(param, v),
    }
}
```

Clamps mirror the manual `SetLayerParam` ranges so automated and dragged values
behave identically.

---

## 5. Why "refresh fixes it" was a red herring

On a fresh load the panel is rebuilt from the initial snapshot **once**:
`createLayerCard` stamps the card's `innerHTML`, then a one-shot pass writes the
current values. That made the values *look* correct after a refresh and framed
the bug as "live updates broken" rather than "two specific write paths broken."
In reality:

- Bug A only ever affected the **master** panel (and only with a layer present).
- Bug B only ever affected **three layer fields**.

They overlapped into one vague "nothing updates live" symptom. Splitting the
symptom by scope — *is it master-scoped or layer-scoped?* — points straight at
A vs B.

---

## 6. Takeaways

1. **Frozen-live-value ≠ broken transport.** When `onmessage` fires every frame
   with no errors, the WebSocket/broadcast layer is fine. Look at the *write*:
   wrong node, or unchanged value.
2. **Global `querySelector` + duplicated `data-param` is a trap.** The master
   and per-layer panels share param names. Always scope DOM lookups to their
   container (`#master-fx` vs the layer `card`). DOM order decides which
   duplicate a global selector hits.
3. **Mind the layer-field / effect-uniform split.** `speed`/`opacity`/`fps` are
   layer fields; everything else is an `EffectUniforms` value reached via
   `set_by_name`. Any code that writes a layer param — manual actions,
   automation, resets — must handle *both* halves, or `set_by_name`'s `_ => {}`
   arm will quietly swallow it.
4. **Don't restart on a fresh branch to escape a bug you don't understand.**
   Both bugs lived in core live-sync plumbing, not the editor feature — a
   rebuild would have carried them along.
