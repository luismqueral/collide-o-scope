# Clip Cropping / Custom Loop Points — Design Sketch

Trim a clip to an **in-point** and **out-point** so it loops over a chosen window instead
of the whole file — for both video and audio. Grounded in how the decoders loop today.
This is a sketch (approach + touch points + open questions), not a build plan.

---

## 1. How looping works today (the thing we're generalizing)

Both decoders loop the **entire file** by *reopening from the top* at EOF — no seeking,
which the code calls out as "simpler and robust across codecs":

- **Video** — at EOF, `next_frame()` flushes then calls `reopen()` (`src/video/decoder.rs:160,207`),
  resetting `frame_count = 0`. `progress()` = `frame_count % total_frames / total_frames`
  (`decoder.rs:177`). Images set the `still` flag and yield one frame then hold (`decoder.rs:34,131`).
- **Audio** — same dance: at EOF `next_chunk()` reopens (`src/audio/decoder.rs:105,165`).
  The decode thread runs ahead of playback, pushing chunks into a ring buffer the cpal
  callback drains (module doc, `audio/decoder.rs:1-10`).
- **Layer transport** — `advance()` (`src/layers/mod.rs:187`) pulls *n* frames per tick
  using `speed`/`fps`/`frame_accumulator`; it doesn't know about loop windows.

So a loop point feature has one core idea: **loop at the out-point (or EOF) and resume at
the in-point**, instead of always 0 → EOF → 0.

## 2. Representation: store the window in seconds

Source-time **seconds** are the natural unit — codec-agnostic and map cleanly to both
sides (video frame = `secs * fps`; audio sample = `secs * rate`). On the layer / patch:

```
loop_start: Option<f32>   // seconds; None = start of file
loop_end:   Option<f32>   // seconds; None = end of file
```

`Option` + `#[serde(default)]` keeps old patches loading unchanged (None/None = today's
whole-file loop). Normalized 0..1 is the alternative, but absolute seconds survive a clip
being swapped for a longer/shorter one less surprisingly. We need the clip's **duration**
to set/clamp the out-point — video has `total_frames` (`decoder.rs:31`) → `secs = frames/fps`;
audio comes from the stream duration. Duration should ride along in `LayerSnapshot` so the
UI can draw the range.

## 3. Enforcing the out-point

The cleanest seam: generalize the EOF check. In `next_frame()` / `next_chunk()`, the loop
trigger becomes **"reached out-point OR hit real EOF"**:

- **Video:** out-point frame = `round(loop_end * fps)`. When `frame_count >= out_frame`,
  do the same flush+reopen the EOF path already does, then re-seek to the in-point (§4).
- **Audio:** out-point sample = `round(loop_end * out_rate)`; track samples emitted since
  (re)open and loop when the count crosses it. Mind that a chunk may straddle the boundary —
  v1 can loop at the first chunk *past* the out-point (a few ms of slop) and tighten later
  by splitting the chunk.

`progress()` should then report position **within the window** (`(pos - in) / (out - in)`),
not the whole file, so any scrubber/▮ reflects the trimmed region.

## 4. Enforcing the in-point — the one real decision (seek vs. skip)

Resuming at a non-zero in-point needs us to *get there* after reopen. Two options, mirroring
the existing reopen-not-seek tradeoff:

- **(a) Decode-and-discard from the top.** After `reopen()`, pull and throw away frames/chunks
  until reaching the in-point. **Robust across every codec** (same reason the project reopens
  rather than seeks), zero new ffmpeg surface. Cost: wasteful for a *deep* in-point on every
  loop — fine for short VJ loops, bad for "last 5 s of a 10-minute file."
- **(b) `av_seek_frame` to the nearest keyframe ≤ in-point, then decode-discard to it.** The
  standard approach; far cheaper for deep in-points. Cost: keyframe-snapping logic, codec
  edge cases, and it breaks the project's deliberate "no seek" simplicity.

**Recommendation:** ship **(a)** for v1 (matches the codebase's philosophy, trivial, correct),
and treat **(b)** as a later optimization gated on whether deep in-points on long files
actually show up in use.

## 5. Touch points

- **`src/video/decoder.rs`** — carry `loop_start`/`loop_end` (or in/out frame counts); add the
  out-point check beside the EOF branch (`~131,160`); skip-to-in after `reopen()` (`207`);
  make `progress()` window-relative (`177`).
- **`src/audio/decoder.rs`** — mirror: out-point sample check beside EOF (`105`); skip-to-in
  after `reopen()` (`165`). Watch the **buffered decode thread** — changing loop points live
  means the ring buffer still holds old-window audio, so there's a buffer's worth of latency
  before the new window is heard (flush vs. accept latency is an open question).
- **`src/layers/mod.rs`** — hold the window on the layer; pass it down when constructing/
  resetting the source. `advance()` itself stays mostly as-is (it just pulls frames; the
  decoder decides when to loop).
- **`src/patch/mod.rs`** — `loop_start`/`loop_end` fields with serde defaults; round-trip.
- **`src/web/state.rs`** — `WebAction::SetLoopPoints { layer_id, start, end }`; `LayerSnapshot`
  gains the window **+ clip duration** so the panel can render a range control.
- **`src/render_export.rs`** — export pulls frames through the same decoder, so honoring loop
  points in the decoder makes export match the live preview for free (live/offline parity).
- **frontend** — a trim control (§6).

## 6. UI sketch

- **v1 (cheap):** two handles / two numeric inputs for in and out (seconds or %), shown on a
  simple timeline bar per layer card. No waveform/thumbnails — just the range.
- **later:** a **filmstrip** of thumbnails for video and a **waveform** for audio behind the
  two handles, so you can see where you're cutting. Both are real work (decode N thumbnails;
  compute a waveform) and belong in a follow-up.

## 7. Wrinkles / open questions

- **A/V loop drift.** Video and audio decoders loop *independently* (each reopens on its own
  clock) — already a known v1 limitation (`docs/audio-plan.md §6`). Explicit loop points make
  both honor the same *window* but don't give them a shared clock, so a long session can still
  drift at loop boundaries. Loop points neither cause nor fix this.
- **Loop click (audio).** `reopen()` discards the resampler's latency tail — "a negligible
  click at the loop point" today (`audio/decoder.rs:163`). A tight musical loop makes that
  click more noticeable; a tiny crossfade at the seam is the eventual fix.
- **Live edits vs. buffering.** Moving the out-point while playing: the video reacts within a
  frame or two; the audio decode thread is buffered ahead, so it lags unless we flush.
- **Loop modes (stretch).** Forward (today), **ping-pong**, **play once / hold last frame**.
  Reverse needs real seeking, so it's well beyond v1.
- **Clamping.** Enforce `0 <= start < end <= duration`, and a sane minimum window so you can't
  trap the decoder in a sub-frame loop.
- **Still images.** No-op — one frame, no window to loop.

## 8. Verdict

Shallow at the core: the out-point is a generalization of the EOF branch the decoders already
have, and the in-point is "skip after reopen" (decode-and-discard, option **a**) — no new
ffmpeg surface for v1. The effort sits in the **edges** (patch field, web action +
duration-in-snapshot, and a trim UI) and in the **niceties** (waveform/filmstrip, loop-seam
crossfade, live-edit buffer flush) — all of which can land incrementally after a numeric v1.
