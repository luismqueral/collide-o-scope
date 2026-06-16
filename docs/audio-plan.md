# Audio Plan: Per-layer Audio + Audio FX + Master Bus

A design plan for adding audio to collide-o-scope. Audio can come **from a video
clip's own audio track** or from **standalone audio files you add yourself**. Every
layer gets two new sections in its card — **AUDIO** (mute / volume / pan) and
**AUDIO FX** (EQ + delay) — sitting **above the COLOR section**. A **master audio
bus** (master volume + limiter + meter) lands in the right column.

Grounded in how the code works today (file:line) with a proposed architecture, the
wiring path, the IA, parameter tables, and a phased build order. Meant to iterate on.

## Decisions locked in (from planning)

1. **Uniform layer model.** Audio-only files (mp3/wav/…) become *normal layer cards*
   with the same AUDIO / AUDIO FX groups — they just produce no video (placeholder
   thumb; visual FX groups hidden/inert). One mental model for everything.
2. **Master audio bus.** Add a master AUDIO section (master volume + limiter +
   level meter) so summed layers don't clip the live output.
3. **Varispeed.** A layer's `speed` (0.25×–4×) pitches its audio up/down like a
   turntable. Cheap resample, classic VJ feel — no time-stretch / phase vocoder.
4. **Volume in dB.** Per-layer and master volume are dB sliders (`−60 .. +6 dB`,
   0 dB = unity, gain = `10^(dB/20)`). Fixed 3-band EQ, plain tap delay, no tempo sync.
5. **Sync = wall-clock (v1).** Audio and video both advance on elapsed real time and
   each loops at its own EOF — no shared master clock. Accepts a little loop drift over
   long runs in exchange for simplicity; making audio the master clock (video resyncs
   to it) is deferred to a later phase. See §6.
6. **Dependencies approved.** Add `cpal` (output) + a small SPSC ring-buffer crate
   (`rtrb`/`ringbuf`); reuse ffmpeg's `swresample` for resampling (no extra dep).
7. **Audio-only thumbnails = static music-note icon (v1).** Rendered waveform thumbs
   deferred.

---

## 1. Why audio is a *new subsystem*, not "another effect"

The visual effects pipeline is GPU-shader-driven: one `EffectUniforms` struct → one
`effects.wgsl`, run per-layer and at master (see `docs/proposed-layer-effects.md`).
Adding a *visual* effect is "add a field to a uniform." **None of that applies to
audio.** Three facts about today's code set the scope:

- **No audio anywhere.** `VideoDecoder::open` selects only the video stream
  (`src/video/decoder.rs:28–32`, `streams().best(Type::Video)`) and there is no audio
  output device, no `cpal`, no DSP. `grep` for audio/cpal/samples finds nothing.
- **Playback is a 30 fps *pull* loop.** The render loop only advances when
  `now - last_frame_time >= FRAME_DURATION` (`src/main.rs:1447`, `TARGET_FPS = 30`
  at `:31`), then advances each layer's footage by real elapsed time
  (`layer.advance(dt, queue)`, `:1472`) and pushes web state (`:1516`). Video stays
  on wall-clock via a frame accumulator (`Layer::advance`, `src/layers/mod.rs:141–165`).
- **Audio can't ride that clock.** Audio hardware needs a *continuous* stream of
  samples at 44.1/48 kHz delivered to a real-time callback — it cannot be "pulled
  once per video frame." So audio needs its own decode + buffering + output path,
  decoupled from the render loop, communicating by messages and shared params.

**Consequence:** the work is a new `src/audio/` subsystem (decode → resample → DSP →
mix → device) plus a thin control channel from the render loop. The visual pipeline
is untouched except where a layer carries audio params alongside its `effects`.

---

## 2. Architecture

```
                    render loop (30fps, main.rs)
                          │ commands: add/remove layer, play/pause,
                          │ speed, loop-reset, param changes
                          ▼
   ┌─────────────────────────────────────────────────────────┐
   │  AudioEngine  (src/audio/)                                │
   │                                                           │
   │  per layer (has-audio):                                   │
   │    AudioDecoder (ffmpeg Type::Audio) → swresample to      │
   │    f32 stereo @ device rate  ──►  per-layer DSP           │
   │    (mute, volume, equal-power pan, 3-band EQ, delay)      │
   │            │                                              │
   │            ▼  push                                        │
   │      ring buffer (SPSC, lock-free)                        │
   │            │                                              │
   └────────────┼──────────────────────────────────────────── ┘
                ▼  pull (real-time)
        cpal output callback (own thread, device rate)
          sum layers → master volume → limiter → device
          (also writes peak level to an atomic → meter)
```

**Threads & who does what**
- **Producer (decode + per-layer DSP).** One worker thread (or a small pool) decodes
  each audio source, resamples to the device's sample-rate/stereo via ffmpeg's
  `software::resampling` (swresample — already available through `ffmpeg-next`, no new
  decode dep), applies the layer's DSP, and writes into that layer's ring buffer.
  This is block-based and *not* real-time-critical, so it may read params behind a
  cheap `Mutex<AudioParams>` once per block.
- **Consumer (cpal callback).** Real-time thread. Does only allocation-free, lock-free
  work: drain each layer's ring buffer, sum, apply master volume + limiter, write to
  the device, and store the output peak in an `AtomicU32` (f32 bits) for the meter.
  **No locks, no decode, no ffmpeg in the callback.**
- **Render loop.** Sends commands (add/remove/clip-swap, play/pause, speed, loop
  reset) over a channel and updates the per-layer/master param blocks. Reads the meter
  atomic once per frame to ship it in the web snapshot.

**Varispeed.** Pitch-follows-speed = read each layer's buffer with a fractional read
pointer advanced by `speed` (linear interpolation between samples). This is separate
from the fixed source→device sample-rate conversion in the resampler. `speed > 1`
drains the buffer faster (higher pitch), `< 1` slower (lower pitch) — so the producer
must keep enough lead buffered (size the ring for the 4× worst case).

**Clock / sync model (v1).** Both audio and video advance on the *same wall-clock*
elapsed time and both loop at their file's end — no shared master clock. This is the
same philosophy as the existing video accumulator and is "good enough" for a visual
performance tool. Sample-accurate A/V sync is an explicit **non-goal for v1** (see §6).

---

## 3. Data model & wiring path

Mirrors the per-effect wiring path in `docs/proposed-layer-effects.md §1`, adapted for
audio. New code is the `src/audio/` module; the rest is threading the params through
the existing snapshot / action / patch / frontend seams.

**New module `src/audio/`**
- `mod.rs` — `AudioEngine` (owns cpal stream, mixer, command channel, meter atomic),
  `AudioParams` (plain struct, *not* a GPU uniform), `MasterAudioParams`.
- `decoder.rs` — `AudioDecoder` (ffmpeg `Type::Audio` + `software::resampling` to f32
  interleaved stereo at device rate; loops at EOF like `VideoDecoder` does).
- `dsp.rs` — biquad EQ (RBJ-cookbook low-shelf / mid-peak / high-shelf), delay line
  (ring buffer w/ feedback + dry/wet), equal-power pan, simple peak limiter.
- `output.rs` — cpal device/stream setup, the mixer, SPSC ring buffers (small new dep,
  e.g. `rtrb`/`ringbuf`).

**`AudioParams` (per layer)** — `mute: bool`, `volume: f32`, `pan: f32`,
`eq_low/eq_mid/eq_high: f32` (dB), `delay_time: f32` (ms), `delay_feedback: f32`,
`delay_mix: f32` (dry↔wet). Plus a `Default`.

**1. `src/layers/mod.rs`** — give `Layer` an `audio: AudioParams` and make the video
source optional so audio-only clips fit the uniform model:
- Today `decoder: VideoDecoder` is mandatory and `Layer::new` fails with no video
  stream. Introduce a `LayerKind { Video, Image, Audio }` and make the video source
  `Option<VideoDecoder>` (or wrap in a `LayerSource` enum). `Layer::new` branches: a
  file with a video stream → as today; audio-only → placeholder texture (1×1 / a
  generated waveform thumb), `kind = Audio`.
- `advance()` no-ops the video decode for `kind == Audio`; the compositor
  (`render_layers`, `src/renderer/state.rs`) skips audio-only layers (no pixels).
- Extend `is_supported_media` (`src/layers/mod.rs:191`) with audio extensions:
  `mp3, wav, flac, m4a, aac, ogg, oga, aif, aiff`.

**2. `src/web/state.rs`**
- `LayerSnapshot` (`:233`) gains the audio fields + `has_audio: bool` + `kind: String`
  (so the UI dims/hides visual groups for audio-only cards).
- New `MasterAudioSnapshot { volume, limiter, meter }`; add it + the live `meter`
  level to `AppSnapshot` (`:27`).
- Reuse `WebAction::SetLayerParam` (`:343`) for per-layer audio params; add
  `SetMasterAudioParam { param, value }` (or fold into `SetParam`). Add
  `ResetLayerGroup` support for the new `audio` / `audiofx` groups (`:339`).

**3. `src/main.rs`**
- `SetLayerParam` dispatch (`:348–`) gains clamp arms for each audio param (pattern
  like `"opacity"`/`"speed"` at `:352–360`); writes into `layer.audio` and forwards to
  the engine.
- `add_layer` (`:168`) branches video vs audio-only and registers the source with the
  `AudioEngine`.
- `push_web_state` (`:917`) fills the new audio fields + master meter.
- Engine lifecycle: build `AudioEngine` at startup; on play/pause
  (`ToggleLayerPause`/`ToggleMasterPause`), `speed` change, and loop reset, send the
  matching command. Map layer `speed` → varispeed read rate; map `paused` → silence.

**4. `src/patch/mod.rs`** — audio must round-trip through saved patches *and* export
replay:
- `LayerConfig` (`:200`) + `EffectsConfig` (or a sibling `AudioConfig`) gain the audio
  fields with `#[serde(default)]`; `from_layer`/`apply_to_layer` (`:719`/`:742`),
  `top_fields`/`set_field` (`:766`/`:780`), and `param_meta` (`:31`) entries for each.
- `PatchState` (`:796`) carries `MasterAudioConfig`.

**5. Frontend**
- `createLayerCard` (`static/app.js:775`): insert an **AUDIO** group and an **AUDIO FX**
  group *between* `data-layer-group="blend"` (`:795`) and `="color"` (`:829`). They use
  the existing `.param-row` slider / `toggle-row` / `select-row` patterns — so
  randomize/reset/automation come for free. For audio-only cards, mark visual groups
  hidden/inert based on `kind`.
- Master **AUDIO** group in the right column (`static/index.html`, a new `.fx-group`
  near `data-group="output"` `:55`): master volume slider, limiter toggle, and a
  meter element fed by the per-frame snapshot.
- `syncLayer` / master sync paths update the new sliders; meter is redrawn from
  `snapshot.meter` each frame. `matrix-schema.js` should list the audio params so they
  appear in the matrix grid too.

---

## 4. Information architecture

**Per-layer card order** becomes:
`BLEND → AUDIO → AUDIO FX → COLOR → DIGITAL → WARP → KEY → SHIFT → …`
(AUDIO/AUDIO FX inserted right above COLOR, as requested.)

**Audio-only clips (uniform model).** A standalone audio file added from the library
creates a normal layer card whose video bits are inert: a waveform / music-note
placeholder thumbnail, BLEND/COLOR/DIGITAL/WARP/KEY/SHIFT either hidden or visibly
dimmed, and AUDIO + AUDIO FX active. It occupies a layer slot but contributes no
pixels to the composite.

**Master bus.** Right column gets a master **AUDIO** group: master volume, a limiter
toggle (clip guard), and a level meter. This is the single summing point so multiple
layers can't blow out the live output.

**Media library.** Audio files appear alongside video tiles (`syncLibrary`,
`static/app.js`). They need a thumbnail: a static
music-note icon for v1 (video thumbs are JPEG frames today — audio has no frame, so
this is a small new branch in the thumbnail path; rendered waveform thumbs deferred).

---

## 5. Parameter sets (proposed — iterate)

### AUDIO (per layer)
| Param | Range | Default | Notes |
|---|---|---|---|
| `mute` | bool | off | hard mute (gain 0) |
| `volume` | −60 .. +6 dB | 0 dB | dB slider; −60 dB = floor/silent, 0 dB = unity. Gain = `10^(dB/20)` |
| `pan` | −1 .. 1 | 0 | equal-power (−1 L, 0 C, +1 R) |

### AUDIO FX (per layer)
Fixed 3-band EQ + a plain tap delay. No adjustable frequencies, no tempo sync.

| Param | Range | Default | Notes |
|---|---|---|---|
| `eq_low` | −24 .. +12 dB | 0 | low-shelf, fixed ~120 Hz |
| `eq_mid` | −24 .. +12 dB | 0 | peaking, fixed ~1 kHz |
| `eq_high` | −24 .. +12 dB | 0 | high-shelf, fixed ~6 kHz |
| `delay_time` | 0 .. 1000 ms | 0 | plain tap (ms), no BPM sync |
| `delay_feedback` | 0 .. 0.95 | 0 | clamp <1 to avoid runaway |
| `delay_mix` | 0 .. 1 | 0 | dry↔wet |

### Master AUDIO (right column)
| Param | Range | Default | Notes |
|---|---|---|---|
| `master_volume` | −60 .. +6 dB | 0 dB | dB slider; final gain before limiter |
| `limiter` | bool | on | brick-wall / soft-clip guard |
| `meter` | read-only | — | output peak (0..1), pushed each frame |

---

## 6. Sync & known concerns

- **Loop drift on long sets** *(accepted — see Decision 5)*. Video loops by reopening
  at EOF (`decoder.rs:reopen`); audio loops at its own EOF. Over many loops of a short
  clip, A/V can drift a few ms. Accepted for v1; the later refinement is to have the
  engine expose playback position so the video loop can resync to it (audio-as-master).
- **Buffer latency.** cpal buffer size sets output latency (a few ms–tens of ms).
  Tune for "tight enough" without underruns; varispeed at 4× needs extra lead buffer.
- **Param updates into the callback.** Per-layer DSP runs in the producer (lock OK);
  the callback only touches ring buffers + master atomics. Keep it that way.
- **Device selection / channels.** v1 targets the default output device, stereo.
  Device picking / multichannel routing is future work.

---

## 7. Export

`render_export.rs` replays a captured `PatchState` to render frames **offline** (not
real-time), so audio export is a *separate* effort: render the mix to a buffer
(decode + DSP offline) and mux it into the output file via ffmpeg. **Out of scope for
v1** — live audio first; export stays video-only until a later phase. The patch
round-trip (step 4) still stores audio params so an export-with-audio phase has the
data it needs.

---

## 8. Phasing & recommendation

1. **Phase 0 — Plumbing + sound out.** Add `cpal` + ring buffer dep; `AudioEngine`
   skeleton; decode the audio stream from *video* files; resample → stereo → device at
   1×. Goal: a clip's audio plays and loops. No controls yet. Proves the threading +
   clock model end-to-end (lowest risk to de-risk first).
2. **Phase 1 — AUDIO + master bus (the MVP).** Per-layer mute/volume/pan, master
   volume/limiter/meter, varispeed (pitch-follows-speed), play/pause + loop wiring.
   Full path: layers → web/state → main → patch → frontend. *This is the core ask.*
3. **Phase 2 — AUDIO FX.** Fixed 3-band EQ (biquads) + plain tap delay
   (time/feedback/mix). No tempo sync, no adjustable EQ frequencies.
4. **Phase 3 — Audio-only clips.** Layer-source refactor (`Option<VideoDecoder>` /
   `LayerKind`), audio file extensions, waveform thumbnails, uniform audio-only cards.
   *Can run in parallel with Phase 1/2* since the user wants both sources co-equally —
   only sequenced later because it touches the layer/render/library seams more widely.
5. **Phase 4 — Future.** Offline audio mixdown + mux for export; per-layer meters;
   ducking/sidechain; spectrum; device/channel selection.

**Recommendation:** do Phase 0 first to validate the audio engine in isolation, then
Phase 1 for a usable live result, pulling the Phase 3 layer-source refactor forward if
adding standalone audio clips early matters more than EQ/delay.

---

## 9. Open questions

All v1 questions resolved — see "Decisions locked in" at the top. Nothing blocking;
next step is Phase 0.

## Dependencies (new, approved)

- `cpal` — cross-platform audio output (macOS CoreAudio).
- `rtrb` or `ringbuf` — lock-free SPSC ring buffer for the callback boundary.
- Resampling reuses `ffmpeg-next`'s `software::resampling` (swresample) — must track
  the system ffmpeg version, same gotcha as the existing video decode path.
