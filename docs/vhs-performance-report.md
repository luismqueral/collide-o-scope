# VHS / NTSC Framerate Slowdown — Report & Fix Plan

**Symptom:** Turning on the VHS effect drops the framerate for the *whole* app, including every other layer and effect. Nothing else does this.

**Short answer:** It's both. The VHS effect is fundamentally a **CPU** effect (ntsc-rs does its work on the processor, not the graphics card), and the way we currently bolt it into the render loop **amplifies** that cost. The CPU work is unavoidable; the amplification is fixable.

---

## 1. Why VHS is different from every other effect

Think of it like After Effects. Most of our effects (pixelate, RGB split, color, glitch, warp) are **GPU shaders** — they're like a GPU-accelerated plugin that runs on the graphics card. The frame is *born* on the GPU, gets effects applied on the GPU, and gets shown from the GPU. The CPU barely touches the pixels. That's why you can stack a lot of them cheaply.

VHS (the `ntsc-rs` library) is **not** a shader. It's a CPU library. It expects a plain array of pixels sitting in normal memory, and it loops over them on the processor. Our own file even says so:

> `src/ntsc/mod.rs:1` — *"Applies analog VHS effects ... as a CPU-based post-process on the final composite RGBA buffer."*

So to use it, we have to physically move the finished frame **off the GPU, into CPU memory, process it, then push it back to the GPU** — every single frame. None of the other effects pay this cost.

---

## 2. What actually happens each frame when VHS is on

This is the sequence in `src/main.rs:1364-1384`. It only runs when VHS is enabled:

```
1. Submit GPU work so the composite frame is finished        (queue.submit)
2. Read the whole frame back from GPU → CPU   ← BLOCKS        (readback_composite)
3. Shrink it to half size on the CPU          ← 1 core        (NtscState::apply)
4. Run ntsc-rs on the half-size frame         ← CPU, the real cost
5. Grow it back to full size on the CPU       ← 1 core
6. Upload the frame back to the GPU                           (write_composite)
```

Every one of those steps runs **one after another, on the main thread** — the same thread that's trying to hit your framerate. Their time adds directly onto each frame. Here's the cost of each:

### Step 2 — The blocking read-back (the "stall")
`src/renderer/state.rs:816`
```rust
let _ = self.device.poll(wgpu::PollType::Wait { ... });
```
`Wait` means: *stop and do nothing until the GPU is completely finished and has handed the pixels back.* Normally the CPU and GPU work at the same time (CPU preps frame N+1 while the GPU draws frame N). This line forces them to take turns — the CPU sits idle waiting for the GPU, then the GPU sits idle waiting for the CPU. We lose all the overlap.

It also allocates a fresh full-frame buffer here every frame (`src/renderer/state.rs:826`). At 1080p that's ~8 MB allocated and thrown away 30+ times a second. *(Note: the GPU-side staging buffer at line 771 is correctly reused — it's only this CPU-side `Vec` that churns.)*

### Steps 3 & 5 — The resize loops (single-threaded)
`src/ntsc/mod.rs:109-123` (downscale) and `135-143` (upscale).
These are hand-written nested `for` loops that touch every pixel on **one CPU core**. At 1080p the upscale alone copies ~2 million pixels per frame, one at a time, while your other 7+ cores sit idle. Step 3 also allocates another fresh buffer each frame (`src/ntsc/mod.rs:108`, ~2 MB at half-res).

*(The half-res trick you added already cut step 4's cost by ~4×, which is why it's processed small — that part is good.)*

### Step 4 — ntsc-rs itself (the irreducible cost)
This is the actual VHS simulation: head switching, tracking noise, snow, chroma loss, etc. It's CPU work by nature and scales with how many pixels it processes. We can't make this free — only feed it fewer pixels or run it less often.

---

## 3. So: CPU problem, or codebase problem?

| Cost | Nature | Can we fix it? |
|------|--------|----------------|
| ntsc-rs processing (step 4) | Inherent CPU cost | Only reduce/defer it |
| Blocking GPU↔CPU stall (step 2) | **Codebase structure** | Yes |
| Single-threaded resize (steps 3,5) | **Codebase structure** | Yes |
| Per-frame allocations (steps 2,3) | **Codebase structure** | Yes |

Your instinct was right on both counts: the core is a CPU cost, but our code makes it hurt more than it has to. The three "codebase structure" rows are the opportunity.

---

## 4. Fix options

Ordered easiest/safest first. Each is independent — we can do one, stop, and measure.

### Option A — Reuse the buffers *(small, safe, low risk)*
Stop allocating the two big `Vec`s every frame; allocate once and reuse. Removes ~10 MB/frame of allocation churn.
- **Effort:** small · **Risk:** low · **Expected gain:** modest, but free.

### Option B — Parallelize the resize loops with `rayon` *(small change, good gain)*
`rayon` turns those single-core pixel loops into all-core loops with a near one-line change (`for` → `par_chunks`). Steps 3 & 5 get ~4–8× faster depending on your CPU.
- **Effort:** small · **Risk:** low · **Expected gain:** good. Adds one dependency (`rayon`).

### Option C — Run VHS at a lower/decoupled framerate *(cheap, very effective for live use)*
Process VHS every *other* frame (or every 3rd) and reuse the last VHS result in between. The video keeps playing full-speed; only the analog grain updates a bit slower — usually invisible for a VHS look. Halving the rate roughly halves the whole cost.
- **Effort:** small–medium · **Risk:** low · **Expected gain:** large · **Trade-off:** slightly less "frantic" noise animation.

### Option D — Do the down/upscale on the GPU *(bigger, biggest structural win)*
Shrink the frame to half-res in a shader *before* reading back, and upscale in a shader *after* writing back. Then step 2 only moves a quarter as many pixels across the bus, and steps 3 & 5 disappear entirely. The blocking stall remains but moves far less data.
- **Effort:** medium · **Risk:** medium (touches the render pipeline) · **Expected gain:** large.

### Option E — Fully async/double-buffered readback *(advanced, removes the stall)*
Let the GPU and CPU keep overlapping by always processing the *previous* frame's pixels (one frame of latency) instead of blocking. Removes the step-2 stall entirely.
- **Effort:** large · **Risk:** higher · **Expected gain:** large · **Trade-off:** 1 frame of VHS latency + more moving parts. Probably overkill if C+D already get you there.

---

## 5. Recommended sequence

1. **A + B together** — buffer reuse + `rayon`. Small, safe, and should give a noticeable bump on their own. Good first PR.
2. **Measure.** If live performance is comfortable, stop here.
3. If still not enough: **C** (decouple the VHS framerate) — biggest bang for the least structural risk.
4. Only if you want it truly cheap at full res: **D** (GPU-side scaling). Leave **E** unless D isn't enough.

All of these are additive and reversible. None require rewriting how the other effects work. The export path (`apply_full_res`, `src/ntsc/mod.rs:150`) is separate and unaffected by A–E, so render quality won't change.

---

## Key code references
- Per-frame VHS sequence: `src/main.rs:1364-1384`
- Blocking read-back + CPU `Vec` alloc: `src/renderer/state.rs:763` (poll at `:816`, alloc at `:826`)
- Write-back to GPU: `src/renderer/state.rs:838`
- Half-res downscale / upscale loops + `small` alloc: `src/ntsc/mod.rs:95-147` (alloc at `:108`)
- Full-res export path (unaffected): `src/ntsc/mod.rs:150`
