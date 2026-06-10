# Plan: server-side generation + YouTube autopilot

How `collide-o-scope` goes from an interactive tool to an unattended pipeline that renders procedural videos and posts them to YouTube on its own.

## The split

Two halves, each doing what it's already good at:

- **Rust = the renderer.** The binary grows a few headless CLI subcommands (`render`, `walk`, `batch`) that take a patch (or an anchor + temperature), select videos, roll a duration, and write finished MP4s to a folder. No window, no web panel.
- **Python = the scheduler + uploader.** The legacy `autopilot.py` already does organic timing, the upload window, the manifest, delete-after-upload, and YouTube OAuth. It keeps doing all of that. The *only* thing that changes is the one line where it shells out to render — it calls the Rust binary instead of `multi-layer.py`.

This is deliberate. The Python upload path (`scripts/upload/youtube-upload.py`, OAuth via `client_secret.json` + token cache) is working and battle-tested. There's no reason to reimplement YouTube's API client in Rust. The Rust binary just needs to drop MP4s in a folder the way `multi-layer.py` does today.

```
            ┌────────────────────────── VPS (cron, hourly) ──────────────────────────┐
            │                                                                         │
  cron ─▶  autopilot.py tick                                                          │
            │   ├─ maybe_refresh_sources()   (unchanged)                              │
            │   ├─ maybe_render()  ──▶  collide-o-scope batch ...   ◀── THE ONE CHANGE │
            │   │                          (Rust, headless)  ──▶  output/*.mp4        │
            │   └─ maybe_upload()  ──▶  youtube-upload.py --file --public (unchanged)  │
            │                                                                         │
            └─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1 — Rust CLI subcommands

### 1.1 Add the `rand` crate

Python has `random.Random(seed)` in the stdlib. Rust keeps randomness out of the language core, so add it explicitly. In `Cargo.toml`:

```toml
rand = "0.8"
```

Then a seeded, reproducible generator is:

```rust
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

let mut rng = StdRng::seed_from_u64(seed);
```

Same mental model as `random.Random(seed)` — a seed in, a deterministic sequence out. Carry the seed into the output metadata and any piece is re-renderable from its name.

### 1.2 Dispatch subcommands before the event loop

Today `main()` (`src/main.rs:1391`) treats `args[1]` as a file or folder, then unconditionally builds the winit `EventLoop` and runs the GUI. The headless entry just branches *before* that:

```rust
fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();

    // Headless subcommands bail out before any window/web server is created.
    match args.get(1).map(|s| s.as_str()) {
        Some("render") => return cli::run_render(&args[2..]),
        Some("walk")   => return cli::run_walk(&args[2..]),
        Some("batch")  => return cli::run_batch(&args[2..]),
        _ => {}
    }

    // ... existing GUI path (web server + EventLoop) unchanged ...
}
```

Put the new code in a `src/cli/mod.rs` module. Because the subcommands `return` early, none of the winit/egui/web machinery ever initializes — that's what "bypass winit" means concretely. No GPU surface, no window, no browser opening.

### 1.3 The render primitive (already done)

`render_export.rs` is the whole renderer already. `run_export` builds a headless wgpu device (`compatible_surface: None`), opens a `VideoDecoder` per layer from `{library_folder}/{filename}`, runs the effect + composite pipelines, applies NTSC, and pipes raw RGBA into one `ffmpeg` over stdin. It's driven by:

```rust
ExportConfig { width, height, fps, duration_secs, output_path }
```

The GUI calls this through `ExportJob::start(patch, config, library_folder)`, which spawns a background thread so the UI stays responsive. The CLI doesn't need that — it can call the underlying `run_export` **synchronously** (or `start` then join the handle) and exit when the file is written. A CLI process has nothing else to do while it renders.

So `render` is thin:

```
collide-o-scope render --patch patches/anchor.yaml \
                       --library videos/ \
                       --out output/foo.mp4 \
                       --duration 90
```

It deserializes the YAML into a `PatchState` (serde already derives this — `src/patch/mod.rs:55`), builds an `ExportConfig`, calls `run_export`. Done.

### 1.4 The walk engine

This is the new logic, and it's small because the reflection registry already exists. `param_meta(name)` (`src/patch/mod.rs:28`) hands back `{ step, min, max }` for every knob, and `EffectsConfig::set_field(key, value)` / `LayerConfig::set_field` write a field by string name. So the walker never hard-codes parameter names:

```rust
/// Nudge every numeric field of a patch by a gaussian step scaled by temperature.
fn walk(patch: &mut PatchState, temperature: f32, rng: &mut StdRng) {
    // master effects
    for (_group, fields) in patch.master.grouped_fields() {
        for (key, current) in fields {
            if let Some(meta) = param_meta(key) {
                if let Ok(v) = current.parse::<f32>() {
                    let step = gaussian(rng) * temperature * meta.step;
                    let next = (v + step).clamp(meta.min, meta.max);
                    patch.master.set_field(key, &next.to_string());
                }
            }
        }
    }
    // repeat for each layer's effects + the layer-level knobs (opacity/speed/fps)
}
```

`gaussian(rng)` is a standard-normal sample (`rand_distr::StandardNormal`, or roll your own Box–Muller — it's four lines). `step` from the registry is the natural unit of "one nudge" — the same number the keyboard uses for a single keypress — so `temperature = 1.0` reads as roughly one keystroke of drift per parameter. The blog's temperature ladder (0.1 / 0.5 / 1.0 / 2.0) maps directly onto this multiplier.

Because clamping uses each field's own `min`/`max`, the walk can't produce an invalid patch — the same guarantee `apply_to_uniforms` already enforces.

```
collide-o-scope walk --anchor patches/anchor.yaml \
                     --temperature 0.5 \
                     --count 10 \
                     --out-dir patches/walked/
```

Writes N perturbed YAML patches (or renders them directly — see `batch`).

### 1.5 Duration roll + random video selection

The `[min, max]` convention from the legacy `DEFAULTS` becomes one helper:

```rust
fn roll_range(rng: &mut StdRng, lo: f32, hi: f32) -> f32 {
    rng.gen_range(lo..=hi)
}
let duration = roll_range(&mut rng, 60.0, 180.0); // the locked 60–180s window
```

Video selection mirrors `multi-layer.py::select_videos`: scan the library folder, `rng`-sample however many layers the patch wants, and (optionally) roll a random start offset per clip. The patch's `LayerConfig.filename` fields get overwritten with the sampled names before rendering.

### 1.6 The batch subcommand (ties it together)

`batch` is what `autopilot.py` actually calls. One invocation = one finished video:

```
collide-o-scope batch --library videos/ \
                      --anchors patches/ \
                      --out-dir output/ \
                      --temperature 0.5 \
                      --seed 1234
```

Steps, all seeded off `--seed` so the whole thing is reproducible:

1. Pick a random anchor patch from `--anchors`.
2. `walk()` it by `--temperature`.
3. Sample random videos from `--library` into the layer slots.
4. Roll a duration in `[60, 180]`.
5. (later) generate a markov title; write it into ffmpeg metadata like `multi-layer.py` does.
6. `run_export` → write `output/<title-or-seed>.mp4`.
7. (optional) append a line to a `render-log.jsonl` so source-usage tracking still works.

Match the output **filename and folder** convention `autopilot.py` expects (an `.mp4` in the project's `output/` dir), because that's how it counts "ready vs uploaded."

---

## Part 2 — Server / autopilot reuse

### 2.1 What stays exactly as-is

Everything in `autopilot.py` except one function:

- `acquire_lock` (fcntl) — prevents hourly cron from stacking when a render runs long.
- `maybe_upload` / `_upload_file` — organic timing math, the upload window, daily quota, `youtube-upload.py --file --public`, manifest append, delete-after-upload.
- `maybe_refresh_sources` — source pool rotation.
- The whole comment-feedback path (off by default).

The YouTube OAuth story is unchanged: `client_secret.json` + cached token on the VPS, exactly as it works today.

### 2.2 The one change: `maybe_render`

Today `maybe_render` (`autopilot.py:646`) builds a subprocess command pointing at `scripts/blend/multi-layer.py`:

```python
cmd = [
    sys.executable,
    os.path.join(PROJECT_ROOT, 'scripts', 'blend', 'multi-layer.py'),
    '--preset', rhythm['render_preset'],
    '--project', project_name,
    '--output-dir', output_dir,
]
```

Swap it for the Rust binary's `batch` subcommand:

```python
cmd = [
    os.path.join(PROJECT_ROOT, 'bin', 'collide-o-scope'), 'batch',
    '--library', library_dir,
    '--anchors', anchors_dir,
    '--out-dir', output_dir,
    '--temperature', str(rhythm.get('temperature', 0.5)),
    '--seed', str(seed_val),     # autopilot already computes seeds
]
```

That's it. `subprocess.run(cmd, check=True, timeout=900)` stays. The seed plumbing (including `--seed` from comment hashes) already exists — the Rust binary just needs to accept `--seed`. Drop a couple of new keys into `rhythm.json` (`temperature`, `anchors_dir`) and they flow through `RHYTHM_DEFAULTS` the same way `render_preset` does.

### 2.3 Deployment gotcha: headless GPU on a VPS

This is the one thing that can bite. `run_export` asks wgpu for a real adapter. On your laptop that's Metal. On a cloud VPS there is usually **no GPU**, so wgpu has to fall back to a software rasterizer or it fails to find an adapter. Three options, in order of preference:

1. **A GPU instance** (most cloud providers offer small ones). Cleanest — same code path as local.
2. **Vulkan software rasterizer** (`lavapipe` / Mesa's `llvmpipe`). Install Mesa, force the Vulkan backend, accept that it's slow. Fine for 60–180s clips rendered hours apart.
3. **Render locally, upload from the VPS.** Keep rendering on your Mac, `rsync` finished MP4s to the VPS, and let `autopilot.py` run with `render_when_below` effectively disabled so it only uploads. Lowest risk if GPU-on-server turns into a yak shave.

Also required on the VPS regardless: `ffmpeg` on `PATH` (the renderer shells out to it), and the video library + anchor patches present. Decide #1 vs #2 vs #3 early — it shapes everything else.

---

## Build order

Each step is independently useful and testable:

1. **`render` subcommand** — patch file in, MP4 out, synchronous. Proves the headless CLI entry works end-to-end with code that already exists (`run_export`).
2. **Add `rand` + `walk`** — anchor + temperature → N patches. Eyeball them in the GUI before rendering.
3. **`batch`** — random anchor + walk + random videos + duration roll → one MP4 with the right filename convention.
4. **Swap `maybe_render`** — point autopilot at the Rust binary. Run `--dry-run` first, then `--status`.
5. **Pick the deployment model** (GPU / lavapipe / render-local) and stand it up on the VPS under cron.
6. **(later)** markov titles + metadata, so filenames and YouTube titles stop being seeds.

---

## Open questions

- **Where do anchor patches come from on the server?** The blog's "quick-save button" builds the corpus locally; those YAML files need to ship to the VPS (commit them, or `rsync`). Does autopilot ever generate fresh anchors, or only walk the curated set?
- **One temperature, or a distribution?** A fixed `0.5` is predictable; rolling temperature in `[0.2, 1.2]` per render gives more variety but occasional duds. The upload window + manual culling may make duds cheap enough not to care.
- **Title generation in Rust or Python?** The markov titler is tiny either way. If Rust owns it, the filename can *be* the title and Python passes it straight to `--title`. If Python owns it, `maybe_render`/`_upload_file` stay closer to today's flow.
- **Render-log compatibility.** `maybe_refresh_sources` reads `render-log.jsonl` for source-usage retirement. If you want that feature, the Rust `batch` must emit the same `{sources: [{file: ...}]}` shape.
