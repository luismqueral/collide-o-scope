# collide-o-scope architecture

a personal framework for generative video synthesis. python scripts, ffmpeg under the hood, organized for both human and LLM-assisted workflows.

---

## philosophy

this is a creative tool, not a product. the priorities are:

1. **scripts are instruments** — each one does a specific thing well, accepts input, produces output
2. **projects are contexts** — each body of work gets its own folder with its own inputs, outputs, presets, and one-off scripts
3. **presets are saved knob positions** — they're data, not logic. flat dictionaries of values
4. **ffmpeg is the engine** — python orchestrates command-line tools, it doesn't replace them
5. **reproducibility by default** — every run records what it did so it can be replayed

---

## directory structure

```
collide-o-scope/
│
├── scripts/                        # reusable tools (the shared toolkit)
│   ├── blend/                      # video compositing
│   │   ├── multi-layer.py          # N-video overlay with colorkey
│   │   └── modes/                  # color keying strategies
│   │       ├── __init__.py
│   │       ├── fixed.py            # hardcoded color keying
│   │       ├── kmeans.py           # ML dominant color extraction
│   │       ├── luminance.py        # brightness-based keying
│   │       └── rembg.py            # ML background removal keying
│   ├── audio/                      # audio processing
│   │   ├── mix.py                  # multi-track mixing + stereo panning
│   │   ├── normalize.py            # loudness normalization
│   │   └── extract.py              # pull audio from video files
│   ├── post/                       # post-processing effects
│   │   ├── analog.py               # grain, breathing, color drift, vignette
│   │   ├── color-grade.py          # temperature, contrast, saturation, curves
│   │   └── stabilize.py            # optical flow smoothing
│   ├── source/                     # source material management
│   │   ├── scan-library.py         # build/update video metadata cache
│   │   ├── download.py             # yt-dlp wrapper for scraping
│   │   └── filter.py               # HD filtering, deduplication
│   └── utils/                      # shared infrastructure
│       ├── __init__.py
│       ├── ffprobe.py              # video metadata (duration, resolution, audio)
│       ├── cache.py                # JSON metadata cache read/write
│       ├── random.py               # seeded PRNG wrapper
│       └── config.py               # preset loading + merging
│
├── presets/                        # universal presets (organized by script type)
│   ├── blend/
│   │   ├── classic-white.json
│   │   ├── kmeans-default.json
│   │   └── luminance-auto.json
│   ├── post/
│   │   ├── vintage-film.json
│   │   └── high-contrast.json
│   └── audio/
│       └── wide-stereo.json
│
├── projects/                       # synthesis sessions and bodies of work
│   └── [project-name]/
│       ├── input/                  # source material for this project
│       ├── output/                 # rendered results
│       ├── debug/                  # debug frames, palette images, masks
│       ├── scripts/                # one-off scripts specific to this project
│       ├── presets/                # presets specific to this project
│       │   └── blend/
│       ├── config.json             # settings used for last run
│       ├── manifest.json           # full record (sources, seeds, colors, trims)
│       └── README.md               # studio notes — what you were going for
│
├── library/                        # raw material archive
│   ├── video/                      # scraped youtube footage
│   └── audio/                      # standalone audio sources
│
├── tools/                          # standalone utilities
│   ├── batch-render.py             # run N synthesis sessions sequentially
│   ├── save-preset.py              # promote manifest settings to a preset
│   └── camera-search.html          # DCF pattern generator for youtube search
│
├── context/                        # reference material (original python scripts)
│
├── video_cache.json                # library metadata cache
├── requirements.txt
├── .gitignore
├── CHANGELOG.md
└── README.md
```

### where things go

| you want to... | it goes in... |
|---|---|
| add a reusable tool | `scripts/[category]/` |
| try something experimental for one project | `projects/[name]/scripts/` |
| save settings you like for any project | `presets/[category]/` |
| save settings you like for one project | `projects/[name]/presets/[category]/` |
| add source footage | `library/video/` |
| add audio sources | `library/audio/` |
| build a standalone utility | `tools/` |

### when a one-off becomes reusable

if a script in `projects/[name]/scripts/` turns out to be useful beyond that project, move it to `scripts/[category]/` and generalize it. that's the natural lifecycle: experiment in a project, graduate to the shared toolkit.

---

## how scripts work

### every script follows the same pattern

```python
"""
script-name.py - One Line Description

Longer explanation of what this does and when you'd use it.

USAGE:
  python scripts/category/script-name.py -i input.mp4 -o output.mp4
  python scripts/category/script-name.py --preset my-preset
"""

from scripts.utils.config import load_config
from scripts.utils.ffprobe import get_video_duration
from scripts.utils.random import create_rng
import argparse

# =============================================================================
# DEFAULTS
#
# What this section controls and why you'd change it.
# Any value here can be overridden by a preset or CLI flag.
#
# Ranges like [min, max] = "pick randomly within this window each run."
# Single values = "use exactly this."
# =============================================================================

DEFAULTS = {

    # --- section name ---

    # what this knob does
    # what low values look like vs high values
    # what range is typical
    "knob_name": default_value,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', default=None)
    parser.add_argument('--project', default=None)
    # ... script-specific args
    args = parser.parse_args()

    config = load_config(DEFAULTS, preset_name=args.preset, preset_category='blend', project_dir=args.project)
    # ... do the work

if __name__ == "__main__":
    main()
```

### config resolution order

each layer only overrides what it specifies. unset values carry through from the layer below.

```
  code DEFAULTS          what the script uses if you say nothing
       │
       ▼
  preset file            --preset name (project-level checked first, then universal)
       │
       ▼
  CLI flags              --fps 18 --num-videos 5
       │
       ▼
  final config           what actually runs
```

### preset lookup

when `--preset neon-mask` is passed:

1. if `--project dark-city-loop` is also passed, check `projects/dark-city-loop/presets/[category]/neon-mask.json`
2. then check `presets/[category]/neon-mask.json`
3. not found → error

project presets shadow universal ones.

### what a preset file looks like

```json
{
  "name": "neon-mask",
  "mode": "kmeans",
  "num_videos": 4,
  "fps": 24,
  "similarity": [0.25, 0.40],
  "saturation": [1.1, 1.5],
  "hue_shift": [-30, 30]
}
```

only the values being overridden. no comments needed — the script's DEFAULTS block is the documentation.

---

## session workflow

what happens when you trigger a synthesis run:

```
trigger
  python scripts/blend/multi-layer.py --preset kmeans-default
       │
       ▼
1. CONFIGURE ─── merge defaults → preset → CLI flags into final config
       │
       ▼
2. INITIALIZE SESSION ─── generate seed, create project folder, write config.json
       │
       ▼
3. SELECT SOURCES ─── scan library (cached), filter by resolution/audio, random sample
       │
       ▼
4. ANALYZE ─── per-video: extract frame → run color mode → get key colors
       │         modes: fixed | kmeans | luminance | rembg
       ▼
5. BUILD FILTER CHAIN ─── construct ffmpeg filter_complex string
       │                   trim → loop → color correct → colorkey → scale → overlay
       ▼
6. AUDIO ─── select tracks from sources, trim/loop, pan stereo, normalize, mix
       │
       ▼
7. RENDER ─── execute single ffmpeg command → output video
       │
       ▼
8. POST-PROCESS (optional) ─── chain: analog → color-grade → stabilize
       │                        each reads a video, writes a new one
       ▼
9. RECORD ─── write manifest.json with full session details for reproducibility
```

### manifest.json

the manifest is the complete record of a session. it captures everything needed to reproduce the output exactly:

```json
{
  "seed": 1738942817,
  "timestamp": "20260208_143022",
  "config": { "...merged config that was used..." },
  "sources": [
    {
      "path": "library/video/DSC_0047.mp4",
      "trim_start": 47,
      "duration": 180,
      "resolution": "1920x1080",
      "has_audio": true
    }
  ],
  "colorkeys": [
    {
      "layer": 1,
      "mode": "kmeans",
      "colors": ["0xA3B2C1", "0x4F6B3A"],
      "similarity": 0.31,
      "blend": 0.02
    }
  ],
  "audio": {
    "sources": [0, 2],
    "panning": [-0.4, 0.6],
    "normalized": true
  },
  "post": ["analog", "color-grade"],
  "output": "projects/dark-city-loop/output/output_20260208_143022.mp4"
}
```

replay with: `python scripts/blend/multi-layer.py --from-manifest projects/dark-city-loop/manifest.json`

---

## commenting style

### DEFAULTS blocks

every value gets a comment explaining what it does in artistic/practical terms, not technical terms. include what low vs high values look like and what range is typical.

```python
DEFAULTS = {
    # how close a pixel must be to the key color to become transparent
    # lower = stricter (less transparency), higher = looser (more transparency)
    # [min, max] → randomized per run for variety
    "similarity": [0.2, 0.4],

    # output frame rate
    # 12-18 = dreamy/choppy, 24 = film, 30 = video standard
    "fps": 30,
}
```

### inline comments in scripts

explain the WHY, not the WHAT. assume the reader understands python but not ffmpeg filter chains or video processing concepts.

---

## implementation plan

### phase 1: scaffold
create directory structure, README, requirements.txt, .gitignore. no code.

### phase 2: shared utils
extract `ffprobe.py`, `cache.py`, `random.py`, `config.py` from existing scripts into `scripts/utils/`. this is the foundation everything imports from.

### phase 3: first blend script
port `blend-video-alt.py` to `scripts/blend/multi-layer.py`. replace globals with DEFAULTS, import from utils, add argparse. color keying logic stays inline for now. goal: same output as the old script, new structure.

### phase 4: preset system
implement config loading (defaults → preset → CLI). create starter presets in `presets/blend/`. add `--preset` flag. build `tools/save-preset.py`.

### phase 5: color modes
extract kmeans, luminance, rembg, fixed into `scripts/blend/modes/`. each exports `get_colors(video_path, config) → [(color, similarity, blend)]`. pure refactor, no new behavior.

### phase 6: audio pipeline
extract audio mixing into `scripts/audio/`. works standalone (`python scripts/audio/mix.py -i video.mp4`) and as import from blend script.

### phase 7: post-processing
port analog-processor.py, color grading, stabilization to `scripts/post/` with own DEFAULTS and preset support.

### phase 8: session management
auto-create project folders, write config.json and manifest.json per run, implement `--from-manifest` replay.

### phase 9: tooling
batch-render.py, save-preset.py, camera-search.html, standalone library scanner.

### dependency graph

```
phase 1 ─── scaffold
  │
phase 2 ─── shared utils
  │
phase 3 ─── first working blend script
  │
  ├── phase 4 ─── presets
  ├── phase 5 ─── color modes (refactor)
  ├── phase 6 ─── audio (extraction)
  └── phase 7 ─── post-processing
      │
phase 8 ─── session management
      │
phase 9 ─── tooling
```

phases 4-7 are independent of each other. phase 8 needs most of them. phase 9 is polish.

---

## LLM guidance

this section describes how an LLM working in this codebase should behave. it's written to be adapted into a system prompt, cursor rule, or similar.

### understanding the project

this is a personal creative tool for generative video art. the user creates video compositions by layering scraped youtube footage and applying color keying (making certain colors transparent so lower layers show through). the core engine is ffmpeg. python orchestrates it.

the codebase is organized as independent scripts that share utilities. it is NOT a library, framework, or application. there is no build step, no type system, no package publishing. scripts are run directly with `python scripts/[category]/script.py`.

### when creating new scripts

1. **follow the script template exactly.** every script starts with a docstring, imports from `scripts/utils/`, defines a `DEFAULTS` dict, uses `argparse`, and calls `load_config()`. no exceptions.

2. **DEFAULTS are the documentation.** comment every value with what it does artistically, not technically. include what low vs high values look like. include typical ranges. group by concern with `# --- section name ---` separators.

3. **decide where it lives:**
   - reusable across projects → `scripts/[category]/`
   - specific to one project → `projects/[name]/scripts/`
   - standalone utility → `tools/`

4. **one script, one job.** a blend script blends. an audio script handles audio. a post script applies effects. don't combine concerns. if you need to chain operations, call scripts sequentially or build a project-specific orchestration script.

5. **always import from utils.** don't inline ffprobe calls, cache logic, or config loading. use the shared modules:
   - `from scripts.utils.ffprobe import get_video_duration, get_video_resolution, video_has_audio`
   - `from scripts.utils.cache import build_cache`
   - `from scripts.utils.random import create_rng`
   - `from scripts.utils.config import load_config`

6. **ffmpeg is the engine.** build ffmpeg commands as lists of strings and run them with `subprocess.run()`. don't use ffmpeg wrapper libraries. the user needs to see and understand the exact ffmpeg command being constructed.

### when modifying existing scripts

1. **don't change the interface.** if a script accepts `--preset` and `-i` and `-o`, keep those flags. add new flags, don't rename or remove existing ones.

2. **don't move DEFAULTS values to a central location.** each script owns its defaults. the shared `config.py` handles merging logic only, not values.

3. **if adding a new config value,** add it to DEFAULTS with a comment following the existing style. if it's a range, use `[min, max]` list format.

### when working with presets

1. **presets are flat dictionaries.** no logic, no conditionals, no inheritance. just key-value pairs that override DEFAULTS.

2. **presets go in the right place:**
   - universal → `presets/[category]/name.json`
   - project-specific → `projects/[name]/presets/[category]/name.json`

3. **preset keys must match DEFAULTS keys.** don't invent new keys in a preset that the script doesn't understand.

### when creating projects

1. **use the standard structure:**
   ```
   projects/[name]/
   ├── input/
   ├── output/
   ├── debug/
   ├── scripts/      (only if needed)
   ├── presets/       (only if needed)
   ├── config.json
   ├── manifest.json
   └── README.md
   ```

2. **README.md is studio notes.** write what the project is about, what techniques worked, what to revisit. informal tone.

3. **don't create empty directories preemptively.** only create `scripts/` and `presets/` inside a project when there's actually something to put there.

### general principles

- **python, not node.** this is a python-centric workspace. scripts use ffmpeg, imagemagick, and sox via subprocess. ML dependencies (numpy, sklearn, PIL, rembg) are python-native.

- **no frameworks.** no flask, no fastapi, no django. scripts are standalone. if a web UI is added later, it lives in `_app/` and calls into scripts.

- **flat over nested.** avoid deep directory hierarchies. one level of nesting under `scripts/` is usually enough.

- **working code over clean code.** this is a creative tool. a script that produces interesting output is more valuable than a beautifully architected script that doesn't run yet.

- **the context/ folder is read-only reference.** it contains the original python scripts. don't modify them. refer to them when porting to the new structure.

- **always update CHANGELOG.md** after making changes. entries should be plain english, most recent on top, grouped by date.
