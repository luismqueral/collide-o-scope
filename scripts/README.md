# scripts

the shared toolkit. reusable instruments organized by what they do.

## layout

```
scripts/
├── blend/          # video compositing (multi-layer.py is the main one)
│   └── modes/      # color keying strategies (fixed, kmeans, luminance, rembg)
├── audio/          # mixing, panning, normalization, extraction
├── post/           # analog effects, color grading, stabilization
├── source/         # library scanning, downloading, filtering
├── text/           # metadata generation (titles, descriptions, markov chains)
└── utils/          # shared infrastructure (ffprobe, cache, config, seeded rng)
```

## how scripts work

each script follows the same pattern:

1. `DEFAULTS` dict at the top — every parameter, fully commented
2. `argparse` CLI — `--preset`, `--project`, plus script-specific flags
3. imports from `scripts.utils` for config loading, ffprobe, caching, rng
4. config resolved as: defaults → preset → CLI overrides

## running a script

```bash
python scripts/blend/multi-layer.py                        # all defaults
python scripts/blend/multi-layer.py --preset classic-white  # with preset
python scripts/blend/multi-layer.py --fps 24 --num-videos 5 # CLI overrides
```
