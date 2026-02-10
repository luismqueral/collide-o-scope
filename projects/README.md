# projects

each body of work gets its own folder. a project is a context — it has its own inputs, outputs, presets, and optionally its own scripts.

## structure

```
projects/
└── [project-name]/
    ├── input/          # source material specific to this project
    ├── output/         # rendered videos (gitignored)
    ├── debug/          # debug frames, palettes, masks (gitignored)
    ├── scripts/        # one-off scripts only for this project
    ├── presets/        # project-level preset overrides
    │   └── blend/
    ├── config.json     # settings from last run
    └── manifest.json   # full session record (sources, seeds, trims)
```

## what goes here vs. in scripts/

if a script is reusable across projects, it lives in `scripts/`. if it's a weird one-off experiment that only makes sense for this specific project, it lives in `projects/[name]/scripts/`.

## outputs

rendered videos and debug frames are gitignored. the metadata (title, description, artist) is embedded in each MP4 container, so the video file itself is the record.
