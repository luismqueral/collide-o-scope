# presets

saved knob positions. each preset is a flat JSON file that overrides default values in a script.

## how they work

every script has a `DEFAULTS` dict at the top with every parameter it uses, commented with what each one does and why you'd change it. a preset overrides some of those defaults. anything not in the preset stays at the script's default.

```
script defaults → preset overrides → CLI overrides
```

so a preset is just the middle layer. you can always override a preset value from the command line.

## structure

organized by script type:

```
presets/
├── blend/          # for scripts/blend/multi-layer.py
├── post/           # for scripts/post/ (analog, color-grade, etc)
└── audio/          # for scripts/audio/ (mix, normalize, etc)
```

## example

`blend/classic-white.json` overrides the color mode to fixed white keying:

```json
{
  "name": "classic-white",
  "mode": "fixed",
  "colorkey_hex": "0xFFFFFF",
  "similarity": [0.25, 0.35]
}
```

`[min, max]` values get randomized per run within that range. single values are used as-is.

## project-level presets

projects can also have their own presets in `projects/[name]/presets/`. these take priority over universal presets with the same name, so a project can shadow a global preset without modifying it.
