# changelog

2026-02-09
- added README.md to each directory — library, presets, projects, scripts, tools, context. each explains what the directory is for, what goes there, and how it connects to the rest of the project
- added .cursor/rules/voice-and-tone.mdc — codifies the writing style used across the project: lowercase everything, say less, explain the why not the what, talk like a colleague. applies to docs, comments, changelogs, and generated text
- changed embedded MP4 artist metadata from "collide-o-scope" to "luis queral"
- renamed project from video-synth to collide-o-scope — updated directory name, all internal references in ARCHITECTURE.md, README.md, multi-layer.py (artist metadata tag), and scripts/utils/__init__.py
- wired metadata generator into multi-layer.py render pipeline — each video now gets a cryptic title and description embedded directly in the MP4 container via ffmpeg -metadata flags (title, comment, artist). added get_metadata() to scripts/utils/ffprobe.py to read metadata back from any MP4 using ffprobe, so the bulk uploader can scan a directory and build a queue from the embedded data
- added variable length to titles and descriptions — titles now range from single words ("rust", "dense") through codes ("#8968") to full phrases ("extracted the chalk, flattened the lacquer, inverted the sync"), and descriptions range from one word ("faint") to multi-sentence sprawl with chains and fragments woven together. new title generators: single_word, long_fragment, chain, just_code
- removed all technical/session data from description generator — descriptions are now purely observational, cryptic, and markov-chain-based. no fps, no duration, no mode info, no seed numbers, no layer counts. just vibes
- added predictive text chain generator (markov chain) to scripts/text/metadata.py — hand-curated word-to-word transition table that produces iPhone-middle-button-mash-style text ("camera left overnight it loops without beginning and left behind it"). works standalone via --chain flag or importable as generate_chain(). ~90 words in the transition table, weighted toward experimental film / found footage vocabulary
- built cryptic title/description generator in scripts/text/metadata.py — produces gallery-placard-style titles ("nothing split, nothing copied", "vestibule #8203", "only the chalk remained") and fragmented descriptions with observational and cryptic one-liners. weighted title patterns, multiple separator styles, importable as module or standalone CLI
- batch rendered 10 more blend videos in projects/first-blend-test/output/ using classic-white preset — each with a unique auto-generated seed producing different layer combinations, audio mixes, and stereo panning

2026-02-08
- ported blend-video-alt.py to scripts/blend/multi-layer.py using the new architecture: DEFAULTS dict with full comments, argparse CLI (--preset, --mode, --fps, --num-videos, --seed, etc.), imports from shared utils, config layering (defaults → preset → CLI)
- all four color modes (fixed, kmeans, luminance, rembg) working inline in multi-layer.py — will be extracted to scripts/blend/modes/ in phase 5
- audio pipeline (multi-source mixing, stereo panning, loudness normalization) working inline — will be extracted to scripts/audio/ in phase 6
- added presets: kmeans-default.json and luminance-auto.json alongside classic-white.json
- scaffolded project structure: scripts/, presets/, projects/, library/, tools/ directories with README, requirements.txt, .gitignore
- built shared utils foundation in scripts/utils/ — four modules extracted from existing blend scripts:
  - ffprobe.py: video metadata (duration, resolution, fps, audio detection, volume analysis)
  - cache.py: JSON metadata cache for the video library (build, load, save, prune stale entries)
  - random.py: seeded PRNG wrapper with from_range() for resolving [min, max] config values
  - config.py: layered config merging (defaults → preset → CLI) with project-level preset shadowing
- created first preset: presets/blend/classic-white.json
- copied camera-search.html into tools/
- created ARCHITECTURE.md capturing the full project plan: directory structure, script patterns, preset system, session workflow, implementation phases, and LLM guidance for working in the codebase
- decided to stay python-centric (not migrate to Node/TypeScript) after evaluating tradeoffs — ffmpeg/ML ecosystem is python-native, and the media-tool-kit script-based pattern fits the creative workflow better
- outlined 9-phase implementation plan from scaffold through tooling, with shared utils as the foundation everything builds on
