# changelog

2026-02-18
- added `dominant` color mode to `scripts/blend/multi-layer.py` — keys out the most frequent color in each video per-layer, no external deps. uses ffmpeg to sample frames at low res, quantizes pixels, and counts. each layer gets its own key color based on what's actually in the footage (greens from foliage, near-black from shadows, grays from sky, etc). added `dominant-auto` preset in `presets/blend/` and rendered test videos in alt-blend-test
- fixed corrupt output audio — mixed sample rates (44.1kHz aac + 48kHz opus) were producing 96kHz output that most players and youtube can't handle. added `aresample=48000` to the audio chain in `scripts/blend/multi-layer.py` so everything lands at a standard rate
- rendered 3 more 15-second test videos in alt-blend-test with the audio fix, all confirmed 48kHz
- created `projects/alt-blend-test/` as a second project using the new library — rendered the first video with the `kmeans-default` preset (4 layers, 109s). kmeans fell back to fixed keying since numpy/sklearn aren't installed locally, but the structure is ready for real color analysis once those deps are in
- renamed `tools/test-source-download.py` to `tools/download-source-material.py` so the command reads like the job it does and is easier to remember in day-to-day sourcing runs
- optimized `scripts/blend/multi-layer.py` to use 12x less memory (12.5GB -> 0.98GB peak) and render 4x faster by replacing the `loop` filter (which pre-buffered thousands of frames in RAM) with ffmpeg's `-stream_loop` input option (re-reads from disk, near-zero memory). also removed `aloop` from the audio chain (redundant now) and disabled `normalize` by default (3.5x slower with no memory benefit). renders now complete on the 4GB server without crashing
- added 4GB swap to the hetzner VPS and set `output_size` to 1280x720 in the `classic-white` preset as belt-and-suspenders for the memory-constrained server
- replaced the entire server source library (485 old videos) with 44 fresh downloads from `library/video/` — cleared the video cache so the renderer rebuilds on next run
- stripped the probability math out of `maybe_upload` in `tools/autopilot.py` — each tick now just uploads 1 video if under the daily cap, no dice rolls or jitter. the cron schedule is the timing (reverted — keeping organic logic)
- fixed a bug where autopilot renders were going to `projects/archive/output/` instead of the active project's output dir — the `--project` flag only affected preset lookup, not the output path. added explicit `--output-dir` to both seeded and regular render calls in `tools/autopilot.py`
- moved 1682 orphaned renders from `archive/output/` to `first-blend-test/output/` on the server so the upload pipeline can see them
- rebooted the server after sshd became unresponsive — uploads had been stalled for ~2 days because of this + the output dir mismatch

2026-02-15
- hardened `tools/test-source-download.py` to prevent duplicate ingest and bad media drift — it now skips videos already present in `library/video` or `library/video/_archive` (by youtube id) and verifies each new download with ffprobe/size checks before keeping it
- ran another random source-ingest pass for 50 downloads to keep momentum after the first archive split — all pulls succeeded and expanded the active blend-ready pool in `library/video`
- ran a bulk source-ingest test with `tools/test-source-download.py` to pull 25 random camera-pattern youtube videos and verify the downloader path under heavier load
- reorganized `library/video/` to match the active blending convention — kept only newly pulled candidates in the root and moved all previous source videos into `library/video/_archive` for preservation without blending pickup
- added `tools/test-source-download.py` as a tiny source-ingest test path so we can validate discovery ideas quickly — it supports either a direct `--url` download or a `--random` camera-pattern search pick, then downloads with `yt-dlp` into `library/video`
- updated `tools/README.md` to document the new test downloader and why it exists, so the source-material workflow is easier to run without digging through scripts
- added `tools/upload-draft.py` to force a one-off youtube upload as private (draft-like) from a project output folder, so upload checks can run on demand without waiting for autopilot timing
- set safer defaults in the draft uploader — it prefers files not marked uploaded in `upload-manifest.json`, supports `--dry-run`, and can target a specific file when needed
- updated `tools/README.md` to list active utilities including the new draft uploader so the ops path is easier to find
- added a local-only server access runbook in `server-access.local.md` with exact ssh/provision/sync/cron steps, plus server origin details (hetzner CX23 in helsinki) so server handoffs are documented in one place without exposing secrets in the repo
- updated `.gitignore` to ignore `server-access.local.md` so private access notes stay uncommitted by default
- added comment-driven title generation for new renders — `tools/autopilot.py` now pulls youtube comments from videos uploaded in the last 90 days and passes a comment into each render so new titles can absorb audience language
- updated `scripts/blend/multi-layer.py` with a `--title-comment` input that steers only the title pattern (description flow stays the same), so this can be used by autopilot without changing the rest of the render pipeline
- enabled the feature in `projects/first-blend-test/rhythm.json` with `title_from_comments: true` and a 90-day lookback window, so the current project starts using recent channel comments immediately
- fixed comment fetch filtering in autopilot to apply likes threshold correctly after fetch, avoiding a broken argument path that prevented reliable comment intake

2026-02-12
- added --comment-title flag to tools/upload-draft.py — fetches a random comment from published videos and uses it as the video title. the cryptic metadata description stays underneath. trimmed to youtube's 100-char limit. falls back gracefully if no comments exist yet. also added --title override to youtube-upload.py to support this
- committed and pushed comment feedback loop + autopilot pipeline to origin, pulled latest on the VPS (77.42.87.199). initialized git tracking on the server (was previously rsync-only) so future deploys are just `git pull`
- synced full video library to server — 431 new clips transferred (~8 GB), server now has all 492 source videos matching local
- rendered 20 new videos with classic-white preset, 1-2 min duration range — output in projects/archive/output/. changed default duration from [60, 180] to [60, 120] for tighter outputs
- expanded the video library from 61 to 492 clips — pulled 431 new videos from the archive, prioritizing 162 HD clips (720p to 4K) alongside 269 SD clips. cleared the video cache so it rebuilds on next render
- built the comment feedback loop — audience comments now feed back into the video-making pipeline across four dimensions: seeding renders, burning text into video, driving polls, and generating response videos. the audience shapes the work without knowing it
- added scripts/youtube/comments.py — fetches comments from published videos via the YouTube Data API. uses the same OAuth credentials as the upload scripts. returns structured data (text, author, likes, video_id) and can pull from a single video or scan recent uploads via the manifest. includes comment_to_seed() which hashes comment text into a deterministic render seed via sha256
- added scripts/youtube/polls.py — generates soft poll questions that map to real config knobs ("next: lights or darks?" -> luminance_target, "faster or slower?" -> fps, "vivid or muted?" -> saturation, etc). reads responses via simple keyword matching and returns config overrides. seven poll types covering mode, layers, fps, keying tightness, duration, saturation, and color method
- added scripts/post/burn-comments.py — overlays comment text into videos using ffmpeg drawtext. style is deliberately lo-fi: low opacity (0.12-0.28), monospace, small, randomized position, with fade in/out. each comment appears for 3-8 seconds. comments from the previous burst's videos get burned into the next batch
- added response and seed-credit generators to scripts/text/metadata.py — generate_response_title() and generate_response_description() absorb words from a comment into the existing pattern system (the comment gets digested, not quoted). generate_seed_description() weaves a hint that the video was shaped by a comment. new CLI flags: --response and --seed-desc
- wired comment feedback into autopilot.py — rendering phase now fetches comments before rendering, reads poll results for config overrides, renders comment-seeded videos (up to 20% of batch), then post-processes new videos with comment burning and response metadata. uploading phase attaches poll questions to some uploads and records them in polls-state.json for next-cycle reading
- added --poll-text flag to youtube-upload.py — the autopilot passes poll questions through to the upload script which appends them to the video description
- updated rhythm.json for first-blend-test with comment feedback config — comment_feedback: true, seed_from_comments, burn_comments, burn_comment_chance (0.4), poll_chance (0.25), response_chance (0.12), min_comment_likes (0), comment_lookback_videos (10)
- added polls-state.json to .gitignore — per-machine state that tracks which videos got poll questions
- added delete_after_upload to autopilot — videos are automatically deleted from disk after successful upload to youtube. no reason to hoard gigabytes of MP4s that are already online. controlled by rhythm.json (on by default). also manually deleted 32 previously-uploaded videos, freeing ~7.7GB
- wired title_from_comments into the autopilot upload loop — when enabled in rhythm.json, the autopilot fetches recent youtube comments and uses a random one as the video title. falls back to generated metadata when no comments exist. controlled by title_from_comments, title_comment_strategy, and comment_lookback_videos in rhythm.json
- deleted all 32 previously scheduled videos from youtube — clean slate for direct public uploads
- stripped autopilot.py down to its simplest form — no phases, no bursts, no quiet periods. every hourly tick: render if the pool is low, roll dice to maybe upload 1-2 videos as public. 6/day at random times within a configurable window. the organic feel comes from per-tick probability math (remaining uploads / hours left, ±30% jitter). rhythm.json is now just window_hours, uploads_per_day, render config, and delete_after_upload
- rewrote autopilot.py to upload directly as public — killed the scheduling phase entirely. no more private+publishAt dance. three phases now (quiet/rendering/uploading) instead of four. organic timing comes from per-tick probability: each hourly cron tick rolls dice based on remaining daily quota vs hours left in the window. early ticks are unlikely to fire, late ones feel the pressure. the audience sees someone posting when they feel like it
- added --public and --file flags to youtube-upload.py — --public uploads as public immediately (no publishAt), --file uploads a single MP4 directly without needing a schedule.json. the autopilot uses both for direct organic uploads
- replaced schedule.json tracking with upload-manifest.json — simpler tracking file that just records filename, video_id, timestamp, and url. lives at projects/PROJECT_NAME/upload-manifest.json, gitignored
- removed burst_density from rhythm.json — no longer needed since we're not pre-scheduling publish dates. the upload density is now emergent from the per-tick probability math
- updated youtube-publishing.mdc cursor rule to document the new direct-upload workflow, manifest-based tracking, and per-tick timing math
- added tools/dashboard.py — a lightweight web dashboard that shows autopilot phase, upload progress, pending schedule, recent uploads with youtube links, rhythm config, and tail of the autopilot log. auto-refreshes every 60s. runs as a systemd service on port 8080
- provisioned hetzner CX23 VPS in helsinki (77.42.87.199, ~$4/mo) — synced codebase, video library, and youtube credentials. autopilot cron running hourly, uploading 6 videos/day from the first burst of 28 pending videos

2026-02-11
- built tools/autopilot.py — a state machine daemon for organic burst publishing. cycles through quiet/rendering/scheduling/uploading phases with randomized timing so the channel feels like a person, not a bot. runs via hourly cron on a VPS. each project gets a rhythm.json that defines the posting personality (burst size, density, cooldown, solo post chance)
- added scripts/upload/youtube-delete.py — delete videos from youtube by ID or by filtering schedule.json entries by duration. --clean-schedule removes deleted entries from the schedule file. used to clean up 14 accidental 15-second uploads
- upgraded youtube OAuth scope from youtube.upload to full youtube — needed for delete capability. backward-compatible, just required one re-auth
- deleted 14 accidentally-uploaded 15-second videos from youtube and cleaned them from schedule.json and disk
- scheduled first organic burst: 34 videos across Feb 17-21 at 8-9/day with irregular times, first 6 uploaded today
- added tools/setup-server.sh — provisioning script for a fresh ubuntu VPS (hetzner CX22 recommended, ~$4/mo). installs system deps, clones repo, sets up python packages, prints cron setup instructions
- added projects/first-blend-test/rhythm.json — defines the posting personality for the first batch: 10-25 video bursts, 8-15/day density, 4-21 day cooldowns, 5% solo post chance during quiet periods
- added autopilot-state.json to .gitignore — per-machine state that tracks current phase and transition times
- updated .cursor/rules/youtube-publishing.mdc to cover the full autopilot workflow: phases, rhythm config, cron setup, manual overrides, and the new delete script

2026-02-10
- added preview GIF band to README — extracted 3-second clips from four output videos into docs/preview/, displayed as a horizontal row under the project description so you can see what the thing actually produces
- rewrote schedule.py to support multiple posts per day — new --per-day and --window flags replace the old interval-based scheduling. --per-day 2 --window 08:00-17:00 posts twice daily at random times within the window. actual daily count varies ±1 for organic feel. --skip-chance adds random gaps where no videos post. --keep-uploaded preserves already-uploaded entries when regenerating a schedule
- uploaded all 40 videos to youtube — scheduled Feb 10 through Mar 5, ~2/day with organic gaps. fixed timezone bug in youtube-upload.py where local times were being sent as UTC, causing videos to sit as private instead of scheduled
- rendered 40 videos for first-blend-test using 1-3 min duration range
- changed default blend duration from 2-5 minutes to 1-3 minutes — shorter outputs feel tighter, still enough room for the layers to breathe

2026-02-09
- built phase 9 tooling — three new scripts for the render-to-youtube pipeline:
  - tools/batch-render.py: render N videos in sequence with unique seeds, configurable preset/fps/mode, auto-creates project folders for batches, reports timing stats
  - scripts/upload/schedule.py: scans a directory of MP4s, reads embedded metadata, assigns publishAt dates spaced across a time window, outputs schedule.json. videos missing metadata get titles/descriptions generated on the spot
  - scripts/upload/youtube-upload.py: reads schedule.json, authenticates via Google OAuth, uploads each video as private with scheduled publish time, saves progress after each upload so you can resume across days (quota is ~6 uploads/day)
- added client_secret.json and token.json to .gitignore for youtube API credentials
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
