# tools

standalone utilities that don't fit the script pattern. things you run once or occasionally, not as part of the synthesis pipeline.

## what's here

- **camera-search.html** — opens in a browser. generates youtube search URLs for common camera file naming patterns (DCIM, IMG, MOV, CAM, etc.) to help find raw, low-view footage for scraping.
- **batch-render.py** — renders N videos in sequence with one command.
- **autopilot.py** — hourly uploader loop that renders when pool is low and uploads with organic timing.
- **upload-draft.py** — uploads one project video as private (draft-like) for quick youtube checks.
- **download-source-material.py** — source discovery helper. either downloads one specific youtube URL or runs a camera-pattern search (`dcim`, `img_0004`, etc.), picks one result at random, and downloads it with `yt-dlp`.

## what will go here

- **youtube-bulk-schedule.py** — scan a directory of MP4s, read embedded metadata, upload and schedule to YouTube
