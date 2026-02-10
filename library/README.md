# library

raw material. the footage and audio that everything gets built from.

## video/

scraped youtube footage — low-view camera uploads, DCIM dumps, security cameras, livestock auctions, glass appraisals, whatever shows up. these are the source layers that get composited together.

not checked into git (too large). put your `.mp4` files here and run `python scripts/source/scan-library.py` to cache their metadata.

## audio/

standalone audio sources for mixing into renders. optional — the blend script can also pull audio directly from the video sources.

## getting material

the original workflow uses yt-dlp to scrape videos. the `tools/camera-search.html` page helps find low-view uploads by generating youtube search URLs for common camera file naming patterns (DCIM, IMG, MOV, etc).
