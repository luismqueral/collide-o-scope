"""
cache.py - Video metadata caching

Builds and maintains a JSON cache of video file metadata (duration, resolution,
audio presence). Avoids re-probing every video on every run — the library scan
happens once, and subsequent runs read from cache.

The cache auto-updates: new files get scanned, deleted files get pruned.
"""

import os
import json
from scripts.utils.ffprobe import (
    get_video_duration,
    get_video_resolution,
    video_has_audio,
)


# video file extensions we recognize
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.3g2')


def load_cache(cache_file):
    """
    Load cached video metadata from a JSON file.

    Args:
        cache_file: Path to the cache JSON file

    Returns:
        Dictionary of {video_path: {width, height, duration, has_audio}}
        Empty dict if cache doesn't exist or is corrupted
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_cache(cache_file, cache_data):
    """
    Save video metadata cache to a JSON file.

    Args:
        cache_file: Path to write the cache
        cache_data: Dictionary of video metadata
    """
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)


def build_cache(folder, cache_file):
    """
    Build or update the video metadata cache for a folder.

    Scans only new videos that aren't already cached. Removes entries
    for videos that have been deleted from the folder.

    Args:
        folder: Directory containing video files
        cache_file: Path to the cache JSON file

    Returns:
        Full cache dictionary with all video metadata
    """
    cache = load_cache(cache_file)

    # find all video files in the folder
    all_videos = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(VIDEO_EXTENSIONS) and not f.startswith('.')
    ]

    # scan new videos that aren't already cached
    new_videos = [v for v in all_videos if v not in cache]

    if new_videos:
        print(f"Scanning {len(new_videos)} new videos (cached: {len(cache)})...")
        for i, video_path in enumerate(new_videos):
            if (i + 1) % 50 == 0:
                print(f"  Scanned {i + 1}/{len(new_videos)}...")

            width, height = get_video_resolution(video_path)
            duration = get_video_duration(video_path)
            has_audio = video_has_audio(video_path)

            cache[video_path] = {
                'width': width,
                'height': height,
                'duration': duration,
                'has_audio': has_audio
            }

        save_cache(cache_file, cache)
        print(f"Cache updated: {len(cache)} total videos")
    else:
        print(f"Using cached metadata for {len(cache)} videos")

    # remove entries for deleted videos
    existing_videos = set(all_videos)
    stale_entries = [v for v in cache if v not in existing_videos]
    if stale_entries:
        for v in stale_entries:
            del cache[v]
        save_cache(cache_file, cache)
        print(f"Removed {len(stale_entries)} stale cache entries")

    return cache
