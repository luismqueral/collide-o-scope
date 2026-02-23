"""
download-source-material.py - grab one youtube source clip fast

two modes:
    --url      download a specific youtube url
    --random   run a camera-pattern search, pick one result, download it

this is a test harness for source discovery work - not the full agent
pipeline. it is intentionally tiny so we can validate the download path
before we build orchestration on top.

requires:
    yt-dlp installed and on PATH

usage:
    python3 tools/download-source-material.py --url "https://www.youtube.com/watch?v=abc123"
    python3 tools/download-source-material.py --random
    python3 tools/download-source-material.py --random --query "dcim 0004"
    python3 tools/download-source-material.py --random --dry-run
"""

import os
import sys
import json
import random
import argparse
import subprocess
from datetime import datetime
import re


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "library", "video")
DEFAULT_SEARCH_QUERIES = [
    "dcim 0004",
    "img_0004",
    "mov_0004",
    "dsc_0004",
    "gx01",
]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".wmv", ".flv", ".mpeg", ".mpg", ".3gp", ".3g2", ".mts", ".m2ts", ".ts"}


def run_cmd(cmd):
    """run a subprocess and return stdout, or raise with stderr context."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(stderr or "command failed")
    return result.stdout


def ensure_ytdlp():
    """fail early with a clear message if yt-dlp is missing."""
    try:
        run_cmd(["yt-dlp", "--version"])
    except Exception:
        print("error: yt-dlp is not installed or not on PATH")
        print("install with: python3 -m pip install -U yt-dlp")
        sys.exit(1)


def ensure_ffprobe():
    """ffprobe is used to reject broken output files early."""
    try:
        run_cmd(["ffprobe", "-version"])
    except Exception:
        print("error: ffprobe is not installed or not on PATH")
        print("install FFmpeg so ffprobe is available")
        sys.exit(1)


def extract_video_id(value):
    """pull youtube-like ids from filenames/strings."""
    match = re.search(r"([A-Za-z0-9_-]{11})", value or "")
    return match.group(1) if match else None


def collect_existing_video_ids(output_dir):
    """
    gather ids already present in library/video and _archive.
    this keeps reruns from downloading the same youtube video again.
    """
    known_ids = set()
    paths = [output_dir, os.path.join(output_dir, "_archive")]
    for base in paths:
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            path = os.path.join(base, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1].lower() not in VIDEO_EXTENSIONS:
                continue
            vid = extract_video_id(name)
            if vid:
                known_ids.add(vid)
    return known_ids


def resolve_video_id(url):
    """ask yt-dlp for canonical id so duplicate checks are reliable."""
    value = run_cmd(["yt-dlp", "--no-warnings", "--print", "id", "--no-playlist", url]).strip()
    if not value:
        raise RuntimeError("could not resolve video id")
    video_id = value.splitlines()[-1].strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise RuntimeError(f"unexpected video id format: {video_id}")
    return video_id


def fetch_search_results(query, limit):
    """
    ask yt-dlp for search metadata only.
    using yt-dlp search keeps this script credential-free and simple.
    """
    raw = run_cmd(["yt-dlp", "--dump-single-json", f"ytsearch{limit}:{query}"])
    payload = json.loads(raw)
    entries = payload.get("entries", []) or []
    results = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        video_id = entry.get("id")
        title = entry.get("title")
        if not video_id or not title:
            continue
        results.append(
            {
                "id": video_id,
                "title": title,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "duration": entry.get("duration"),
                "uploader": entry.get("uploader"),
            }
        )
    return results


def select_random_result(query, limit, existing_ids=None, seed=None):
    """search and choose one random video result."""
    results = fetch_search_results(query, limit)
    if not results:
        raise RuntimeError(f"no results found for query: {query}")
    if existing_ids:
        results = [item for item in results if item["id"] not in existing_ids]
        if not results:
            raise RuntimeError(f"all results already exist for query: {query}")
    rng = random.Random(seed)
    return rng.choice(results), len(results)


def sanitize_name(value):
    """make filenames safe across shells/filesystems."""
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("-")
    out = "".join(cleaned).strip("-")
    return out[:80] if out else "clip"


def verify_download(filepath):
    """
    quick integrity pass - catches empty/corrupt files before they enter pool.
    this is intentionally light and fast: size + ffprobe parse + video stream.
    """
    if not os.path.exists(filepath):
        return False, "missing file after download"

    size = os.path.getsize(filepath)
    if size < 256 * 1024:
        return False, f"file too small ({size} bytes)"

    try:
        raw = run_cmd(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-show_streams",
                "-of",
                "json",
                filepath,
            ]
        )
        payload = json.loads(raw)
    except Exception as e:
        return False, f"ffprobe failed: {e}"

    streams = payload.get("streams", []) or []
    if not any(s.get("codec_type") == "video" for s in streams):
        return False, "no video stream found"

    duration = payload.get("format", {}).get("duration")
    try:
        duration_value = float(duration)
    except (TypeError, ValueError):
        return False, "missing duration metadata"
    if duration_value <= 0:
        return False, "non-positive duration"

    return True, "ok"


def download_url(url, output_dir, video_id, prefix=None):
    """download one url as mp4 when possible, with stable naming and checks."""
    os.makedirs(output_dir, exist_ok=True)
    date_tag = datetime.now().strftime("%Y%m%d")
    if prefix:
        stem = sanitize_name(prefix)
        template = os.path.join(output_dir, f"{date_tag}_{stem}_{video_id}.%(ext)s")
    else:
        template = os.path.join(output_dir, f"{date_tag}_{video_id}.%(ext)s")

    before_names = set(os.listdir(output_dir))

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*+ba/b",
        "-o",
        template,
        url,
    ]
    subprocess.run(cmd, check=True)

    # pick the newest file with this id; works even if yt-dlp reused filename.
    candidates = []
    for name in os.listdir(output_dir):
        if name in before_names:
            continue
        full = os.path.join(output_dir, name)
        if not os.path.isfile(full):
            continue
        if video_id not in name:
            continue
        candidates.append(full)
    if not candidates:
        for name in os.listdir(output_dir):
            full = os.path.join(output_dir, name)
            if os.path.isfile(full) and video_id in name:
                candidates.append(full)
    if not candidates:
        raise RuntimeError("download finished but output file was not found")
    candidates.sort(key=os.path.getmtime, reverse=True)
    chosen = candidates[0]

    ok, reason = verify_download(chosen)
    if not ok:
        try:
            os.remove(chosen)
        except OSError:
            pass
        raise RuntimeError(f"download verification failed: {reason}")

    return chosen


def main():
    parser = argparse.ArgumentParser(
        description="test script: download one youtube source video"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--url", help="specific youtube url to download")
    mode.add_argument(
        "--random",
        action="store_true",
        help="search camera-style terms, pick one random video",
    )

    parser.add_argument(
        "--query",
        default=None,
        help="search query for --random mode (default: random camera pattern)",
    )
    parser.add_argument(
        "--search-size",
        type=int,
        default=20,
        help="how many search results to sample from in --random mode",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="where the downloaded file should go",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional random seed for repeatable picks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show selected url and exit without downloading",
    )
    args = parser.parse_args()

    ensure_ytdlp()
    ensure_ffprobe()
    existing_ids = collect_existing_video_ids(args.output_dir)

    if args.url:
        selected_url = args.url.strip()
        selected_id = resolve_video_id(selected_url)
        if selected_id in existing_ids:
            print("mode: url")
            print(f"url: {selected_url}")
            print(f"video-id: {selected_id}")
            print("skip: duplicate video id already exists in library/video or _archive")
            return
        label = "manual-url"
        print(f"mode: url")
        print(f"url: {selected_url}")
        print(f"video-id: {selected_id}")
    else:
        query_pool = [args.query] if args.query else list(DEFAULT_SEARCH_QUERIES)
        random.Random(args.seed).shuffle(query_pool)
        selected = None
        pool_size = 0
        query = None
        for candidate_query in query_pool:
            try:
                candidate, size = select_random_result(
                    candidate_query,
                    args.search_size,
                    existing_ids=existing_ids,
                    seed=args.seed,
                )
                selected = candidate
                pool_size = size
                query = candidate_query
                break
            except RuntimeError:
                continue
        if selected is None:
            print("no non-duplicate results found across configured queries")
            return
        selected_url = selected["url"]
        selected_id = selected["id"]
        label = query
        print("mode: random-search")
        print(f"query: {query}")
        print(f"pool: {pool_size}")
        print(f"picked: {selected['title']}")
        print(f"url: {selected_url}")
        print(f"video-id: {selected_id}")
        if selected.get("uploader"):
            print(f"uploader: {selected['uploader']}")

    print(f"output-dir: {os.path.abspath(args.output_dir)}")
    if args.dry_run:
        print("dry-run: true (no download)")
        return

    try:
        saved_path = download_url(selected_url, args.output_dir, selected_id, prefix=label)
    except (subprocess.CalledProcessError, RuntimeError) as e:
        print(f"download failed: {e}")
        sys.exit(1)

    print(f"saved: {saved_path}")
    print("done")


if __name__ == "__main__":
    main()
