"""
schedule.py - assign publish dates to a batch of videos

scans a directory for MP4 files, reads their embedded metadata
(title + description), and generates a schedule JSON mapping
each file to a publishAt datetime.

the schedule is the contract between rendering and uploading.
youtube-upload.py reads this file and uses it as its upload queue.

supports multiple posts per day with random times within a window,
and organic jitter so the schedule doesn't feel robotic.

usage:
    python scripts/upload/schedule.py --dir projects/my-series/output
    python scripts/upload/schedule.py --dir projects/my-series/output --per-day 2 --window 08:00-17:00
    python scripts/upload/schedule.py --dir projects/my-series/output --per-day 2 --window 08:00-17:00 --skip-chance 0.15
"""

import os
import sys
import json
import glob
import random
import argparse
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.utils.ffprobe import get_metadata
from scripts.text.metadata import generate_metadata


def scan_videos(directory):
    """
    find all MP4 files in a directory and read their embedded metadata.
    returns a list of dicts with file path, title, and description.
    videos missing metadata get freshly generated titles/descriptions.
    """
    pattern = os.path.join(directory, '*.mp4')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"no MP4 files found in {directory}")
        return []

    videos = []
    rng = random.Random()

    for f in files:
        meta = get_metadata(f)
        title = meta.get('title', '')
        description = meta.get('comment', '')

        # generate metadata for videos that don't have any
        if not title or title.strip() == '':
            generated = generate_metadata(rng=rng)
            title = generated['title']
            description = generated['description']
            print(f"  generated metadata for {os.path.basename(f)}")

        videos.append({
            'file': os.path.abspath(f),
            'filename': os.path.basename(f),
            'title': title,
            'description': description,
            'artist': meta.get('artist', 'luis queral'),
        })

    return videos


def load_existing_schedule(schedule_path):
    """
    read an existing schedule.json to find already-uploaded entries.
    returns a dict mapping filename -> entry for any that have been
    uploaded, so we can preserve their status in a regenerated schedule.
    """
    if not os.path.exists(schedule_path):
        return {}

    try:
        with open(schedule_path) as f:
            entries = json.load(f)
        return {
            e['filename']: e for e in entries
            if e.get('status') == 'uploaded'
        }
    except Exception:
        return {}


def build_schedule(videos, start_date, per_day=2,
                   window_start="08:00", window_end="17:00",
                   skip_chance=0.0,
                   tags=None, category="Film & Animation"):
    """
    assign publishAt datetimes with organic spacing.

    places N videos per day at random times within a time window.
    skip_chance introduces occasional gaps — some days get no posts,
    which makes the output feel less like a machine and more like
    a person who sometimes forgets to post.

    args:
        videos: list of video dicts from scan_videos()
        start_date: first publish date (datetime.date)
        per_day: how many videos to post per day (default 2)
        window_start: earliest post time, HH:MM (default "08:00")
        window_end: latest post time, HH:MM (default "17:00")
        skip_chance: probability of skipping a day, 0.0-1.0 (default 0)
        tags: list of tags to apply to all videos
        category: youtube category name

    returns:
        list of schedule entries (video info + publish_at)
    """
    default_tags = tags or ["generative", "video art", "found footage", "experimental"]
    rng = random.Random()

    # parse window into minutes-from-midnight for easy random picking
    start_h, start_m = map(int, window_start.split(':'))
    end_h, end_m = map(int, window_end.split(':'))
    window_start_mins = start_h * 60 + start_m
    window_end_mins = end_h * 60 + end_m

    schedule = []
    video_queue = list(videos)
    current_date = start_date

    while video_queue:
        # occasionally skip a day for organic feel
        if skip_chance > 0 and rng.random() < skip_chance:
            current_date += timedelta(days=1)
            continue

        # pick random times within the window for today's batch
        # how many videos today — usually per_day, but sometimes ±1
        # for variety (occasionally post 1 or 3 instead of always 2)
        if per_day >= 2:
            today_count = per_day + rng.choice([-1, 0, 0, 0, 1])
            today_count = max(1, today_count)
        else:
            today_count = per_day

        # don't overshoot remaining videos
        today_count = min(today_count, len(video_queue))

        # pick random times, sorted so they're chronological within the day
        times = sorted([
            rng.randint(window_start_mins, window_end_mins)
            for _ in range(today_count)
        ])

        for t in times:
            video = video_queue.pop(0)
            pub_hour = t // 60
            pub_minute = t % 60

            publish_at = datetime(
                current_date.year, current_date.month, current_date.day,
                pub_hour, pub_minute, rng.randint(0, 59)  # random seconds too
            )

            schedule.append({
                'file': video['file'],
                'filename': video['filename'],
                'title': video['title'],
                'description': video['description'],
                'tags': default_tags,
                'category': category,
                'privacy': 'private',
                'publish_at': publish_at.strftime('%Y-%m-%dT%H:%M:%S'),
                'status': 'pending',
            })

        current_date += timedelta(days=1)

    return schedule


def save_schedule(schedule, output_path):
    """write the schedule to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(schedule, f, indent=2)
    print(f"\nschedule saved: {output_path}")
    print(f"  {len(schedule)} videos")
    if schedule:
        first_date = schedule[0]['publish_at'][:10]
        last_date = schedule[-1]['publish_at'][:10]
        days_span = (datetime.strptime(last_date, '%Y-%m-%d') -
                     datetime.strptime(first_date, '%Y-%m-%d')).days + 1
        print(f"  span: {first_date} → {last_date} ({days_span} days)")
        print(f"  first: {schedule[0]['publish_at']} — {schedule[0]['title']}")
        print(f"  last:  {schedule[-1]['publish_at']} — {schedule[-1]['title']}")


def main():
    parser = argparse.ArgumentParser(
        description='assign publish dates to a batch of rendered videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/upload/schedule.py --dir projects/my-series/output
  python scripts/upload/schedule.py --dir projects/my-series/output --per-day 2 --window 08:00-17:00
  python scripts/upload/schedule.py --dir projects/my-series/output --per-day 3 --window 10:00-20:00 --skip-chance 0.15
  python scripts/upload/schedule.py --dir projects/my-series/output --start 2026-03-01
  python scripts/upload/schedule.py --dir projects/my-series/output --tags "art,generative,glitch"
        """
    )

    parser.add_argument('--dir', required=True,
                        help='directory containing MP4 files to schedule')
    parser.add_argument('--start', default=None,
                        help='first publish date (YYYY-MM-DD, default: tomorrow)')
    parser.add_argument('--per-day', type=int, default=2,
                        help='videos per day (default: 2). actual count varies ±1 for organic feel')
    parser.add_argument('--window', default='08:00-17:00',
                        help='time window for posts, HH:MM-HH:MM (default: 08:00-17:00)')
    parser.add_argument('--skip-chance', type=float, default=0.0,
                        help='probability of skipping a day, 0.0-1.0 (default: 0). adds gaps for organic rhythm')
    parser.add_argument('--tags', default=None,
                        help='comma-separated tags for all videos')
    parser.add_argument('--output', '-o', default=None,
                        help='output path for schedule JSON (default: schedule.json in video dir)')
    parser.add_argument('--preview', action='store_true',
                        help='print schedule without saving')
    parser.add_argument('--keep-uploaded', action='store_true',
                        help='preserve uploaded entries from existing schedule.json (default: fresh start)')

    args = parser.parse_args()

    # default start date is tomorrow
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    else:
        start_date = (datetime.now() + timedelta(days=1)).date()

    tags = [t.strip() for t in args.tags.split(',')] if args.tags else None

    # parse window
    if '-' in args.window:
        window_start, window_end = args.window.split('-')
    else:
        # fallback: treat as center time with ±4h window
        window_start = args.window
        window_end = args.window

    print(f"scanning {args.dir}...")
    videos = scan_videos(args.dir)
    if not videos:
        return

    print(f"found {len(videos)} videos")

    # optionally preserve already-uploaded entries
    output_path = args.output or os.path.join(args.dir, 'schedule.json')
    uploaded = {}
    if args.keep_uploaded:
        uploaded = load_existing_schedule(output_path)
        if uploaded:
            print(f"  preserving {len(uploaded)} already-uploaded entries")

    # separate uploaded from pending videos
    uploaded_entries = []
    pending_videos = []
    for v in videos:
        if v['filename'] in uploaded:
            uploaded_entries.append(uploaded[v['filename']])
        else:
            pending_videos.append(v)

    schedule = build_schedule(
        pending_videos,
        start_date=start_date,
        per_day=args.per_day,
        window_start=window_start,
        window_end=window_end,
        skip_chance=args.skip_chance,
        tags=tags,
    )

    # merge uploaded entries back at the top
    full_schedule = uploaded_entries + schedule

    if args.preview:
        print(f"\n{'='*60}")
        print(f"  schedule preview ({len(full_schedule)} videos)")
        print(f"{'='*60}\n")
        for entry in full_schedule:
            status = f" [{entry['status']}]" if entry.get('status') != 'pending' else ''
            print(f"  {entry['publish_at']}  {entry['title']}{status}")
            print(f"    {entry['filename']}")
            print()
    else:
        save_schedule(full_schedule, output_path)


if __name__ == '__main__':
    main()
