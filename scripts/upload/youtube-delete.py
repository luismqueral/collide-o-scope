"""
youtube-delete.py - delete videos from youtube by ID

reads video IDs from a schedule.json (filtering by duration or status)
or accepts IDs directly via --ids. uses the same OAuth flow as
youtube-upload.py.

requires the full youtube scope (not just youtube.upload).
if your token.json only has upload scope, delete it and re-authenticate.

usage:
    python scripts/upload/youtube-delete.py --ids VIDEO_ID1 VIDEO_ID2
    python scripts/upload/youtube-delete.py --schedule schedule.json --max-duration 30
    python scripts/upload/youtube-delete.py --schedule schedule.json --max-duration 30 --dry-run
"""

import os
import sys
import json
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def get_authenticated_service(client_secret_path, token_path):
    """
    authenticate with youtube API using OAuth 2.0.
    same as youtube-upload.py but with full youtube scope for delete access.
    """
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError:
        print("missing dependencies. install with:")
        print("  pip install google-api-python-client google-auth-oauthlib")
        sys.exit(1)

    SCOPES = ['https://www.googleapis.com/auth/youtube']

    creds = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(client_secret_path):
                print(f"error: {client_secret_path} not found")
                sys.exit(1)

            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, 'w') as f:
            f.write(creds.to_json())

    return build('youtube', 'v3', credentials=creds)


def get_video_duration(filepath):
    """ask ffprobe how long a video is, returns seconds or -1 on failure."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', filepath],
            capture_output=True, text=True
        )
        info = json.loads(result.stdout)
        return float(info['format'].get('duration', -1))
    except Exception:
        return -1


def delete_video(youtube, video_id, title="", dry_run=False):
    """
    delete a single video from youtube.

    returns True on success, False on failure.
    """
    if dry_run:
        print(f"  [dry run] would delete: {video_id}  {title}")
        return True

    try:
        youtube.videos().delete(id=video_id).execute()
        print(f"  deleted: {video_id}  {title}")
        return True
    except Exception as e:
        print(f"  failed to delete {video_id}: {e}")
        return False


def find_short_videos(schedule_path, max_duration):
    """
    find uploaded videos in a schedule whose files are shorter than max_duration.
    returns list of (entry, duration) tuples.
    """
    with open(schedule_path) as f:
        schedule = json.load(f)

    matches = []
    for entry in schedule:
        if entry.get('status') != 'uploaded':
            continue
        if not entry.get('video_id'):
            continue

        filepath = entry.get('file', '')
        if not os.path.exists(filepath):
            # file might have been moved, try relative to schedule dir
            schedule_dir = os.path.dirname(schedule_path)
            filepath = os.path.join(schedule_dir, entry.get('filename', ''))

        if os.path.exists(filepath):
            dur = get_video_duration(filepath)
            if 0 < dur <= max_duration:
                matches.append((entry, dur))

    return matches


def main():
    parser = argparse.ArgumentParser(
        description='delete videos from youtube',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/upload/youtube-delete.py --ids abc123 def456
  python scripts/upload/youtube-delete.py --schedule schedule.json --max-duration 30
  python scripts/upload/youtube-delete.py --schedule schedule.json --max-duration 30 --dry-run
        """
    )

    parser.add_argument('--ids', nargs='+', default=None,
                        help='youtube video IDs to delete')
    parser.add_argument('--schedule', default=None,
                        help='path to schedule.json — finds uploaded videos to delete')
    parser.add_argument('--max-duration', type=float, default=None,
                        help='delete uploaded videos shorter than this (seconds). requires --schedule')
    parser.add_argument('--dry-run', action='store_true',
                        help='preview what would be deleted without actually deleting')
    parser.add_argument('--credentials', default=None,
                        help='directory containing client_secret.json and token.json')
    parser.add_argument('--clean-schedule', action='store_true',
                        help='also remove deleted entries from schedule.json')

    args = parser.parse_args()

    if not args.ids and not args.schedule:
        parser.error("need either --ids or --schedule")

    if args.max_duration and not args.schedule:
        parser.error("--max-duration requires --schedule")

    # collect videos to delete
    to_delete = []

    if args.ids:
        to_delete = [{'video_id': vid, 'title': vid} for vid in args.ids]

    elif args.schedule and args.max_duration:
        matches = find_short_videos(args.schedule, args.max_duration)
        if not matches:
            print("no matching videos found")
            return
        to_delete = [entry for entry, dur in matches]
        print(f"found {len(matches)} video(s) under {args.max_duration}s:\n")
        for entry, dur in matches:
            print(f"  {dur:.0f}s  {entry['title']}")
            print(f"       https://youtube.com/watch?v={entry['video_id']}")
        print()

    if not to_delete:
        print("nothing to delete")
        return

    print(f"{len(to_delete)} video(s) to delete")
    if args.dry_run:
        print("(dry run — nothing will actually be deleted)\n")

    # authenticate
    youtube = None
    if not args.dry_run:
        creds_dir = args.credentials or PROJECT_ROOT
        client_secret = os.path.join(creds_dir, 'client_secret.json')
        token = os.path.join(creds_dir, 'token.json')
        youtube = get_authenticated_service(client_secret, token)

    # delete each video
    deleted_ids = set()
    deleted = 0
    failed = 0

    for entry in to_delete:
        vid = entry['video_id']
        title = entry.get('title', vid)

        if delete_video(youtube, vid, title=title, dry_run=args.dry_run):
            deleted += 1
            deleted_ids.add(vid)
        else:
            failed += 1

    print(f"\ndone: {deleted} deleted, {failed} failed")

    # optionally clean up schedule.json
    if args.clean_schedule and args.schedule and deleted_ids and not args.dry_run:
        with open(args.schedule) as f:
            schedule = json.load(f)

        before = len(schedule)
        schedule = [e for e in schedule if e.get('video_id') not in deleted_ids]
        after = len(schedule)

        with open(args.schedule, 'w') as f:
            json.dump(schedule, f, indent=2)

        print(f"cleaned schedule: {before} → {after} entries ({before - after} removed)")


if __name__ == '__main__':
    main()
