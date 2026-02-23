"""
youtube-upload.py - upload videos to youtube

two modes:
    --schedule  bulk upload from a schedule.json (the original mode)
    --file      upload a single file directly

by default uploads as private with a publishAt date (if one exists
in the schedule entry). pass --public to upload as public immediately
— this is what the autopilot uses for direct organic uploading.

requires:
    pip install google-api-python-client google-auth-oauthlib

setup:
    1. go to Google Cloud Console (console.cloud.google.com)
    2. create a project and enable YouTube Data API v3
    3. create OAuth 2.0 credentials (Desktop application)
    4. download as client_secret.json
    5. place client_secret.json in the project root (gitignored)
    6. first run will open a browser for authentication
    7. credentials are cached in token.json for future runs

usage:
    python scripts/upload/youtube-upload.py --schedule schedule.json
    python scripts/upload/youtube-upload.py --schedule schedule.json --public --limit 6
    python scripts/upload/youtube-upload.py --file video.mp4 --public
    python scripts/upload/youtube-upload.py --file video.mp4 --public --dry-run

quota notes:
    youtube API gives you 10,000 units/day with a standard key.
    each upload costs ~1,600 units. that's roughly 6 uploads per day.
    use --limit to stay within quota. the script marks uploaded entries
    in the schedule so you can resume the next day.
"""

import os
import sys
import json
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# youtube API category IDs
# https://developers.google.com/youtube/v3/docs/videoCategories/list
CATEGORY_MAP = {
    "Film & Animation": "1",
    "Music": "10",
    "Entertainment": "24",
    "Education": "27",
    "Science & Technology": "28",
    "People & Blogs": "22",
}


def get_authenticated_service(client_secret_path, token_path):
    """
    authenticate with youtube API using OAuth 2.0.
    first run opens a browser, subsequent runs use cached token.
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

    # full youtube scope — covers upload, delete, and metadata updates.
    # originally used youtube.upload only, but we need delete capability
    # for managing published videos. backward-compatible; just requires
    # a one-time re-auth if the old token only had upload scope.
    SCOPES = ['https://www.googleapis.com/auth/youtube']

    creds = None

    # load cached credentials
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    # refresh or get new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        if not creds or not creds.valid:
            if not os.path.exists(client_secret_path):
                print(f"error: {client_secret_path} not found")
                print("download OAuth credentials from Google Cloud Console")
                print("see the docstring at the top of this file for setup instructions")
                sys.exit(1)

            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            creds = flow.run_local_server(port=0)

        # cache for next time
        with open(token_path, 'w') as f:
            f.write(creds.to_json())

    return build('youtube', 'v3', credentials=creds)


def upload_video(youtube, entry, dry_run=False, public=False):
    """
    upload a single video to youtube.

    when public=False (default): uploads as private with a publishAt time
    so youtube auto-publishes at the scheduled time.

    when public=True: uploads as public immediately. this is how the
    autopilot does direct organic uploading — the timing comes from
    when the cron tick fires, not from youtube's scheduler.

    args:
        youtube: authenticated youtube API service
        entry: schedule entry dict (or dict with file/title/description)
        dry_run: if True, just print what would happen
        public: if True, upload as public with no publishAt

    returns:
        video ID on success, None on failure
    """
    filepath = entry['file']
    title = entry['title']
    description = entry.get('description', '')
    tags = entry.get('tags', [])
    category = entry.get('category', 'Film & Animation')
    publish_at = entry.get('publish_at')

    category_id = CATEGORY_MAP.get(category, "1")

    privacy = 'public' if public else 'private'

    if dry_run:
        print(f"  [dry run] would upload: {os.path.basename(filepath)}")
        print(f"    title: {title}")
        print(f"    privacy: {privacy}")
        if not public and publish_at:
            print(f"    publish: {publish_at}")
        return "DRY_RUN"

    if not os.path.exists(filepath):
        print(f"  error: file not found: {filepath}")
        return None

    try:
        from googleapiclient.http import MediaFileUpload

        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags,
                'categoryId': category_id,
            },
            'status': {
                'privacyStatus': privacy,
                'selfDeclaredMadeForKids': False,
            },
        }

        # add scheduled publish time if not public and a time is provided.
        # youtube requires ISO 8601 with timezone. schedule times are local,
        # so we convert to UTC before sending. without this, youtube thinks
        # your 5pm is 5pm UTC and the video just sits as private.
        if not public and publish_at:
            from datetime import datetime, timezone
            local_dt = datetime.strptime(publish_at, '%Y-%m-%dT%H:%M:%S')
            local_dt = local_dt.astimezone()  # attach local timezone
            utc_dt = local_dt.astimezone(timezone.utc)
            body['status']['publishAt'] = utc_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')

        media = MediaFileUpload(
            filepath,
            mimetype='video/mp4',
            resumable=True,
            chunksize=10 * 1024 * 1024,  # 10MB chunks
        )

        request = youtube.videos().insert(
            part='snippet,status',
            body=body,
            media_body=media,
        )

        # upload with progress
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                print(f"    uploading... {pct}%", end='\r')

        video_id = response['id']
        print(f"  uploaded ({privacy}): https://youtube.com/watch?v={video_id}")
        return video_id

    except Exception as e:
        print(f"  upload failed: {e}")
        return None


def upload_single_file(filepath, public=False, dry_run=False, credentials_dir=None,
                       poll_text=None, title_override=None):
    """
    upload a single MP4 file directly — no schedule.json needed.
    reads title/description from the embedded metadata.
    used by the autopilot for direct organic uploads.

    Args:
        filepath: path to the MP4 file
        public: if True, upload as public immediately
        dry_run: if True, print what would happen
        credentials_dir: directory with client_secret.json / token.json
        poll_text: optional poll question to append to the description.
                   comes from scripts/youtube/polls.py — the autopilot
                   decides which uploads get polls based on poll_chance.
        title_override: if set, use this as the title instead of embedded
                        metadata. used by upload-draft.py --comment-title
                        to name videos after audience comments.

    returns:
        video ID on success, None on failure
    """
    from scripts.utils.ffprobe import get_metadata
    from scripts.text.metadata import generate_metadata
    import random

    meta = get_metadata(filepath)
    title = meta.get('title', '')
    description = meta.get('comment', '')

    # generate metadata if the video doesn't have any baked in
    if not title or title.strip() == '':
        generated = generate_metadata(rng=random.Random())
        title = generated['title']
        description = generated['description']

    # caller can force a specific title — the description stays as-is
    # so the cryptic metadata still appears below the comment-as-title
    if title_override:
        title = title_override

    # append poll question if provided — goes at the end of the
    # description so it doesn't disrupt the cryptic metadata above
    if poll_text:
        description = description + poll_text

    entry = {
        'file': os.path.abspath(filepath),
        'filename': os.path.basename(filepath),
        'title': title,
        'description': description,
        'tags': ["generative", "video art", "found footage", "experimental"],
        'category': 'Film & Animation',
    }

    privacy = 'public' if public else 'private'

    if dry_run:
        print(f"  [dry run] would upload: {entry['filename']}")
        print(f"    title: {title}")
        print(f"    privacy: {privacy}")
        return "DRY_RUN"

    creds_dir = credentials_dir or PROJECT_ROOT
    client_secret = os.path.join(creds_dir, 'client_secret.json')
    token = os.path.join(creds_dir, 'token.json')
    youtube = get_authenticated_service(client_secret, token)

    print(f"  {entry['filename']}")
    print(f"    title: {title}")

    return upload_video(youtube, entry, dry_run=False, public=public)


def main():
    parser = argparse.ArgumentParser(
        description='upload videos to youtube',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/upload/youtube-upload.py --schedule schedule.json
  python scripts/upload/youtube-upload.py --schedule schedule.json --public --limit 6
  python scripts/upload/youtube-upload.py --file video.mp4 --public
  python scripts/upload/youtube-upload.py --file video.mp4 --public --dry-run
        """
    )

    # two modes: schedule (bulk) or file (single)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--schedule',
                      help='path to schedule.json for bulk upload')
    mode.add_argument('--file',
                      help='path to a single MP4 to upload directly')

    parser.add_argument('--public', action='store_true',
                        help='upload as public immediately (default: private with publishAt)')
    parser.add_argument('--dry-run', action='store_true',
                        help='preview what would be uploaded without actually uploading')
    parser.add_argument('--limit', type=int, default=None,
                        help='max videos to upload (default: all pending). use to stay within API quota (~6/day)')
    parser.add_argument('--credentials', default=None,
                        help='directory containing client_secret.json and token.json (default: project root)')
    parser.add_argument('--poll-text', default=None,
                        help='poll question to append to the description (used by autopilot)')
    parser.add_argument('--title', default=None,
                        help='override the video title (ignores embedded metadata)')

    args = parser.parse_args()

    # --- single file mode ---
    if args.file:
        if not os.path.exists(args.file):
            print(f"error: file not found: {args.file}")
            sys.exit(1)

        video_id = upload_single_file(
            args.file,
            public=args.public,
            dry_run=args.dry_run,
            credentials_dir=args.credentials,
            poll_text=args.poll_text,
            title_override=args.title,
        )
        if video_id:
            print(f"\ndone: {video_id}")
        else:
            print("\nupload failed")
            sys.exit(1)
        return

    # --- schedule mode ---
    if not os.path.exists(args.schedule):
        print(f"error: schedule not found: {args.schedule}")
        sys.exit(1)

    with open(args.schedule) as f:
        schedule = json.load(f)

    # filter to pending entries only
    pending = [e for e in schedule if e.get('status') == 'pending']

    if not pending:
        print("nothing to upload — all entries are already done")
        return

    if args.limit:
        pending = pending[:args.limit]

    print(f"\n{len(pending)} video(s) to upload")
    if args.dry_run:
        print("(dry run — nothing will actually be uploaded)\n")
    print()

    # authenticate (skip for dry run)
    youtube = None
    if not args.dry_run:
        creds_dir = args.credentials or PROJECT_ROOT
        client_secret = os.path.join(creds_dir, 'client_secret.json')
        token = os.path.join(creds_dir, 'token.json')
        youtube = get_authenticated_service(client_secret, token)

    # upload each video
    uploaded = 0
    failed = 0

    for entry in pending:
        print(f"\n[{uploaded + failed + 1}/{len(pending)}] {entry['title']}")
        print(f"  file: {entry['filename']}")
        if not args.public and entry.get('publish_at'):
            print(f"  scheduled: {entry['publish_at']}")

        video_id = upload_video(youtube, entry, dry_run=args.dry_run, public=args.public)

        if video_id:
            # mark as uploaded in the schedule
            entry['status'] = 'uploaded'
            entry['video_id'] = video_id
            entry['uploaded_at'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            uploaded += 1
        else:
            entry['status'] = 'failed'
            failed += 1

        # save progress after each upload so we can resume if interrupted
        if not args.dry_run:
            with open(args.schedule, 'w') as f:
                json.dump(schedule, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  done: {uploaded} uploaded, {failed} failed")
    remaining = len([e for e in schedule if e.get('status') == 'pending'])
    if remaining:
        print(f"  {remaining} still pending — run again to continue")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
