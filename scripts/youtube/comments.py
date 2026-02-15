"""
comments.py - fetch and filter youtube comments

the foundation for the comment feedback loop. reads comments from
published videos so they can be used as render seeds, burned into
video overlays, tallied for polls, or absorbed into response titles.

uses the same OAuth credentials as youtube-upload.py — the full
youtube scope already covers comment reading. costs ~1 API unit
per call (vs 1,600 for uploads) so quota impact is negligible.

usage as module:
    from scripts.youtube.comments import fetch_comments, pick_seed_comment

    comments = fetch_comments(video_id="abc123")
    seed_comment = pick_seed_comment(video_ids=["abc123", "def456"])

usage as CLI:
    python scripts/youtube/comments.py --video VIDEO_ID
    python scripts/youtube/comments.py --video VIDEO_ID --min-likes 2
    python scripts/youtube/comments.py --manifest path/to/upload-manifest.json --limit 20
    python scripts/youtube/comments.py --manifest path/to/upload-manifest.json --pick-seed
"""

import os
import sys
import json
import hashlib
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def get_authenticated_service(credentials_dir=None):
    """
    authenticate with youtube API using the project's OAuth credentials.
    reuses the same client_secret.json / token.json as the upload scripts.
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

    # youtube.force-ssl is required for commentThreads — the generic
    # youtube scope doesn't actually cover comment reading. we request
    # both so the token also works for uploads without re-auth.
    SCOPES = [
        'https://www.googleapis.com/auth/youtube',
        'https://www.googleapis.com/auth/youtube.force-ssl',
    ]

    creds_dir = credentials_dir or PROJECT_ROOT
    client_secret = os.path.join(creds_dir, 'client_secret.json')
    token = os.path.join(creds_dir, 'token.json')

    creds = None

    if os.path.exists(token):
        creds = Credentials.from_authorized_user_file(token, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(client_secret):
                print(f"error: {client_secret} not found")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(client_secret, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token, 'w') as f:
            f.write(creds.to_json())

    return build('youtube', 'v3', credentials=creds)


def fetch_comments(video_id, youtube=None, max_results=100, credentials_dir=None):
    """
    fetch top-level comments from a single video.

    returns a list of dicts, each with:
        text, author, like_count, video_id, comment_id, published_at

    Args:
        video_id: youtube video ID
        youtube: authenticated youtube API service (or None to create one)
        max_results: max comments to fetch (API pages at 100)
        credentials_dir: directory with client_secret.json / token.json

    Returns:
        list of comment dicts, newest first
    """
    if youtube is None:
        youtube = get_authenticated_service(credentials_dir)

    comments = []
    page_token = None

    while len(comments) < max_results:
        try:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),
                order='time',
                pageToken=page_token,
                textFormat='plainText',
            )
            response = request.execute()
        except Exception as e:
            # video might have comments disabled, be deleted, etc.
            print(f"  could not fetch comments for {video_id}: {e}")
            break

        for item in response.get('items', []):
            snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'text': snippet['textDisplay'],
                'author': snippet['authorDisplayName'],
                'like_count': snippet.get('likeCount', 0),
                'video_id': video_id,
                'comment_id': item['snippet']['topLevelComment']['id'],
                'published_at': snippet['publishedAt'],
            })

        page_token = response.get('nextPageToken')
        if not page_token:
            break

    return comments


def fetch_comments_from_manifest(manifest_path, youtube=None, lookback=10,
                                  max_per_video=50, credentials_dir=None):
    """
    fetch comments across recent uploaded videos using the upload manifest.

    reads the manifest to find video IDs, then fetches comments from the
    most recent `lookback` videos. returns all comments merged together.

    Args:
        manifest_path: path to upload-manifest.json
        youtube: authenticated youtube API service (or None to create one)
        lookback: how many recent videos to check
        max_per_video: max comments per video
        credentials_dir: directory with client_secret.json / token.json

    Returns:
        list of comment dicts from across recent videos
    """
    if not os.path.exists(manifest_path):
        return []

    with open(manifest_path) as f:
        manifest = json.load(f)

    # get video IDs from the most recent uploaded entries
    uploaded = [
        e for e in manifest
        if e.get('status') == 'uploaded' and e.get('video_id')
        and e['video_id'] not in ('unknown', 'DRY_RUN')
    ]

    # most recent first
    uploaded.sort(key=lambda e: e.get('uploaded_at', ''), reverse=True)
    recent_ids = [e['video_id'] for e in uploaded[:lookback]]

    if not recent_ids:
        return []

    if youtube is None:
        youtube = get_authenticated_service(credentials_dir)

    all_comments = []
    for vid in recent_ids:
        comments = fetch_comments(vid, youtube=youtube, max_results=max_per_video)
        all_comments.extend(comments)

    return all_comments


def filter_comments(comments, min_likes=0, keywords=None, exclude_authors=None):
    """
    filter a list of comments.

    Args:
        comments: list of comment dicts
        min_likes: minimum like count
        keywords: if set, only include comments containing at least one keyword
        exclude_authors: set of author names to skip (e.g. the channel owner)

    Returns:
        filtered list
    """
    filtered = comments

    if min_likes > 0:
        filtered = [c for c in filtered if c['like_count'] >= min_likes]

    if exclude_authors:
        exclude_lower = {a.lower() for a in exclude_authors}
        filtered = [c for c in filtered if c['author'].lower() not in exclude_lower]

    if keywords:
        kw_lower = [k.lower() for k in keywords]
        filtered = [
            c for c in filtered
            if any(kw in c['text'].lower() for kw in kw_lower)
        ]

    return filtered


# =============================================================================
# SEED GENERATION
#
# hash a comment's text into a deterministic integer seed.
# the same comment always produces the same video.
# =============================================================================

def comment_to_seed(text):
    """
    hash comment text into an integer seed for the render pipeline.
    uses sha256 so different comments produce very different seeds.
    the same text always gives the same seed — deterministic.

    Args:
        text: comment text string

    Returns:
        integer seed (fits in 31 bits for python's random module)
    """
    h = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return int(h, 16) % (2**31)


def pick_seed_comment(comments, rng=None, strategy='random'):
    """
    select a comment to use as a render seed.

    strategies:
        random     - pick any comment at random
        most_liked - pick the comment with the most likes
        recent     - pick the most recently posted comment

    Args:
        comments: list of comment dicts
        rng: random.Random instance (for 'random' strategy)
        strategy: selection strategy

    Returns:
        comment dict, or None if no comments available
    """
    if not comments:
        return None

    if strategy == 'most_liked':
        return max(comments, key=lambda c: c['like_count'])

    elif strategy == 'recent':
        return max(comments, key=lambda c: c['published_at'])

    else:
        # random — fall back to first if no rng
        if rng is None:
            import random
            rng = random.Random()
        return rng.choice(comments)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='fetch and inspect youtube comments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/youtube/comments.py --video VIDEO_ID
  python scripts/youtube/comments.py --video VIDEO_ID --min-likes 2
  python scripts/youtube/comments.py --manifest projects/first-blend-test/upload-manifest.json
  python scripts/youtube/comments.py --manifest projects/first-blend-test/upload-manifest.json --pick-seed
        """
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', help='fetch comments from a single video ID')
    source.add_argument('--manifest', help='fetch comments from recent videos in a manifest')

    parser.add_argument('--lookback', type=int, default=10,
                        help='how many recent videos to check (manifest mode, default: 10)')
    parser.add_argument('--limit', type=int, default=50,
                        help='max comments per video (default: 50)')
    parser.add_argument('--min-likes', type=int, default=0,
                        help='only show comments with at least this many likes')
    parser.add_argument('--keywords', nargs='+', default=None,
                        help='only show comments containing these keywords')
    parser.add_argument('--pick-seed', action='store_true',
                        help='pick a comment and show its seed value')
    parser.add_argument('--seed-strategy', default='random',
                        choices=['random', 'most_liked', 'recent'],
                        help='how to pick the seed comment (default: random)')
    parser.add_argument('--json', action='store_true',
                        help='output as JSON')
    parser.add_argument('--credentials', default=None,
                        help='directory containing client_secret.json and token.json')

    args = parser.parse_args()

    # fetch comments
    if args.video:
        comments = fetch_comments(
            args.video,
            max_results=args.limit,
            credentials_dir=args.credentials,
        )
    else:
        comments = fetch_comments_from_manifest(
            args.manifest,
            lookback=args.lookback,
            max_per_video=args.limit,
            credentials_dir=args.credentials,
        )

    # filter
    comments = filter_comments(
        comments,
        min_likes=args.min_likes,
        keywords=args.keywords,
    )

    if args.pick_seed:
        comment = pick_seed_comment(comments, strategy=args.seed_strategy)
        if comment:
            seed = comment_to_seed(comment['text'])
            if args.json:
                print(json.dumps({**comment, 'seed': seed}, indent=2))
            else:
                print(f"\n  seed comment:")
                print(f"    text: {comment['text'][:80]}")
                print(f"    author: {comment['author']}")
                print(f"    likes: {comment['like_count']}")
                print(f"    seed: {seed}")
                print(f"    video: https://youtube.com/watch?v={comment['video_id']}")
                print()
        else:
            print("no comments found")
        return

    # display
    if args.json:
        print(json.dumps(comments, indent=2))
    else:
        print(f"\n  {len(comments)} comment(s)\n")
        for c in comments:
            text_preview = c['text'][:100].replace('\n', ' ')
            likes = f" [{c['like_count']} likes]" if c['like_count'] > 0 else ""
            print(f"  {c['author']}{likes}: {text_preview}")
        print()


if __name__ == '__main__':
    main()
