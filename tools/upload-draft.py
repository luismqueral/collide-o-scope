"""
upload-draft.py - force one video upload as private

"draft" in youtube api terms is private visibility.
this script picks a video from a project output folder and uploads it
as private using the existing youtube uploader.

defaults are safe:
- prefers videos that are not marked uploaded in upload-manifest.json
- does not touch upload-manifest.json
- supports --dry-run

--comment-title pulls a random comment from published videos and uses
it as the video title. the cryptic metadata description stays intact
underneath. if no comments exist yet, falls back to normal metadata.

usage:
    python tools/upload-draft.py --project first-blend-test --dry-run
    python tools/upload-draft.py --project first-blend-test
    python tools/upload-draft.py --project first-blend-test --random
    python tools/upload-draft.py --project first-blend-test --comment-title
    python tools/upload-draft.py --project first-blend-test --random --comment-title --dry-run
    python tools/upload-draft.py --project first-blend-test --file projects/first-blend-test/output/output_20260210_172252.mp4
"""

import os
import sys
import json
import random
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_uploaded_set(manifest_path):
    """read uploaded filenames from upload-manifest.json if it exists."""
    if not os.path.exists(manifest_path):
        return set()

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()

    return {
        entry.get('filename')
        for entry in manifest
        if entry.get('status') == 'uploaded' and entry.get('filename')
    }


def pick_video(project_dir, filepath=None, random_pick=False, include_uploaded=False):
    """
    pick one mp4 from project output.
    by default prefers files not marked uploaded in upload-manifest.json.
    """
    output_dir = os.path.join(project_dir, 'output')
    manifest_path = os.path.join(project_dir, 'upload-manifest.json')

    if filepath:
        abs_path = filepath if os.path.isabs(filepath) else os.path.join(PROJECT_ROOT, filepath)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"file not found: {abs_path}")
        return os.path.abspath(abs_path)

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"project output dir not found: {output_dir}")

    all_mp4s = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.lower().endswith('.mp4')
    ]
    all_mp4s.sort()

    if not all_mp4s:
        raise RuntimeError(f"no mp4 files found in {output_dir}")

    candidates = list(all_mp4s)
    if not include_uploaded:
        uploaded = load_uploaded_set(manifest_path)
        unuploaded = [path for path in all_mp4s if os.path.basename(path) not in uploaded]
        if unuploaded:
            candidates = unuploaded

    if random_pick:
        return os.path.abspath(random.choice(candidates))

    # newest by modified time feels right for quick checks
    newest = max(candidates, key=os.path.getmtime)
    return os.path.abspath(newest)


def main():
    parser = argparse.ArgumentParser(
        description='force one project video upload as private (draft-like)'
    )
    parser.add_argument('--project', required=True,
                        help='project name, e.g. first-blend-test')
    parser.add_argument('--file', default=None,
                        help='specific mp4 file to upload')
    parser.add_argument('--random', action='store_true',
                        help='pick a random candidate instead of newest')
    parser.add_argument('--include-uploaded', action='store_true',
                        help='allow files already marked uploaded in upload-manifest.json')
    parser.add_argument('--comment-title', action='store_true',
                        help='use a random youtube comment as the video title')
    parser.add_argument('--credentials', default=None,
                        help='directory with client_secret.json and token.json')
    parser.add_argument('--dry-run', action='store_true',
                        help='show what would upload without sending')
    args = parser.parse_args()

    project_dir = os.path.join(PROJECT_ROOT, 'projects', args.project)
    if not os.path.exists(project_dir):
        print(f"error: project not found: {project_dir}")
        sys.exit(1)

    try:
        selected = pick_video(
            project_dir,
            filepath=args.file,
            random_pick=args.random,
            include_uploaded=args.include_uploaded,
        )
    except Exception as e:
        print(f"error: {e}")
        sys.exit(1)

    print(f"selected: {selected}")
    print("visibility: private (draft-like)")

    # optionally grab a random comment to use as the title.
    # pulls from published videos via the upload manifest — if
    # no comments exist yet the title falls back to embedded metadata.
    comment_title = None
    if args.comment_title:
        manifest_path = os.path.join(project_dir, 'upload-manifest.json')
        try:
            sys.path.insert(0, PROJECT_ROOT)
            from scripts.youtube.comments import (
                fetch_comments_from_manifest, pick_seed_comment,
            )
            creds_dir = args.credentials or PROJECT_ROOT
            comments = fetch_comments_from_manifest(
                manifest_path,
                lookback=10,
                max_per_video=50,
                credentials_dir=creds_dir,
            )
            if comments:
                chosen = pick_seed_comment(comments, rng=random.Random(), strategy='random')
                if chosen:
                    # trim to youtube's 100-char title limit
                    comment_title = chosen['text'].strip().replace('\n', ' ')[:100]
                    print(f"comment title: {comment_title}")
                    print(f"  from: {chosen['author']} on https://youtube.com/watch?v={chosen['video_id']}")
            if not comment_title:
                print("no comments found — falling back to embedded metadata title")
        except Exception as e:
            print(f"could not fetch comments: {e}")
            print("falling back to embedded metadata title")

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, 'scripts', 'upload', 'youtube-upload.py'),
        '--file', selected,
    ]
    if comment_title:
        cmd.extend(['--title', comment_title])
    if args.credentials:
        cmd.extend(['--credentials', args.credentials])
    if args.dry_run:
        cmd.append('--dry-run')

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()
