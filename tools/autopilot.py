"""
autopilot.py - steady organic uploader

posts 6 videos a day at random times. renders more when the pool
runs low. that's it. no phases, no bursts, no quiet periods.

designed to run via cron (every hour) on a VPS. each tick it:
    1. checks if the video pool is running low → renders if needed
    2. rolls dice to decide whether to upload this tick
    3. if yes, uploads 1-2 videos as public

the organic feel comes from the per-tick probability math:
remaining uploads are spread across remaining hours in the window,
with ±30% jitter. morning ticks rarely fire. evening ticks almost
always do if there's quota left. looks like a person.

usage:
    python tools/autopilot.py --project first-blend-test
    python tools/autopilot.py --project first-blend-test --dry-run
    python tools/autopilot.py --project first-blend-test --status

cron (every hour):
    0 * * * * cd /path/to/collide-o-scope && python3 tools/autopilot.py --project first-blend-test >> /var/log/autopilot.log 2>&1
"""

import os
import sys
import json
import subprocess
import argparse
import time
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# =============================================================================
# RHYTHM DEFAULTS
#
# the posting personality. override per-project via rhythm.json.
# =============================================================================

RHYTHM_DEFAULTS = {

    # what hours uploads can happen
    "window_hours": ["06:00", "23:00"],

    # max uploads per day (youtube API quota: 10,000 units, ~1,600 per upload)
    "uploads_per_day": 6,

    # --- rendering ---

    # how many videos to render per batch
    "render_batch_size": [20, 40],

    # render a new batch when fewer than this many videos are ready
    "render_when_below": 10,

    # preset to use for rendering
    "render_preset": "classic-white",

    # --- comment feedback ---
    #
    # the audience shapes the work without knowing it.
    # comments become seeds, overlays, votes, and response prompts.
    # everything here is off by default — opt in via rhythm.json.

    # master switch — set True to enable the comment feedback loop
    "comment_feedback": False,

    # use comment text as render seeds (hash comment → seed integer)
    "seed_from_comments": True,

    # burn comment text into videos as faded overlays
    "burn_comments": True,

    # chance that any given video gets comments burned in (0.0-1.0)
    "burn_comment_chance": 0.4,

    # chance that a video in the upload phase gets a poll question
    # appended to its description
    "poll_chance": 0.25,

    # chance that a render is flagged as a "response" to a comment,
    # absorbing the comment's words into the title/description
    "response_chance": 0.12,

    # minimum likes on a comment before it's eligible for seeding/responses
    "min_comment_likes": 0,

    # how many recent uploaded videos to check for comments
    "comment_lookback_videos": 10,

    # how many days back to scan uploaded videos for comments.
    # 90 days = roughly 3 months.
    "comment_lookback_days": 90,

    # use recent youtube comments to steer titles for newly rendered videos.
    # this works even if the full comment feedback loop is disabled.
    "title_from_comments": False,

    # how to pick comments for title steering: random, most_liked, recent
    "title_comment_strategy": "random",

    # --- cleanup ---

    # delete videos from disk after successful upload.
    # no reason to hoard gigabytes of MP4s that are already on youtube.
    "delete_after_upload": True,
}


def log(msg):
    """timestamped log line."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}")


def acquire_lock(project_dir):
    """prevent multiple autopilot instances from running at once.

    without this, hourly cron stacks up processes when a render batch
    takes longer than an hour — which it always does on a small VPS.
    returns the lock file handle (keep it open) or None if locked.
    """
    lock_path = os.path.join(project_dir, '.autopilot.lock')
    import fcntl
    try:
        lock_file = open(lock_path, 'w')
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        return lock_file
    except (IOError, OSError):
        return None


def load_rhythm(project_dir):
    """
    load the rhythm config for a project.
    merges project-level rhythm.json over RHYTHM_DEFAULTS.
    """
    config = dict(RHYTHM_DEFAULTS)

    rhythm_path = os.path.join(project_dir, 'rhythm.json')
    if os.path.exists(rhythm_path):
        with open(rhythm_path) as f:
            overrides = json.load(f)
        config.update(overrides)
        log(f"loaded rhythm from {rhythm_path}")

    return config


def load_state(state_path):
    """
    load the autopilot state file, or create a fresh one.
    just tracks the seed for reproducible randomness.
    """
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)

    return {
        "seed": int(time.time()) % (2**31),
    }


def save_state(state, state_path):
    """persist state to disk."""
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)


def count_ready_videos(output_dir, uploaded_set):
    """
    count MP4s in the output dir that haven't been uploaded yet.
    returns (count, set of filenames).
    """
    import glob as globmod

    all_mp4s = set(
        os.path.basename(f) for f in globmod.glob(os.path.join(output_dir, '*.mp4'))
    )

    ready = all_mp4s - uploaded_set
    return len(ready), ready


def load_manifest(manifest_path):
    """
    load the upload manifest — tracks what's been uploaded.
    just filename, video_id, timestamp, url.
    """
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return []


def save_manifest(manifest, manifest_path):
    """persist the upload manifest."""
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def get_uploaded_set(manifest):
    """get the set of filenames that have been uploaded."""
    return set(e['filename'] for e in manifest if e.get('status') == 'uploaded')


def uploaded_today(manifest):
    """count how many videos were uploaded today."""
    today = datetime.now().strftime('%Y-%m-%d')
    return sum(
        1 for e in manifest
        if e.get('uploaded_at', '').startswith(today)
    )


def create_rng(seed):
    """simple seeded RNG."""
    import random
    return random.Random(seed)


def pick_from_range(rng, value):
    """resolve a [min, max] range or return a fixed value."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        if isinstance(value[0], int) and isinstance(value[1], int):
            return rng.randint(value[0], value[1])
        return rng.uniform(value[0], value[1])
    return value


def _parse_manifest_time(timestamp):
    """
    parse timestamps from upload-manifest entries.
    returns naive local datetime, or None if parsing fails.
    """
    if not timestamp:
        return None

    value = str(timestamp).strip()
    if value.endswith('Z'):
        value = value[:-1] + '+00:00'

    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
            try:
                dt = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
        else:
            return None

    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)

    return dt


def in_window(rhythm):
    """check if the current time is within the upload window."""
    window = rhythm.get('window_hours', ['06:00', '23:00'])
    start_h, start_m = map(int, window[0].split(':'))
    end_h, end_m = map(int, window[1].split(':'))

    now = datetime.now()
    now_mins = now.hour * 60 + now.minute
    start_mins = start_h * 60 + start_m
    end_mins = end_h * 60 + end_m

    return start_mins <= now_mins <= end_mins


# =============================================================================
# COMMENT FEEDBACK
#
# kept from the previous version — fetches comments, reads polls,
# post-processes renders. only runs when comment_feedback is True.
# =============================================================================

def _fetch_comment_context(manifest_path, rhythm, dry_run=False):
    """
    gather comment and poll data from recent uploads.
    only runs when comment_feedback is enabled in rhythm.json.

    returns dict with 'comments', 'poll_overrides', 'poll_results'.
    """
    comments = []
    poll_overrides = {}
    poll_results = []

    comment_feedback_enabled = rhythm.get('comment_feedback', False)
    title_feedback_enabled = rhythm.get('title_from_comments', False)

    if not comment_feedback_enabled and not title_feedback_enabled:
        return {
            'comments': comments,
            'poll_overrides': poll_overrides,
            'poll_results': poll_results,
        }

    try:
        manifest = load_manifest(manifest_path)
        uploaded = [e for e in manifest if e.get('status') == 'uploaded'
                    and e.get('video_id') not in (None, 'unknown')]

        uploaded.sort(key=lambda e: e.get('uploaded_at', ''), reverse=True)

        lookback_days = rhythm.get('comment_lookback_days', 90)
        recent = []
        if lookback_days and int(lookback_days) > 0:
            cutoff = datetime.now() - timedelta(days=int(lookback_days))
            recent = [
                e for e in uploaded
                if (_parse_manifest_time(e.get('uploaded_at')) or datetime.min) >= cutoff
            ]
            if recent:
                log(f"comment scan window: last {lookback_days} day(s), {len(recent)} video(s)")

        if not recent:
            lookback = rhythm.get('comment_lookback_videos', 10)
            recent = uploaded[:lookback] if len(uploaded) > lookback else uploaded
            if lookback_days:
                log(f"no uploads found inside {lookback_days} day window — "
                    f"falling back to latest {len(recent)} video(s)")

        if recent:
            try:
                from scripts.youtube.comments import fetch_comments, filter_comments
                from scripts.upload.youtube_upload import get_authenticated_service

                creds_dir = PROJECT_ROOT
                client_secret = os.path.join(creds_dir, 'client_secret.json')
                token = os.path.join(creds_dir, 'token.json')

                if not dry_run:
                    youtube = get_authenticated_service(client_secret, token)

                    for entry in recent:
                        vid_comments = fetch_comments(
                            entry['video_id'],
                            youtube=youtube,
                            max_results=50,
                        )
                        vid_comments = filter_comments(
                            vid_comments,
                            min_likes=rhythm.get('min_comment_likes', 0),
                        )
                        comments.extend(vid_comments)

                    if comments:
                        log(f"fetched {len(comments)} comments from {len(recent)} recent videos")
                else:
                    log(f"[dry run] would fetch comments from {len(recent)} recent videos")

            except Exception as e:
                log(f"comment fetching failed (non-fatal): {e}")

        # read poll results
        polls_state_path = os.path.join(
            os.path.dirname(manifest_path), 'polls-state.json'
        )

        if os.path.exists(polls_state_path):
            try:
                from scripts.youtube.polls import read_poll_results, aggregate_poll_overrides

                with open(polls_state_path) as f:
                    active_polls = json.load(f)

                unread = [p for p in active_polls if not p.get('read', False)]

                if unread and not dry_run:
                    youtube = get_authenticated_service(
                        os.path.join(PROJECT_ROOT, 'client_secret.json'),
                        os.path.join(PROJECT_ROOT, 'token.json'),
                    )

                    for poll_entry in unread:
                        try:
                            result = read_poll_results(
                                poll_entry['video_id'],
                                poll_entry['poll'],
                                youtube=youtube,
                            )
                            poll_results.append(result)

                            if result['winner']:
                                log(f"  poll on {poll_entry['video_id']}: "
                                    f"winner={result['winner']} ({result['votes']})")
                            else:
                                log(f"  poll on {poll_entry['video_id']}: no votes")

                            poll_entry['read'] = True

                        except Exception as e:
                            log(f"  poll read failed for {poll_entry['video_id']}: {e}")

                    with open(polls_state_path, 'w') as f:
                        json.dump(active_polls, f, indent=2)

                    poll_overrides = aggregate_poll_overrides(poll_results)
                    if poll_overrides:
                        log(f"poll overrides: {poll_overrides}")

            except Exception as e:
                log(f"poll reading failed (non-fatal): {e}")

    except Exception as e:
        log(f"comment context failed (non-fatal): {e}")

    return {
        'comments': comments,
        'poll_overrides': poll_overrides,
        'poll_results': poll_results,
    }


def _pick_title_comment(comments, rng, strategy='random'):
    """
    pick one comment string for title steering.
    """
    if not comments:
        return None

    valid = [c for c in comments if c.get('text', '').strip()]
    if not valid:
        return None

    if strategy == 'most_liked':
        return max(valid, key=lambda c: c.get('like_count', 0)).get('text')
    if strategy == 'recent':
        return max(valid, key=lambda c: c.get('published_at', '')).get('text')

    return rng.choice(valid).get('text')


def _post_process_renders(output_dir, uploaded_set, comments, rhythm, rng, dry_run=False):
    """
    post-process newly rendered videos with comment feedback.
    burns comments as overlays and embeds response metadata.
    only runs when comment_feedback is True.
    """
    if not rhythm.get('comment_feedback', False):
        return

    if not comments:
        return

    _, ready_files = count_ready_videos(output_dir, uploaded_set)
    if not ready_files:
        return

    burn_enabled = rhythm.get('burn_comments', True)
    burn_chance = rhythm.get('burn_comment_chance', 0.4)
    response_chance = rhythm.get('response_chance', 0.12)

    for filename in sorted(ready_files):
        filepath = os.path.join(output_dir, filename)

        # burn comments as overlay
        if burn_enabled and rng.random() < burn_chance:
            try:
                from scripts.post.burn_comments import burn_comment_overlay

                comment_texts = [c['text'] for c in rng.sample(
                    comments, min(3, len(comments))
                )]

                if dry_run:
                    log(f"[dry run] would burn comments into {filename}")
                else:
                    burn_comment_overlay(filepath, comment_texts)
                    log(f"burned comments into {filename}")

            except Exception as e:
                log(f"comment burn failed for {filename} (non-fatal): {e}")

        # response metadata
        if rng.random() < response_chance and comments:
            try:
                from scripts.youtube.comments import pick_seed_comment

                response_comment = pick_seed_comment(comments, rng=rng)
                if response_comment:
                    if dry_run:
                        log(f"[dry run] would embed response metadata in {filename}")
                    else:
                        log(f"response video: {filename} ← \"{response_comment['text'][:40]}\"")

            except Exception as e:
                log(f"response metadata failed for {filename} (non-fatal): {e}")


# =============================================================================
# CORE LOGIC
# =============================================================================

def maybe_render(rhythm, rng, output_dir, manifest_path, project_name, dry_run=False):
    """
    render more videos if the pool is running low.
    returns True if rendering happened (or would happen in dry-run).
    """
    manifest = load_manifest(manifest_path)
    uploaded_set = get_uploaded_set(manifest)
    ready_count, _ = count_ready_videos(output_dir, uploaded_set)

    threshold = rhythm.get('render_when_below', 10)

    if ready_count >= threshold:
        log(f"{ready_count} videos ready (threshold: {threshold}) — no render needed")
        return False

    need = 1
    log(f"pool is low ({ready_count} ready, threshold {threshold}) — rendering 1 video")

    # fetch comment context for seeded renders
    comment_ctx = _fetch_comment_context(manifest_path, rhythm, dry_run=dry_run)
    poll_overrides = comment_ctx.get('poll_overrides', {})
    comments = comment_ctx.get('comments', [])

    title_from_comments = rhythm.get('title_from_comments', False)
    title_comment_strategy = rhythm.get('title_comment_strategy', 'random')

    # render comment-seeded videos first (if enabled)
    seeded_count = 0
    if (rhythm.get('comment_feedback', False)
            and rhythm.get('seed_from_comments', True)
            and comments):

        max_seeded = min(max(1, need // 5), 5, len(comments))

        from scripts.youtube.comments import pick_seed_comment, comment_to_seed

        for i in range(max_seeded):
            seed_comment = pick_seed_comment(comments, rng=rng)
            if not seed_comment:
                break

            seed_val = comment_to_seed(seed_comment['text'])

            if dry_run:
                log(f"[dry run] would render seeded video #{i+1}: "
                    f"seed={seed_val} from \"{seed_comment['text'][:50]}\"")
                seeded_count += 1
                continue

            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, 'scripts', 'blend', 'multi-layer.py'),
                '--preset', rhythm['render_preset'],
                '--project', project_name,
                '--output-dir', output_dir,
                '--seed', str(seed_val),
            ]
            if title_from_comments and seed_comment.get('text', '').strip():
                cmd.extend(['--title-comment', seed_comment['text'].strip()])

            for key, value in poll_overrides.items():
                if key in ('mode', 'fps', 'num_videos', 'duration'):
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])

            log(f"rendering seeded video: seed={seed_val} "
                f"from \"{seed_comment['text'][:40]}...\"")
            try:
                subprocess.run(cmd, check=True)
                seeded_count += 1
            except subprocess.CalledProcessError as e:
                log(f"seeded render failed: {e}")

    # render the rest one by one so each title can absorb a fresh comment
    remaining = need - seeded_count
    if remaining > 0:
        for i in range(remaining):
            title_comment = None
            if title_from_comments and comments:
                title_comment = _pick_title_comment(
                    comments, rng, strategy=title_comment_strategy
                )

            if dry_run:
                if title_comment:
                    log(f"[dry run] would render video {i+1}/{remaining} "
                        f"with title from comment: \"{title_comment[:50]}\"")
                else:
                    log(f"[dry run] would render video {i+1}/{remaining}")
                continue

            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, 'scripts', 'blend', 'multi-layer.py'),
                '--preset', rhythm['render_preset'],
                '--project', project_name,
                '--output-dir', output_dir,
            ]

            for key, value in poll_overrides.items():
                if key in ('mode', 'fps', 'num_videos', 'duration'):
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])

            if title_comment:
                cmd.extend(['--title-comment', title_comment])
                log(f"rendering {i+1}/{remaining} with title comment: "
                    f"\"{title_comment[:40]}...\"")
            else:
                log(f"rendering {i+1}/{remaining}")

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                log(f"render failed: {e}")
                return False

    # post-process new renders with comment feedback
    manifest = load_manifest(manifest_path)
    uploaded_set = get_uploaded_set(manifest)
    _post_process_renders(output_dir, uploaded_set, comments, rhythm, rng, dry_run=dry_run)

    return True


def maybe_upload(rhythm, rng, output_dir, manifest_path, dry_run=False):
    """
    the heart of the organic timing.

    each tick, we decide whether to upload and how many:
    - are we inside the upload window?
    - have we hit today's API quota?
    - upload_chance = remaining_today / hours_left, ±30% jitter
    - if hit: upload 1 (occasionally 2)

    morning ticks rarely fire. evening ticks almost always do
    if there's quota left.
    """
    manifest = load_manifest(manifest_path)
    uploaded_set = get_uploaded_set(manifest)
    ready_count, ready_files = count_ready_videos(output_dir, uploaded_set)

    if ready_count == 0:
        log("no videos ready to upload")
        return

    # outside the window? skip
    if not in_window(rhythm):
        log(f"outside upload window — {ready_count} videos waiting")
        return

    # check daily quota
    uploads_per_day = rhythm.get('uploads_per_day', 6)
    done_today = uploaded_today(manifest)

    if done_today >= uploads_per_day:
        log(f"hit daily quota ({done_today}/{uploads_per_day}) — done for today")
        return

    remaining_today = uploads_per_day - done_today

    # figure out hours left in the window
    window = rhythm.get('window_hours', ['06:00', '23:00'])
    end_h, end_m = map(int, window[1].split(':'))
    now = datetime.now()
    hours_left = max(1, (end_h * 60 + end_m - now.hour * 60 - now.minute) / 60)

    # probability: spread remaining uploads across remaining hours, with jitter.
    # early → low chance. late → high chance. organic.
    upload_chance = remaining_today / hours_left
    upload_chance *= rng.uniform(0.7, 1.3)
    upload_chance = min(1.0, max(0.0, upload_chance))

    if rng.random() > upload_chance:
        log(f"skipping this tick (chance was {upload_chance:.0%}). "
            f"{remaining_today} left today, {hours_left:.1f}h in window, "
            f"{ready_count} ready")
        return

    # how many — usually 1, sometimes 2
    count = 1
    if remaining_today >= 3 and rng.random() < 0.15:
        count = 2
    count = min(count, remaining_today, ready_count)

    log(f"uploading {count} video(s) (chance was {upload_chance:.0%}, "
        f"{done_today} done today, {ready_count} ready)")

    if dry_run:
        log(f"[dry run] would upload {count} video(s)")
        return

    # pick random videos and upload
    sorted_ready = sorted(ready_files)
    to_upload = rng.sample(sorted_ready, min(count, len(sorted_ready)))

    # polls (if comment feedback is on)
    poll_chance = rhythm.get('poll_chance', 0.25)
    use_polls = rhythm.get('comment_feedback', False) and poll_chance > 0

    for filename in to_upload:
        poll_text = None
        poll_data = None

        if use_polls and rng.random() < poll_chance:
            try:
                from scripts.youtube.polls import generate_poll_question, format_poll_for_description
                poll_data = generate_poll_question(rng)
                poll_text = format_poll_for_description(poll_data)
                log(f"  attaching poll: {poll_data['question']}")
            except Exception as e:
                log(f"  poll generation failed (non-fatal): {e}")

        video_id = _upload_file(filename, output_dir, manifest, manifest_path,
                                poll_text=poll_text, rhythm=rhythm)

        if video_id and poll_data and video_id not in ('unknown', None):
            _record_poll(manifest_path, video_id, poll_data)


# =============================================================================
# HELPERS
# =============================================================================

def _record_poll(manifest_path, video_id, poll_data):
    """record a poll question so we can read results later."""
    polls_state_path = os.path.join(
        os.path.dirname(manifest_path), 'polls-state.json'
    )

    polls = []
    if os.path.exists(polls_state_path):
        with open(polls_state_path) as f:
            polls = json.load(f)

    polls.append({
        'video_id': video_id,
        'poll': poll_data,
        'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'read': False,
    })

    with open(polls_state_path, 'w') as f:
        json.dump(polls, f, indent=2)


def _upload_file(filename, output_dir, manifest, manifest_path, poll_text=None, rhythm=None):
    """
    upload a single file as public and record it in the manifest.
    the actual upload is handled by youtube-upload.py --file --public.

    if poll_text is provided, it's appended to the video description.

    if rhythm['delete_after_upload'] is True, the MP4 is deleted from disk
    after a successful upload.
    """
    filepath = os.path.join(output_dir, filename)

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, 'scripts', 'upload', 'youtube-upload.py'),
        '--file', filepath,
        '--public',
    ]

    if poll_text:
        cmd.extend(['--poll-text', poll_text])

    log(f"uploading: {filename}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = result.stdout

        # parse video ID from output
        video_id = None
        for line in output.split('\n'):
            if 'youtube.com/watch?v=' in line:
                video_id = line.split('watch?v=')[-1].strip()
                break

        if not video_id:
            for line in output.split('\n'):
                if line.strip().startswith('done:'):
                    video_id = line.split('done:')[-1].strip()
                    break

        if video_id and video_id != 'None':
            manifest.append({
                'filename': filename,
                'video_id': video_id,
                'status': 'uploaded',
                'uploaded_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                'url': f"https://youtube.com/watch?v={video_id}",
            })
            save_manifest(manifest, manifest_path)
            log(f"  done: https://youtube.com/watch?v={video_id}")
            _maybe_delete_file(filepath, rhythm)
            return video_id
        else:
            log(f"  uploaded but couldn't parse video ID")
            log(f"  stdout: {output[:500]}")
            manifest.append({
                'filename': filename,
                'video_id': 'unknown',
                'status': 'uploaded',
                'uploaded_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            })
            save_manifest(manifest, manifest_path)
            _maybe_delete_file(filepath, rhythm)
            return 'unknown'

    except subprocess.CalledProcessError as e:
        log(f"  upload failed: {e}")
        if e.stdout:
            log(f"  stdout: {e.stdout[:500]}")
        if e.stderr:
            log(f"  stderr: {e.stderr[:500]}")
        return None


def _maybe_delete_file(filepath, rhythm):
    """delete an uploaded video from disk if the rhythm config says to."""
    if rhythm and rhythm.get('delete_after_upload', True):
        try:
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            os.remove(filepath)
            log(f"  deleted from disk ({size_mb:.0f}MB freed)")
        except OSError as e:
            log(f"  couldn't delete {filepath}: {e}")


def print_status(rhythm, output_dir, manifest_path):
    """print a human-readable status summary."""
    manifest = load_manifest(manifest_path)
    uploaded_set = get_uploaded_set(manifest)
    ready_count, _ = count_ready_videos(output_dir, uploaded_set)
    done_today = uploaded_today(manifest)
    total_uploaded = len([e for e in manifest if e.get('status') == 'uploaded'])

    print(f"\n  autopilot status")
    print(f"  {'='*50}")
    print(f"  videos ready: {ready_count}")
    print(f"  total uploaded: {total_uploaded}")
    print(f"  uploaded today: {done_today}/{rhythm.get('uploads_per_day', 6)}")
    window = rhythm.get('window_hours', ['06:00', '23:00'])
    print(f"  upload window: {window[0]}-{window[1]}")
    print(f"  currently in window: {'yes' if in_window(rhythm) else 'no'}")
    print(f"  render threshold: {rhythm.get('render_when_below', 10)}")
    print(f"  delete after upload: {rhythm.get('delete_after_upload', True)}")

    # show last 5 uploads
    recent = [e for e in manifest if e.get('status') == 'uploaded'][-5:]
    if recent:
        print(f"\n  recent uploads:")
        for e in reversed(recent):
            url = e.get('url', '')
            ts = e.get('uploaded_at', '?')[:16]
            print(f"    {ts}  {e['filename']}")
            if url:
                print(f"              {url}")

    print(f"  {'='*50}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='steady organic uploader — 6 videos/day at random times',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python tools/autopilot.py --project first-blend-test
  python tools/autopilot.py --project first-blend-test --dry-run
  python tools/autopilot.py --project first-blend-test --status

cron (every hour):
  0 * * * * cd /path/to/collide-o-scope && python3 tools/autopilot.py --project first-blend-test >> /var/log/autopilot.log 2>&1
        """
    )

    parser.add_argument('--project', required=True,
                        help='project name (e.g. first-blend-test)')
    parser.add_argument('--dry-run', action='store_true',
                        help='preview what would happen without doing anything')
    parser.add_argument('--status', action='store_true',
                        help='print current status and exit')

    args = parser.parse_args()

    # paths
    project_dir = os.path.join(PROJECT_ROOT, 'projects', args.project)
    output_dir = os.path.join(project_dir, 'output')
    manifest_path = os.path.join(project_dir, 'upload-manifest.json')
    state_path = os.path.join(project_dir, 'autopilot-state.json')

    if not os.path.exists(output_dir):
        print(f"error: project output dir not found: {output_dir}")
        sys.exit(1)

    # load config and state
    rhythm = load_rhythm(project_dir)
    state = load_state(state_path)

    # status mode
    if args.status:
        print_status(rhythm, output_dir, manifest_path)
        return

    # only one autopilot at a time — if a previous tick is still rendering,
    # this tick bows out instead of piling on
    lock = acquire_lock(project_dir)
    if lock is None:
        log("another autopilot instance is already running — skipping this tick")
        return

    # seeded RNG — different roll each hourly tick
    hour_seed = state['seed'] + int(datetime.now().timestamp() / 3600)
    rng = create_rng(hour_seed)

    log("tick")

    # step 1: render if pool is low
    maybe_render(rhythm, rng, output_dir, manifest_path, args.project, dry_run=args.dry_run)

    # step 2: maybe upload
    maybe_upload(rhythm, rng, output_dir, manifest_path, dry_run=args.dry_run)

    # save state
    if not args.dry_run:
        save_state(state, state_path)


if __name__ == '__main__':
    main()
