"""
autopilot.py - organic burst publishing daemon

a state machine that renders and publishes videos in bursts,
modeling the rhythm of someone who makes a bunch of stuff,
dumps it online, then disappears for a while.

designed to run via cron (every hour) on a VPS. each tick,
it checks the current phase and either does nothing, renders,
or uploads — then updates its state file.

the key insight: timing IS the content. we upload directly as
public the moment the cron fires, and the organic pattern comes
from which ticks we decide to act on. some hours get 2 videos,
some get 0, some days are silent. the audience sees a person
who posts when they feel like it.

phases:
    quiet     → nothing happens. 4-21 days of silence.
                occasionally a solo post sneaks out.
    rendering → batch-render produces 10-40 videos.
    uploading → picks 0-N videos per tick, uploads as public.
                organic timing via probability + jitter.

usage:
    python tools/autopilot.py --project first-blend-test
    python tools/autopilot.py --project first-blend-test --dry-run
    python tools/autopilot.py --project first-blend-test --force-phase rendering
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
# the "personality" of the posting pattern. these define how the autopilot
# behaves across cycles. each value can be overridden by a rhythm.json file
# in the project directory.
#
# ranges like [min, max] = "pick randomly within this window each cycle."
# =============================================================================

RHYTHM_DEFAULTS = {

    # --- burst shape ---

    # how many videos to publish per burst
    # smaller bursts feel like quick updates, larger ones feel like a dump
    "burst_size": [10, 25],

    # what hours uploads can happen during a burst
    # wider = more chaotic, narrower = more focused
    "burst_window_hours": ["06:00", "23:00"],

    # --- quiet periods ---

    # how many days of silence between bursts
    # short gaps = prolific, long gaps = mysterious
    "cooldown_days": [4, 21],

    # daily chance of posting a single video during quiet periods
    # like someone woke up at 2am and posted one thing
    "solo_post_chance": 0.05,

    # --- rendering ---

    # how many videos to render per batch
    # autopilot renders when the pool of ready videos drops below a threshold
    "render_batch_size": [20, 40],

    # render a new batch when fewer than this many videos are ready
    "render_when_below": 10,

    # preset to use for rendering
    "render_preset": "classic-white",

    # --- upload ---

    # max uploads per day (youtube API quota: 10,000 units, ~1,600 per upload)
    "uploads_per_day": 6,

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
}


def log(msg):
    """timestamped log line."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}")


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
    the state tracks which phase we're in and when transitions happen.
    """
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)

    # fresh state — start in quiet, first burst soon
    return {
        "phase": "quiet",
        "phase_entered": datetime.now().isoformat(),
        "next_burst": (datetime.now() + timedelta(hours=1)).isoformat(),
        "last_burst_end": None,
        "current_batch_size": None,
        "burst_target": None,
        "burst_uploaded": 0,
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
    load the upload manifest — a simple list of what's been uploaded.
    this replaces schedule.json as the tracking mechanism. simpler:
    just filename, title, video_id, uploaded_at.
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
    """simple seeded RNG for the autopilot's own decisions."""
    import random
    rng = random.Random(seed)
    return rng


def pick_from_range(rng, value):
    """resolve a [min, max] range or return a fixed value."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        if isinstance(value[0], int) and isinstance(value[1], int):
            return rng.randint(value[0], value[1])
        return rng.uniform(value[0], value[1])
    return value


def in_window(rhythm):
    """check if the current time is within the burst upload window."""
    window = rhythm.get('burst_window_hours', ['06:00', '23:00'])
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
# fetches comments from recent uploads and prepares them for use
# as seeds, overlays, poll results, and response prompts.
# all of this is gated behind the comment_feedback rhythm flag.
# =============================================================================

def _fetch_comment_context(manifest_path, rhythm, dry_run=False):
    """
    fetch comments from recent uploads and prepare them for the
    rendering and uploading phases. returns a dict with everything
    the phase handlers need.

    this is the single API call point — all comments are fetched once
    per tick and shared across features.

    returns:
        dict with comments (list), poll_overrides (dict), or empty
        if comment_feedback is off or no comments exist.
    """
    if not rhythm.get('comment_feedback', False):
        return {'comments': [], 'poll_overrides': {}, 'poll_results': []}

    lookback = rhythm.get('comment_lookback_videos', 10)
    min_likes = rhythm.get('min_comment_likes', 0)

    if dry_run:
        log("[dry run] would fetch comments from recent uploads")
        return {'comments': [], 'poll_overrides': {}, 'poll_results': []}

    try:
        from scripts.youtube.comments import fetch_comments_from_manifest, filter_comments

        all_comments = fetch_comments_from_manifest(
            manifest_path,
            lookback=lookback,
            max_per_video=50,
        )

        comments = filter_comments(all_comments, min_likes=min_likes)
        log(f"fetched {len(comments)} comment(s) from recent videos")

    except Exception as e:
        log(f"comment fetch failed (non-fatal): {e}")
        comments = []

    # read poll results if any active polls exist
    poll_overrides = {}
    poll_results = []

    try:
        polls_state_path = os.path.join(os.path.dirname(manifest_path), 'polls-state.json')
        if os.path.exists(polls_state_path):
            from scripts.youtube.polls import read_poll_results, find_active_polls, aggregate_poll_overrides

            active_polls = find_active_polls(manifest_path, polls_state_path)
            if active_polls:
                log(f"reading {len(active_polls)} active poll(s)")
                youtube = None

                for poll_entry in active_polls:
                    try:
                        # lazy auth — reuse across polls
                        if youtube is None:
                            from scripts.youtube.comments import get_authenticated_service
                            youtube = get_authenticated_service()

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

                        # mark as read
                        poll_entry['read'] = True

                    except Exception as e:
                        log(f"  poll read failed for {poll_entry['video_id']}: {e}")

                # save updated polls state
                with open(polls_state_path, 'w') as f:
                    json.dump(active_polls, f, indent=2)

                # aggregate overrides from all poll results
                poll_overrides = aggregate_poll_overrides(poll_results)
                if poll_overrides:
                    log(f"poll overrides: {poll_overrides}")

    except Exception as e:
        log(f"poll reading failed (non-fatal): {e}")

    return {
        'comments': comments,
        'poll_overrides': poll_overrides,
        'poll_results': poll_results,
    }


def _post_process_renders(output_dir, uploaded_set, comments, rhythm, rng, dry_run=False):
    """
    post-process newly rendered videos with comment feedback.

    runs after batch-render completes. finds new MP4s that haven't been
    uploaded yet and applies comment burning and response metadata to
    a random subset.

    this is where comments physically enter the videos — as faded text
    overlays and as words absorbed into titles/descriptions.
    """
    import glob as globmod

    if not rhythm.get('comment_feedback', False) or not comments:
        return

    # find new videos (rendered but not uploaded)
    all_mp4s = sorted(
        globmod.glob(os.path.join(output_dir, '*.mp4'))
    )
    new_videos = [
        f for f in all_mp4s
        if os.path.basename(f) not in uploaded_set
    ]

    if not new_videos:
        return

    burn_chance = rhythm.get('burn_comment_chance', 0.4)
    response_chance = rhythm.get('response_chance', 0.12)
    comment_texts = [c['text'] for c in comments]

    # --- burn comments into a subset of videos ---
    if rhythm.get('burn_comments', True) and comment_texts:
        burn_count = 0
        for video_path in new_videos:
            if rng.random() < burn_chance:
                # pick 1-4 comments to burn in
                n = min(len(comment_texts), rng.randint(1, 4))
                selected = rng.sample(comment_texts, n)

                if dry_run:
                    log(f"  [dry run] would burn {n} comment(s) into {os.path.basename(video_path)}")
                    burn_count += 1
                    continue

                try:
                    from scripts.post import burn_comments as burn_mod
                    result = burn_mod.burn_comments(
                        video_path, selected,
                        rng=rng,
                    )
                    if result:
                        burn_count += 1
                except Exception as e:
                    log(f"  burn failed for {os.path.basename(video_path)}: {e}")

        if burn_count > 0:
            log(f"burned comments into {burn_count} video(s)")

    # --- re-embed response metadata into a subset ---
    if response_chance > 0 and comment_texts:
        response_count = 0
        for video_path in new_videos:
            if rng.random() < response_chance:
                # pick a comment to respond to
                source_comment = rng.choice(comments)

                if dry_run:
                    log(f"  [dry run] would embed response metadata in {os.path.basename(video_path)}")
                    response_count += 1
                    continue

                try:
                    from scripts.text.metadata import generate_response_metadata
                    meta = generate_response_metadata(source_comment['text'], rng=rng)

                    # re-embed metadata via ffmpeg -c copy (no re-encode)
                    tmp_path = video_path + '.meta.mp4'
                    cmd = [
                        'ffmpeg', '-y', '-i', video_path,
                        '-c', 'copy',
                        '-metadata', f"title={meta['title']}",
                        '-metadata', f"comment={meta['description']}",
                        '-metadata', 'artist=luis queral',
                        tmp_path,
                    ]
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    os.replace(tmp_path, video_path)
                    response_count += 1
                    log(f"  response: {os.path.basename(video_path)} -> \"{meta['title']}\"")

                except Exception as e:
                    log(f"  response metadata failed for {os.path.basename(video_path)}: {e}")
                    # clean up temp file
                    tmp_path = video_path + '.meta.mp4'
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

        if response_count > 0:
            log(f"embedded response metadata in {response_count} video(s)")


# =============================================================================
# PHASE HANDLERS
# =============================================================================

def handle_quiet(state, rhythm, rng, output_dir, manifest_path, dry_run=False):
    """
    quiet phase — wait for cooldown to expire.
    occasionally fire off a solo post (uploaded directly as public).
    """
    now = datetime.now()
    next_burst = datetime.fromisoformat(state['next_burst'])

    if now >= next_burst:
        log("cooldown expired — moving to rendering")
        state['phase'] = 'rendering'
        state['phase_entered'] = now.isoformat()
        return state

    # solo post chance — roll the dice.
    # we check once per cron tick (hourly). scale the daily chance
    # so P(at least one hit in 24 ticks) ≈ daily_chance.
    # per-tick chance = 1 - (1 - daily_chance)^(1/24)
    daily_chance = rhythm.get('solo_post_chance', 0.05)
    per_tick_chance = 1 - (1 - daily_chance) ** (1/24)

    manifest = load_manifest(manifest_path)
    uploaded_set = get_uploaded_set(manifest)
    ready_count, ready_files = count_ready_videos(output_dir, uploaded_set)

    if ready_count > 0 and rng.random() < per_tick_chance:
        log("solo post — uploading one video during quiet period")
        if not dry_run:
            solo_file = rng.choice(sorted(ready_files))
            _upload_file(solo_file, output_dir, manifest, manifest_path)
        else:
            log("[dry run] would upload one solo video")

    days_left = (next_burst - now).days
    hours_left = int((next_burst - now).total_seconds() / 3600)
    log(f"quiet — {hours_left}h until next burst ({days_left} days). {ready_count} videos ready")

    return state


def handle_rendering(state, rhythm, rng, output_dir, manifest_path, project_name, dry_run=False):
    """
    rendering phase — check if we need to render, kick off batch-render.
    transitions to uploading when done.

    when comment feedback is enabled, this phase also:
    - fetches comments from recent uploads
    - reads poll results and computes config overrides
    - renders some videos with comment-derived seeds
    - post-processes new videos with comment burns and response metadata
    """
    manifest = load_manifest(manifest_path)
    uploaded_set = get_uploaded_set(manifest)
    ready_count, _ = count_ready_videos(output_dir, uploaded_set)

    # decide batch size for this burst
    if state.get('current_batch_size') is None:
        batch_size = pick_from_range(rng, rhythm['render_batch_size'])
        state['current_batch_size'] = batch_size
        log(f"decided to render {batch_size} videos")

    batch_size = state['current_batch_size']

    # if we already have enough, skip rendering
    if ready_count >= batch_size:
        log(f"already have {ready_count} ready videos (need {batch_size}), skipping render")
        _enter_uploading(state, rhythm, rng, ready_count)
        return state

    need = batch_size - ready_count
    log(f"need to render {need} more videos ({ready_count} ready, target {batch_size})")

    # --- comment feedback: pre-render ---
    comment_ctx = _fetch_comment_context(manifest_path, rhythm, dry_run=dry_run)
    poll_overrides = comment_ctx.get('poll_overrides', {})
    comments = comment_ctx.get('comments', [])

    # render comment-seeded videos first (if enabled and comments exist)
    seeded_count = 0
    if (rhythm.get('comment_feedback', False)
            and rhythm.get('seed_from_comments', True)
            and comments):

        # render a few videos with comment-derived seeds.
        # cap at ~20% of the batch or 5, whichever is smaller.
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

            # render one video with this specific seed
            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, 'scripts', 'blend', 'multi-layer.py'),
                '--preset', rhythm['render_preset'],
                '--project', project_name,
                '--seed', str(seed_val),
            ]

            # apply poll overrides as CLI flags
            for key, value in poll_overrides.items():
                flag = f"--{key.replace('_', '-')}"
                # only pass overrides that multi-layer.py accepts as CLI flags
                if key in ('mode', 'fps', 'num_videos', 'duration'):
                    cmd.extend([flag, str(value)])

            log(f"rendering seeded video: seed={seed_val} "
                f"from \"{seed_comment['text'][:40]}...\"")
            try:
                subprocess.run(cmd, check=True)
                seeded_count += 1
            except subprocess.CalledProcessError as e:
                log(f"seeded render failed: {e}")

    # render the rest via batch-render
    remaining = need - seeded_count
    if remaining > 0:
        if dry_run:
            extra_flags = ''
            for key, value in poll_overrides.items():
                if key in ('mode', 'fps', 'num_videos', 'duration'):
                    extra_flags += f" --{key.replace('_', '-')} {value}"
            log(f"[dry run] would run: batch-render.py --count {remaining} "
                f"--preset {rhythm['render_preset']} --project {project_name}{extra_flags}")
        else:
            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, 'tools', 'batch-render.py'),
                '--count', str(remaining),
                '--preset', rhythm['render_preset'],
                '--project', project_name,
            ]

            # apply poll overrides
            for key, value in poll_overrides.items():
                if key in ('mode', 'fps', 'num_videos', 'duration'):
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])

            log(f"running: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
                log("batch render complete")
            except subprocess.CalledProcessError as e:
                log(f"batch render failed: {e}")
                return state

    # --- comment feedback: post-render ---
    # burn comments and embed response metadata into newly rendered videos
    manifest = load_manifest(manifest_path)
    uploaded_set = get_uploaded_set(manifest)
    _post_process_renders(output_dir, uploaded_set, comments, rhythm, rng, dry_run=dry_run)

    if dry_run:
        _enter_uploading(state, rhythm, rng, batch_size)
    else:
        new_ready, _ = count_ready_videos(output_dir, uploaded_set)
        _enter_uploading(state, rhythm, rng, new_ready)

    return state


def handle_uploading(state, rhythm, rng, output_dir, manifest_path, dry_run=False):
    """
    uploading phase — the heart of the organic timing.

    each tick, we decide whether to upload and how many. the logic:
    - are we inside the burst window? (e.g. 06:00-23:00)
    - have we hit today's API quota?
    - how many hours are left in the window? distribute remaining
      uploads across remaining hours, with noise.
    - occasionally upload 2 in one tick for variety.

    when comment feedback is on, some uploads get a poll question
    appended to their description. the poll is recorded in
    polls-state.json so we can read results next cycle.

    this creates natural irregularity: some hours get videos,
    some don't. early ticks are less likely, late ticks more likely
    (pressure to hit the daily target builds through the day).
    """
    manifest = load_manifest(manifest_path)
    uploaded_set = get_uploaded_set(manifest)
    ready_count, ready_files = count_ready_videos(output_dir, uploaded_set)

    # check if the burst is done
    burst_target = state.get('burst_target', 0)
    burst_uploaded = state.get('burst_uploaded', 0)

    if burst_uploaded >= burst_target or ready_count == 0:
        log(f"burst complete — uploaded {burst_uploaded}/{burst_target}")
        _enter_quiet(state, rhythm, rng)
        return state

    # outside the window? skip
    if not in_window(rhythm):
        log(f"outside burst window — waiting. {ready_count} ready, {burst_uploaded}/{burst_target} uploaded")
        return state

    # check daily quota
    uploads_per_day = rhythm.get('uploads_per_day', 6)
    done_today = uploaded_today(manifest)

    if done_today >= uploads_per_day:
        log(f"hit daily quota ({done_today}/{uploads_per_day}) — waiting for tomorrow")
        log(f"  {ready_count} ready, {burst_uploaded}/{burst_target} burst progress")
        return state

    remaining_today = uploads_per_day - done_today

    # figure out how many hours are left in today's window
    window = rhythm.get('burst_window_hours', ['06:00', '23:00'])
    end_h, end_m = map(int, window[1].split(':'))
    now = datetime.now()
    hours_left = max(1, (end_h * 60 + end_m - now.hour * 60 - now.minute) / 60)

    # probability of uploading this tick:
    # spread remaining uploads across remaining hours, with ±30% jitter.
    # early in the day with lots of hours left → low chance per tick.
    # late in the day with uploads still needed → high chance.
    upload_chance = remaining_today / hours_left
    upload_chance *= rng.uniform(0.7, 1.3)
    upload_chance = min(1.0, max(0.0, upload_chance))

    if rng.random() > upload_chance:
        log(f"skipping this tick (chance was {upload_chance:.0%}). "
            f"{remaining_today} left today, {hours_left:.1f}h remaining in window")
        return state

    # decide how many — usually 1, sometimes 2
    count = 1
    if remaining_today >= 3 and rng.random() < 0.15:
        count = 2
    count = min(count, remaining_today, ready_count)

    log(f"uploading {count} video(s) this tick (chance was {upload_chance:.0%})")

    if dry_run:
        log(f"[dry run] would upload {count} video(s)")
        state['burst_uploaded'] = burst_uploaded + count
        return state

    # pick random videos from the ready pool and upload
    sorted_ready = sorted(ready_files)
    to_upload = rng.sample(sorted_ready, min(count, len(sorted_ready)))

    # decide if any of these uploads get a poll question
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
                                poll_text=poll_text)

        if video_id:
            state['burst_uploaded'] = state.get('burst_uploaded', 0) + 1

            # record the poll so we can read results next cycle
            if poll_data and video_id not in ('unknown', None):
                _record_poll(manifest_path, video_id, poll_data)

    return state


# =============================================================================
# HELPERS
# =============================================================================

def _enter_uploading(state, rhythm, rng, ready_count):
    """transition to uploading phase with a burst target."""
    burst_size = pick_from_range(rng, rhythm['burst_size'])
    # don't target more than what's actually available
    burst_target = min(burst_size, ready_count)

    state['phase'] = 'uploading'
    state['phase_entered'] = datetime.now().isoformat()
    state['current_batch_size'] = None
    state['burst_target'] = burst_target
    state['burst_uploaded'] = 0

    log(f"entering uploading phase — burst target: {burst_target} videos")


def _enter_quiet(state, rhythm, rng):
    """transition to quiet phase with a randomized cooldown."""
    cooldown = pick_from_range(rng, rhythm['cooldown_days'])
    next_burst = datetime.now() + timedelta(days=cooldown)

    state['phase'] = 'quiet'
    state['phase_entered'] = datetime.now().isoformat()
    state['next_burst'] = next_burst.isoformat()
    state['last_burst_end'] = datetime.now().isoformat()
    state['burst_target'] = None
    state['burst_uploaded'] = 0

    log(f"entering quiet for {cooldown} days — next burst around {next_burst.strftime('%Y-%m-%d')}")


def _record_poll(manifest_path, video_id, poll_data):
    """
    record that a video got a poll question so we can read results later.
    saves to polls-state.json alongside the manifest.
    """
    polls_state_path = os.path.join(os.path.dirname(manifest_path), 'polls-state.json')

    polls_state = []
    if os.path.exists(polls_state_path):
        try:
            with open(polls_state_path) as f:
                polls_state = json.load(f)
        except (json.JSONDecodeError, IOError):
            polls_state = []

    polls_state.append({
        'video_id': video_id,
        'poll': poll_data,
        'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'read': False,
    })

    with open(polls_state_path, 'w') as f:
        json.dump(polls_state, f, indent=2)

    log(f"  recorded poll for {video_id}")


def _upload_file(filename, output_dir, manifest, manifest_path, poll_text=None):
    """
    upload a single file as public and record it in the manifest.
    the actual upload is handled by youtube-upload.py --file --public.

    if poll_text is provided, it's passed as --poll-text to youtube-upload.py
    which appends it to the video description. this is how the comment
    feedback loop plants poll questions in uploaded videos.
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
        # capture output to extract video ID
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = result.stdout

        # parse video ID from the output line "uploaded (public): https://youtube.com/watch?v=XXXXX"
        video_id = None
        for line in output.split('\n'):
            if 'youtube.com/watch?v=' in line:
                video_id = line.split('watch?v=')[-1].strip()
                break

        if not video_id:
            # fallback: check the "done: XXXX" line
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
            return video_id
        else:
            log(f"  uploaded but couldn't parse video ID from output")
            log(f"  stdout: {output[:500]}")
            # still record it so we don't re-upload
            manifest.append({
                'filename': filename,
                'video_id': 'unknown',
                'status': 'uploaded',
                'uploaded_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            })
            save_manifest(manifest, manifest_path)
            return 'unknown'

    except subprocess.CalledProcessError as e:
        log(f"  upload failed: {e}")
        if e.stdout:
            log(f"  stdout: {e.stdout[:500]}")
        if e.stderr:
            log(f"  stderr: {e.stderr[:500]}")
        return None


def print_status(state, rhythm, output_dir, manifest_path):
    """print a human-readable status summary."""
    manifest = load_manifest(manifest_path)
    uploaded_set = get_uploaded_set(manifest)
    ready_count, _ = count_ready_videos(output_dir, uploaded_set)
    done_today = uploaded_today(manifest)
    total_uploaded = len([e for e in manifest if e.get('status') == 'uploaded'])

    print(f"\n  autopilot status")
    print(f"  {'='*50}")
    print(f"  phase: {state['phase']}")
    print(f"  phase entered: {state.get('phase_entered', '?')}")

    if state['phase'] == 'quiet':
        next_burst = state.get('next_burst', '?')
        if next_burst != '?':
            nb = datetime.fromisoformat(next_burst)
            delta = nb - datetime.now()
            days = delta.days
            hours = int(delta.total_seconds() / 3600)
            print(f"  next burst: {next_burst[:10]} ({days}d {hours % 24}h from now)")
        print(f"  solo post chance: {rhythm.get('solo_post_chance', 0.05)*100:.0f}%/day")

    if state['phase'] == 'uploading':
        burst_target = state.get('burst_target', '?')
        burst_uploaded = state.get('burst_uploaded', 0)
        print(f"  burst progress: {burst_uploaded}/{burst_target}")

    print(f"  videos ready: {ready_count}")
    print(f"  total uploaded: {total_uploaded}")
    print(f"  uploaded today: {done_today}")
    print(f"  upload limit/day: {rhythm.get('uploads_per_day', 6)}")

    if state.get('last_burst_end'):
        print(f"  last burst ended: {state['last_burst_end'][:10]}")

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
        description='organic burst publishing daemon',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python tools/autopilot.py --project first-blend-test
  python tools/autopilot.py --project first-blend-test --dry-run
  python tools/autopilot.py --project first-blend-test --status
  python tools/autopilot.py --project first-blend-test --force-phase rendering

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
    parser.add_argument('--force-phase', choices=['quiet', 'rendering', 'uploading'],
                        help='force a specific phase (for testing or manual override)')

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

    # force phase if requested
    if args.force_phase:
        log(f"forcing phase: {args.force_phase}")
        state['phase'] = args.force_phase
        state['phase_entered'] = datetime.now().isoformat()

    # status mode — just print and exit
    if args.status:
        print_status(state, rhythm, output_dir, manifest_path)
        return

    # create a seeded RNG from the state seed + current hour
    # (so each hourly tick gets a different roll, but it's reproducible)
    hour_seed = state['seed'] + int(datetime.now().timestamp() / 3600)
    rng = create_rng(hour_seed)

    log(f"tick — phase: {state['phase']}")

    # dispatch to phase handler
    phase = state['phase']

    if phase == 'quiet':
        state = handle_quiet(state, rhythm, rng, output_dir, manifest_path, dry_run=args.dry_run)

    elif phase == 'rendering':
        state = handle_rendering(state, rhythm, rng, output_dir, manifest_path, args.project, dry_run=args.dry_run)

    elif phase == 'uploading':
        state = handle_uploading(state, rhythm, rng, output_dir, manifest_path, dry_run=args.dry_run)

    else:
        log(f"unknown phase: {phase}, resetting to quiet")
        state['phase'] = 'quiet'

    # save state
    if not args.dry_run:
        save_state(state, state_path)
        log(f"state saved — phase: {state['phase']}")
    else:
        log(f"[dry run] would save state — phase: {state['phase']}")


if __name__ == '__main__':
    main()
