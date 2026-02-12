"""
polls.py - soft comment polls for audience-driven config

generates questions that map to real render config knobs, and
reads comment responses via simple keyword matching. the questions
fit the tone — not "VOTE NOW" but "next: lights or darks?"

each poll question maps to a config key and two (or more) options.
the winning option becomes a config override for the next render batch.

usage as module:
    from scripts.youtube.polls import generate_poll_question, read_poll_results

    poll = generate_poll_question()
    # poll = { "question": "lights or darks?", "config_key": "luminance_target", ... }

    results = read_poll_results(video_id, poll, youtube=youtube)
    # results = { "winner": "darks", "config_override": {"luminance_target": "darks"}, ... }

usage as CLI:
    python scripts/youtube/polls.py --generate
    python scripts/youtube/polls.py --generate --count 5
    python scripts/youtube/polls.py --read VIDEO_ID --poll-file poll.json
"""

import os
import sys
import json
import argparse
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# POLL DEFINITIONS
#
# each poll maps a human-readable question to a config key and options.
# the question goes in the video description. the options define what
# keywords to look for in comments and what config override each produces.
#
# kept deliberately minimal — a handful of knobs the audience can push.
# =============================================================================

POLLS = [
    {
        # controls luminance keying direction
        "question": "next: lights or darks?",
        "config_key": "luminance_target",
        "options": [
            {
                "label": "lights",
                "keywords": ["lights", "light", "bright", "brighter", "white"],
                "config_value": "lights",
            },
            {
                "label": "darks",
                "keywords": ["darks", "dark", "darker", "black", "shadow", "shadows"],
                "config_value": "darks",
            },
        ],
        # also force luminance mode so the vote actually matters
        "extra_overrides": {"mode": "luminance"},
    },
    {
        # controls number of layers
        "question": "more layers or fewer?",
        "config_key": "num_videos",
        "options": [
            {
                "label": "more",
                "keywords": ["more", "moar", "lots", "many", "extra", "thicker", "dense", "denser"],
                "config_value": 5,
            },
            {
                "label": "fewer",
                "keywords": ["fewer", "less", "thin", "thinner", "minimal", "simple", "clean"],
                "config_value": 2,
            },
        ],
    },
    {
        # controls frame rate feel
        "question": "faster or slower?",
        "config_key": "fps",
        "options": [
            {
                "label": "faster",
                "keywords": ["faster", "fast", "quick", "speed", "rapid"],
                "config_value": 30,
            },
            {
                "label": "slower",
                "keywords": ["slower", "slow", "dreamy", "choppy", "lazy"],
                "config_value": 14,
            },
        ],
    },
    {
        # controls keying tightness
        "question": "tighter or looser?",
        "config_key": "similarity",
        "options": [
            {
                "label": "tighter",
                "keywords": ["tighter", "tight", "sharp", "crisp", "hard", "less"],
                "config_value": [0.1, 0.2],
            },
            {
                "label": "looser",
                "keywords": ["looser", "loose", "soft", "blurry", "melty", "more"],
                "config_value": [0.35, 0.55],
            },
        ],
    },
    {
        # controls output duration
        "question": "longer or shorter?",
        "config_key": "duration",
        "options": [
            {
                "label": "longer",
                "keywords": ["longer", "long", "extended", "marathon", "more time"],
                "config_value": [180, 300],
            },
            {
                "label": "shorter",
                "keywords": ["shorter", "short", "brief", "quick", "less time"],
                "config_value": [30, 60],
            },
        ],
    },
    {
        # controls color saturation
        "question": "vivid or muted?",
        "config_key": "saturation",
        "options": [
            {
                "label": "vivid",
                "keywords": ["vivid", "colorful", "saturated", "color", "colours", "colors", "bright"],
                "config_value": [1.3, 1.8],
            },
            {
                "label": "muted",
                "keywords": ["muted", "grey", "gray", "dull", "faded", "washed", "desaturated"],
                "config_value": [0.3, 0.6],
            },
        ],
    },
    {
        # controls keying mode
        "question": "next: color key or background removal?",
        "config_key": "mode",
        "options": [
            {
                "label": "color key",
                "keywords": ["color", "colour", "colorkey", "key", "chroma", "fixed", "kmeans"],
                "config_value": "kmeans",
            },
            {
                "label": "background removal",
                "keywords": ["background", "bg", "rembg", "remove", "removal", "ml", "ai", "mask"],
                "config_value": "rembg",
            },
        ],
    },
]


def generate_poll_question(rng=None):
    """
    pick a random poll question.

    returns a dict with the question text, config key, options,
    and any extra overrides. the question is ready to be appended
    to a video description.

    Args:
        rng: random.Random instance

    Returns:
        dict with question, config_key, options, extra_overrides
    """
    if rng is None:
        rng = random.Random()

    poll = rng.choice(POLLS)

    # return a copy so callers can't mutate the source
    return {
        "question": poll["question"],
        "config_key": poll["config_key"],
        "options": poll["options"],
        "extra_overrides": poll.get("extra_overrides", {}),
    }


def format_poll_for_description(poll):
    """
    format a poll question for appending to a video description.
    keeps it short and ambient — not a call to action. the "comment:"
    prefix nudges people to reply without shouting about it.

    Args:
        poll: poll dict from generate_poll_question()

    Returns:
        string to append to the description
    """
    return f"\n\ncomment: {poll['question']}"


def read_poll_results(video_id, poll, youtube=None, credentials_dir=None):
    """
    read comments on a video and tally votes for a poll.

    uses simple keyword matching — looks for option keywords anywhere
    in the comment text. a comment counts toward whichever option
    it matches first. if a comment matches multiple options, only
    the first match counts (prevents gaming by saying both).

    Args:
        video_id: youtube video ID to read comments from
        poll: poll dict (from generate_poll_question or loaded from file)
        youtube: authenticated youtube API service (or None to create one)
        credentials_dir: directory with credentials

    Returns:
        dict with:
            winner: label of the winning option (or None if no votes)
            config_override: dict of config key/value to apply
            votes: dict of label -> count
            total_votes: total comments that matched
    """
    from scripts.youtube.comments import fetch_comments, get_authenticated_service as get_auth

    if youtube is None:
        youtube = get_auth(credentials_dir)

    comments = fetch_comments(video_id, youtube=youtube, max_results=200)

    # tally votes
    votes = {opt['label']: 0 for opt in poll['options']}

    for comment in comments:
        text_lower = comment['text'].lower()
        matched = False

        for opt in poll['options']:
            if matched:
                break
            for kw in opt['keywords']:
                if kw in text_lower:
                    votes[opt['label']] += 1
                    matched = True
                    break

    total = sum(votes.values())

    if total == 0:
        return {
            'winner': None,
            'config_override': {},
            'votes': votes,
            'total_votes': 0,
        }

    # find winner — ties go to the first option (bias toward the first-listed choice)
    winner_label = max(votes, key=lambda k: votes[k])
    winner_opt = next(opt for opt in poll['options'] if opt['label'] == winner_label)

    config_override = {poll['config_key']: winner_opt['config_value']}

    # merge extra overrides (e.g. forcing luminance mode)
    extra = poll.get('extra_overrides', {})
    config_override.update(extra)

    return {
        'winner': winner_label,
        'config_override': config_override,
        'votes': votes,
        'total_votes': total,
    }


def find_active_polls(manifest_path, polls_state_path):
    """
    find videos that have active polls by checking the polls state file.

    the polls state file tracks which videos got poll questions and
    which poll was used. the autopilot writes this when uploading.

    Args:
        manifest_path: path to upload-manifest.json
        polls_state_path: path to polls-state.json

    Returns:
        list of dicts with video_id, poll (the poll definition)
    """
    if not os.path.exists(polls_state_path):
        return []

    with open(polls_state_path) as f:
        polls_state = json.load(f)

    # return polls that haven't been read yet
    return [
        p for p in polls_state
        if not p.get('read', False)
    ]


def aggregate_poll_overrides(poll_results_list):
    """
    merge config overrides from multiple poll results.

    if multiple polls target the same config key, the one with
    the most total votes wins. if there are no votes on any poll,
    returns an empty dict.

    Args:
        poll_results_list: list of read_poll_results() return values

    Returns:
        merged config override dict
    """
    # group by config key, keep the result with most votes
    by_key = {}

    for result in poll_results_list:
        if not result['winner']:
            continue

        for key, value in result['config_override'].items():
            existing = by_key.get(key)
            if existing is None or result['total_votes'] > existing['total_votes']:
                by_key[key] = {
                    'value': value,
                    'total_votes': result['total_votes'],
                }

    return {key: info['value'] for key, info in by_key.items()}


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='generate poll questions and read results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/youtube/polls.py --generate
  python scripts/youtube/polls.py --generate --count 5 --json
  python scripts/youtube/polls.py --read VIDEO_ID --poll '{"question":"lights or darks?",...}'
  python scripts/youtube/polls.py --read VIDEO_ID --poll-file poll.json
        """
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--generate', action='store_true',
                      help='generate poll question(s)')
    mode.add_argument('--read',
                      help='read poll results from a video ID')
    mode.add_argument('--list-polls', action='store_true',
                      help='list all available poll questions')

    parser.add_argument('--count', type=int, default=1,
                        help='how many polls to generate (default: 1)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('--poll', type=str, default=None,
                        help='poll definition as JSON string (for --read)')
    parser.add_argument('--poll-file', type=str, default=None,
                        help='poll definition as JSON file (for --read)')
    parser.add_argument('--json', action='store_true',
                        help='output as JSON')
    parser.add_argument('--credentials', default=None,
                        help='directory with client_secret.json / token.json')

    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.list_polls:
        for i, poll in enumerate(POLLS):
            options = ' / '.join(opt['label'] for opt in poll['options'])
            print(f"  {poll['question']}")
            print(f"    config: {poll['config_key']} -> [{options}]")
            if poll.get('extra_overrides'):
                print(f"    also sets: {poll['extra_overrides']}")
            print()
        return

    if args.generate:
        results = [generate_poll_question(rng) for _ in range(args.count)]

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for poll in results:
                options = ' / '.join(opt['label'] for opt in poll['options'])
                print(f"  {poll['question']}")
                print(f"    -> {poll['config_key']}: [{options}]")
                print(f"    description text: {format_poll_for_description(poll).strip()}")
                print()
        return

    if args.read:
        # load poll definition
        if args.poll_file:
            with open(args.poll_file) as f:
                poll = json.load(f)
        elif args.poll:
            poll = json.loads(args.poll)
        else:
            print("error: --read requires --poll or --poll-file")
            sys.exit(1)

        results = read_poll_results(
            args.read, poll,
            credentials_dir=args.credentials,
        )

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\n  poll results for {args.read}")
            print(f"  question: {poll['question']}")
            print(f"  votes: {results['votes']}")
            print(f"  total: {results['total_votes']}")
            if results['winner']:
                print(f"  winner: {results['winner']}")
                print(f"  config override: {results['config_override']}")
            else:
                print(f"  no votes")
            print()


if __name__ == '__main__':
    main()
