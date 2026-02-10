"""
metadata.py - cryptic title and description generator for video art

generates titles and descriptions that feel like found transmissions,
recovered field notes, and half-remembered observations. designed to
sit alongside generative video art without over-explaining it.

usage as module:
    from scripts.text.metadata import generate_metadata
    meta = generate_metadata(session_info)
    # meta = { "title": "...", "description": "..." }

usage as CLI:
    python scripts/text/metadata.py
    python scripts/text/metadata.py --count 20
    python scripts/text/metadata.py --seed 42 --session '{"mode":"fixed","layers":3}'
"""

import random
import argparse
import json
import re
import os


# =============================================================================
# WORD BANKS
#
# curated for texture. these aren't meant to be pretty — they're meant
# to feel like they were found in a notebook left on a bus.
# =============================================================================

# concrete nouns — objects, materials, places
OBJECTS = [
    "salt", "glass", "wire", "tape", "stone", "dust", "ash", "ice",
    "milk", "rust", "sand", "chalk", "wax", "film", "bone", "wool",
    "clay", "tin", "lead", "silk", "fog", "moss", "soot", "resin",
    "chrome", "linen", "copper", "enamel", "lacquer", "graphite",
]

PLACES = [
    "room", "corridor", "lobby", "lot", "station", "field", "canal",
    "overpass", "basement", "attic", "hangar", "dock", "terminal",
    "stairwell", "yard", "warehouse", "plaza", "vestibule", "annex",
    "landing", "culvert", "threshold", "shoulder", "median", "apron",
]

# states — conditions, qualities
STATES = [
    "still", "slow", "blank", "quiet", "dim", "flat", "raw", "soft",
    "pale", "dull", "warm", "cool", "dense", "thin", "dry", "wet",
    "bright", "dark", "sharp", "loose", "tight", "near", "far", "low",
    "open", "closed", "empty", "full", "clean", "worn", "faint", "clear",
]

# actions — what's happening (or happened)
ACTIONS = [
    "recovered", "erased", "copied", "looped", "stacked", "removed",
    "replaced", "shifted", "rotated", "repeated", "inverted", "folded",
    "masked", "exposed", "flattened", "stretched", "compressed", "split",
    "merged", "dissolved", "extracted", "discarded", "translated", "mapped",
]

# time/sequence words
TEMPORAL = [
    "before", "after", "between", "during", "until", "since",
    "once", "again", "still", "already", "never", "always",
    "early", "late", "soon", "now", "then", "first", "last",
]

# technical fragments — camera/video language
TECHNICAL = [
    "signal", "frame", "scan", "pass", "channel", "track",
    "exposure", "gain", "offset", "sync", "dropout", "artifact",
    "sample", "capture", "feed", "source", "output", "render",
    "sequence", "clip", "take", "reel", "head", "tail",
]

# numbers and codes that feel institutional
def _code(rng):
    """generate a code that looks like it came from a filing system."""
    patterns = [
        lambda: f"{rng.randint(1, 99):02d}",
        lambda: f"{rng.randint(1, 999):03d}",
        lambda: f"{rng.choice('ABCDEFGHJKLMNPRSTUVWX')}{rng.randint(1, 99):02d}",
        lambda: f"{rng.randint(1, 12):02d}.{rng.randint(1, 28):02d}",
        lambda: f"no. {rng.randint(1, 300)}",
        lambda: f"#{rng.randint(1, 9999):04d}",
    ]
    return rng.choice(patterns)()


# =============================================================================
# TITLE PATTERNS
#
# each function returns a title string.
# the goal: something you'd see on a gallery placard or a VHS label
# found in a thrift store. not clever — just present.
# =============================================================================

def _title_object_state(rng):
    """salt, still / glass (dim)"""
    obj = rng.choice(OBJECTS)
    state = rng.choice(STATES)
    formats = [
        f"{obj}, {state}",
        f"{obj} ({state})",
        f"{state} {obj}",
    ]
    return rng.choice(formats)

def _title_place_code(rng):
    """room 04 / corridor B12 / lot 003"""
    place = rng.choice(PLACES)
    code = _code(rng)
    formats = [
        f"{place} {code}",
        f"{place}, {code}",
        f"{place} — {code}",
    ]
    return rng.choice(formats)

def _title_action_object(rng):
    """recovered glass / erased signal"""
    return f"{rng.choice(ACTIONS)} {rng.choice(OBJECTS + TECHNICAL)}"

def _title_study(rng):
    """study for three cameras / notes on white removal"""
    subjects = OBJECTS + TECHNICAL + PLACES
    n = rng.choice(["two", "three", "four", "five", "several"])
    formats = [
        f"study for {n} {rng.choice(TECHNICAL)}s",
        f"notes on {rng.choice(OBJECTS)}",
        f"notes on {rng.choice(ACTIONS).rstrip('ed')}ing",
        f"test {_code(rng)}: {rng.choice(OBJECTS)}",
        f"attempt {rng.randint(1, 40)} ({rng.choice(OBJECTS)})",
    ]
    return rng.choice(formats)

def _title_fragment(rng):
    """what the camera left / after the white was removed"""
    fragments = [
        f"what the {rng.choice(TECHNICAL)} left",
        f"after the {rng.choice(OBJECTS)} was {rng.choice(ACTIONS)}",
        f"before the {rng.choice(TECHNICAL)} {rng.choice(['stopped', 'started', 'cut out'])}",
        f"everything {rng.choice(ACTIONS)} from the {rng.choice(TECHNICAL)}",
        f"the {rng.choice(OBJECTS)} in the {rng.choice(PLACES)}",
        f"the {rng.choice(PLACES)} with no {rng.choice(OBJECTS)}",
        f"nothing {rng.choice(ACTIONS)}, nothing {rng.choice(ACTIONS)}",
        f"only the {rng.choice(OBJECTS)} remained",
    ]
    return rng.choice(fragments)

def _title_minimal(rng):
    """untitled / untitled (03) / — """
    formats = [
        "untitled",
        f"untitled ({_code(rng)})",
        f"untitled, {rng.choice(OBJECTS)}",
        f"— {rng.choice(OBJECTS)} —",
        f"[{rng.choice(TECHNICAL)}]",
        f"({rng.choice(STATES)})",
    ]
    return rng.choice(formats)

def _title_single_word(rng):
    """dust / still / corridor"""
    pool = OBJECTS + STATES + PLACES + TECHNICAL
    return rng.choice(pool)

def _title_compound(rng):
    """salt cathedral / glass arithmetic / dust frequency"""
    return f"{rng.choice(OBJECTS)} {rng.choice(TECHNICAL + PLACES)}"

def _title_sequence(rng):
    """i. white / iii. recovered / part 2: the corridor"""
    numerals = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
    n = rng.choice(numerals)
    word = rng.choice(OBJECTS + STATES + ACTIONS)
    formats = [
        f"{n}. {word}",
        f"part {rng.randint(1, 12)}: {rng.choice(OBJECTS)}",
        f"{rng.choice(OBJECTS)} / {rng.choice(OBJECTS)}",
        f"{rng.choice(STATES)} / {rng.choice(STATES)} / {rng.choice(STATES)}",
    ]
    return rng.choice(formats)

def _title_long_fragment(rng):
    """longer, sentence-like titles from the chain or built by hand"""
    formats = [
        f"the {rng.choice(OBJECTS)} was {rng.choice(ACTIONS)} and the {rng.choice(TECHNICAL)} kept {rng.choice(['running', 'playing', 'recording', 'looping'])}",
        f"what happens when the {rng.choice(OBJECTS)} is {rng.choice(ACTIONS)} from the {rng.choice(TECHNICAL)}",
        f"everything in the {rng.choice(PLACES)} was {rng.choice(STATES)} except the {rng.choice(OBJECTS)}",
        f"the {rng.choice(PLACES)} where the {rng.choice(TECHNICAL)} was left {rng.choice(['running', 'on', 'playing', 'open'])}",
        f"{rng.choice(ACTIONS)} the {rng.choice(OBJECTS)}, {rng.choice(ACTIONS)} the {rng.choice(OBJECTS)}, {rng.choice(ACTIONS)} the {rng.choice(TECHNICAL)}",
        f"a {rng.choice(STATES)} {rng.choice(PLACES)} with {rng.choice(OBJECTS)} on the {rng.choice(['floor', 'wall', 'table', 'screen', 'window'])}",
        f"the {rng.choice(OBJECTS)} in the {rng.choice(PLACES)} and the {rng.choice(OBJECTS)} in the {rng.choice(PLACES)}",
    ]
    return rng.choice(formats)

def _title_chain(rng):
    """a markov chain as a title — variable length"""
    length = rng.choice([3, 4, 5, 6, 7, 8])
    return generate_chain(rng=rng, length=length)

def _title_just_code(rng):
    """03 / B12 / #0041 / no. 7"""
    return _code(rng)


# all title generators, weighted toward the ones that feel best
TITLE_GENERATORS = [
    (_title_object_state, 3),
    (_title_place_code, 2),
    (_title_action_object, 3),
    (_title_study, 2),
    (_title_fragment, 3),
    (_title_minimal, 1),
    (_title_single_word, 2),
    (_title_compound, 3),
    (_title_sequence, 2),
    (_title_long_fragment, 2),
    (_title_chain, 2),
    (_title_just_code, 1),
]

def _weighted_choice(rng, weighted_list):
    """pick from a list of (item, weight) tuples."""
    items, weights = zip(*weighted_list)
    total = sum(weights)
    r = rng.uniform(0, total)
    cumulative = 0
    for item, weight in zip(items, weights):
        cumulative += weight
        if r <= cumulative:
            return item
    return items[-1]


# =============================================================================
# DESCRIPTION PATTERNS
#
# descriptions are built from fragments — short lines that can be
# combined. some are technical, some are observational, some are neither.
# =============================================================================

def _desc_observational(rng):
    """lines that sound like someone watching the video and taking notes."""
    observations = [
        f"the {rng.choice(OBJECTS)} is {rng.choice(STATES)} throughout",
        f"no visible {rng.choice(OBJECTS)} in the final {rng.choice(TECHNICAL)}",
        f"the {rng.choice(TECHNICAL)} degrades after the first minute",
        f"{rng.choice(STATES)} in the corners, {rng.choice(STATES)} in the center",
        f"audio from {rng.choice(['a parking lot', 'an auction', 'a kitchen', 'a hallway', 'somewhere indoors', 'somewhere outside', 'an unknown room', 'a moving vehicle'])}",
        f"the source material was {rng.choice(ACTIONS)} before compositing",
        f"found footage, {rng.choice(ACTIONS)}",
        f"all cameras were {rng.choice(STATES)}",
        f"color correction {rng.choice(['applied', 'minimal', 'aggressive', 'subtle', 'absent'])}",
        f"the layers {rng.choice(['compete', 'cooperate', 'ignore each other', 'bleed together'])}",
        f"nothing was {rng.choice(['planned', 'rehearsed', 'repeated', 'staged', 'intentional'])}",
        f"the audio was {rng.choice(['left alone', 'panned wide', 'barely there', 'louder than expected'])}",
    ]
    return rng.choice(observations)


def _desc_cryptic(rng):
    """one-liners that don't explain anything."""
    lines = [
        f"recorded {rng.choice(TEMPORAL)} it {rng.choice(['happened', 'stopped', 'started', 'changed'])}",
        f"this is the {rng.choice(['only', 'last', 'first', 'second'])} version",
        f"there were others but they were {rng.choice(ACTIONS)}",
        f"the original is {rng.choice(['lost', 'somewhere else', 'longer', 'silent', 'unfinished'])}",
        f"not sure what the {rng.choice(TECHNICAL)} was pointed at",
        f"left running {rng.choice(TEMPORAL)} someone came back",
        f"the {rng.choice(OBJECTS)} was {rng.choice(TEMPORAL)} there",
        f"this happened {rng.choice(TEMPORAL)}",
        f"repeat until {rng.choice(STATES)}",
        f"see also: {_title_compound(rng)}",
    ]
    return rng.choice(lines)


# =============================================================================
# PREDICTIVE CHAIN (markov)
#
# hand-curated word-to-word transitions. like mashing the middle
# suggestion on your phone keyboard, except the dictionary was written
# by someone who watches too much experimental film.
#
# each word maps to a list of possible next words. the chain walks
# forward from a starter word, picking the next word each step.
# the result is almost-coherent, almost-meaningful text.
# =============================================================================

# transition table: word → [possible next words]
# words can appear in multiple lists. shorter lists = more predictable chains.
# longer lists = more chaos. overlap between entries creates loops and echoes.
CHAIN = {
    # starters / anchors
    "the":      ["camera", "signal", "room", "white", "tape", "glass", "dust",
                 "sound", "light", "floor", "wall", "water", "screen", "film",
                 "footage", "end", "beginning", "machine", "silence", "color"],
    "a":        ["room", "signal", "camera", "long", "short", "quiet", "slow",
                 "single", "distant", "kind", "version", "copy", "recording"],
    "this":     ["is", "was", "isn't", "happens", "continues", "repeats",
                 "version", "recording", "room", "time"],
    "it":       ["was", "is", "wasn't", "happens", "continues", "loops",
                 "plays", "stops", "starts", "fades", "runs", "repeats"],

    # verbs / actions
    "was":      ["left", "found", "recorded", "already", "never", "always",
                 "still", "there", "here", "removed", "playing", "running",
                 "quiet", "loud", "empty", "over", "white", "dark"],
    "is":       ["still", "not", "always", "never", "what", "playing",
                 "running", "somewhere", "gone", "here", "there", "quiet",
                 "the", "a", "empty", "over", "almost"],
    "wasn't":   ["there", "supposed", "recorded", "meant", "real",
                 "finished", "the", "a", "what", "playing"],
    "happens":  ["when", "again", "slowly", "after", "before", "between",
                 "every", "in", "without", "the"],
    "continues":["without", "after", "until", "past", "the", "in",
                 "regardless", "somewhere", "playing", "anyway"],
    "repeats":  ["itself", "until", "without", "the", "every", "after",
                 "endlessly", "again", "slowly", "from"],
    "loops":    ["back", "around", "endlessly", "without", "the", "until",
                 "through", "over", "again", "itself"],
    "plays":    ["back", "itself", "in", "the", "without", "again",
                 "on", "through", "slowly", "quietly"],
    "stops":    ["and", "when", "before", "after", "the", "itself",
                 "without", "abruptly", "quietly", "then"],
    "starts":   ["again", "over", "the", "with", "from", "without",
                 "playing", "slowly", "somewhere", "before"],
    "fades":    ["in", "out", "to", "slowly", "into", "the", "without",
                 "before", "between", "away"],
    "runs":     ["out", "through", "the", "without", "on", "until",
                 "backward", "slowly", "continuously", "again"],
    "left":     ["running", "playing", "on", "in", "the", "a", "behind",
                 "open", "alone", "blank", "empty", "here", "overnight"],
    "found":    ["in", "the", "a", "on", "between", "footage", "tape",
                 "recording", "nothing", "somewhere", "later"],
    "recorded": ["over", "the", "in", "a", "from", "before", "after",
                 "without", "by", "something", "nothing", "everything"],
    "removed":  ["the", "from", "everything", "carefully", "without",
                 "by", "and", "what", "color", "white", "most"],
    "playing":  ["in", "the", "back", "on", "without", "somewhere",
                 "still", "again", "itself", "quietly", "through"],

    # adjectives / states
    "still":    ["playing", "running", "there", "here", "white", "quiet",
                 "the", "a", "nothing", "warm", "waiting"],
    "quiet":    ["room", "signal", "the", "and", "in", "tape", "now",
                 "recording", "film", "enough", "static"],
    "slow":     ["tape", "film", "dissolve", "camera", "movement",
                 "version", "the", "and", "white", "pan"],
    "empty":    ["room", "tape", "signal", "screen", "frame", "the",
                 "and", "channel", "corridor", "now", "again"],
    "white":    ["room", "noise", "the", "and", "was", "is", "wall",
                 "screen", "light", "balance", "out", "on"],
    "dark":     ["room", "the", "and", "corner", "was", "corridor",
                 "enough", "water", "screen", "tape"],
    "long":     ["corridor", "tape", "exposure", "pause", "the",
                 "enough", "time", "recording", "signal", "version"],
    "short":    ["loop", "signal", "version", "the", "tape", "burst",
                 "recording", "circuit", "pause", "film"],
    "distant":  ["camera", "signal", "room", "recording", "the",
                 "sound", "white", "hum", "figure", "static"],
    "single":   ["frame", "channel", "camera", "source", "take",
                 "the", "pass", "exposure", "recording", "note"],

    # nouns
    "camera":   ["was", "is", "left", "found", "in", "the", "stopped",
                 "recorded", "running", "pointed", "facing"],
    "signal":   ["was", "is", "lost", "found", "from", "the", "fades",
                 "drops", "returns", "in", "repeats", "cuts"],
    "room":     ["was", "is", "the", "with", "where", "empty",
                 "quiet", "dark", "white", "without", "full"],
    "tape":     ["was", "is", "runs", "loops", "plays", "the", "found",
                 "left", "from", "in", "over", "ends"],
    "glass":    ["was", "is", "the", "on", "in", "and", "over",
                 "broke", "reflects", "dust", "table"],
    "dust":     ["on", "the", "in", "was", "is", "and", "settles",
                 "covers", "everywhere", "from", "glass"],
    "sound":    ["was", "is", "the", "of", "from", "in", "plays",
                 "fades", "loops", "fills", "comes", "and"],
    "light":    ["was", "is", "the", "in", "from", "on", "fades",
                 "comes", "through", "and", "changes", "fills"],
    "floor":    ["was", "is", "the", "and", "in", "dust", "white",
                 "wet", "cold", "empty", "concrete"],
    "wall":     ["was", "is", "the", "and", "white", "blank", "behind",
                 "facing", "opposite", "bare", "wet"],
    "water":    ["was", "is", "the", "on", "in", "from", "runs",
                 "still", "and", "over", "sound", "dark"],
    "screen":   ["was", "is", "the", "goes", "white", "blank", "dark",
                 "shows", "plays", "left", "flickers", "and"],
    "film":     ["was", "is", "the", "runs", "plays", "loops", "found",
                 "left", "from", "grain", "stock", "and"],
    "footage":  ["was", "is", "the", "from", "found", "of", "plays",
                 "loops", "recorded", "left", "recovered", "lost"],
    "color":    ["was", "is", "the", "removed", "shifted", "fades",
                 "from", "and", "changes", "gone", "returns"],
    "silence":  ["was", "is", "the", "in", "and", "between", "after",
                 "before", "fills", "returns", "remains"],
    "machine":  ["was", "is", "the", "runs", "left", "hums", "stops",
                 "continues", "in", "and", "still", "quiet"],
    "recording":["was", "is", "the", "of", "from", "found", "plays",
                 "loops", "left", "ends", "continues", "stops"],
    "end":      ["of", "the", "was", "is", "comes", "and"],
    "beginning":["of", "the", "was", "is", "and", "again"],
    "frame":    ["was", "is", "the", "by", "from", "rate", "left",
                 "missing", "blank", "white", "dark", "and"],
    "channel":  ["was", "is", "the", "left", "open", "empty", "plays",
                 "static", "noise", "and", "from"],
    "static":   ["in", "the", "and", "was", "is", "plays", "between",
                 "from", "fills", "on", "noise"],
    "noise":    ["in", "the", "and", "was", "is", "from", "fills",
                 "floor", "white", "plays", "fades"],
    "figure":   ["in", "the", "was", "is", "moves", "stands",
                 "disappears", "appears", "and", "behind"],
    "hum":      ["of", "the", "in", "and", "was", "is", "from",
                 "fills", "continues", "fades", "low"],
    "version":  ["of", "the", "was", "is", "plays", "that", "found",
                 "left", "this", "another", "last"],

    # prepositions / connectors (keep chains flowing)
    "in":       ["the", "a", "this", "here", "silence", "between",
                 "there", "another", "every", "some"],
    "of":       ["the", "a", "this", "it", "something", "nothing",
                 "what", "silence", "light", "dust", "white"],
    "from":     ["the", "a", "here", "there", "somewhere", "nowhere",
                 "inside", "outside", "before", "behind", "above"],
    "on":       ["the", "a", "loop", "repeat", "tape", "screen",
                 "glass", "film", "its", "itself"],
    "with":     ["the", "a", "no", "nothing", "dust", "static",
                 "sound", "light", "everything", "white"],
    "without":  ["the", "a", "sound", "light", "color", "warning",
                 "end", "beginning", "reason", "stopping", "knowing"],
    "to":       ["the", "a", "nothing", "white", "black", "dust",
                 "silence", "static", "itself", "somewhere"],
    "and":      ["the", "a", "it", "then", "nothing", "everything",
                 "again", "still", "quiet", "left", "was", "is"],
    "between":  ["the", "frames", "channels", "takes", "rooms",
                 "signals", "recordings", "passes", "walls", "static"],
    "through":  ["the", "a", "glass", "static", "dust", "walls",
                 "water", "film", "noise", "silence", "everything"],
    "into":     ["the", "a", "nothing", "white", "static", "dust",
                 "silence", "itself", "somewhere", "noise"],
    "over":     ["the", "a", "it", "and", "again", "itself", "time",
                 "everything", "nothing", "white", "static"],
    "after":    ["the", "a", "it", "everything", "that", "this",
                 "nothing", "silence", "awhile", "dark"],
    "before":   ["the", "a", "it", "everything", "that", "this",
                 "nothing", "anyone", "recording", "the"],
    "until":    ["the", "it", "nothing", "everything", "someone",
                 "silence", "static", "white", "morning", "dark"],
    "every":    ["time", "frame", "channel", "recording", "version",
                 "room", "camera", "signal", "take", "pass"],

    # adverbs / modifiers
    "always":   ["the", "playing", "running", "there", "here", "was",
                 "quiet", "white", "on", "in", "somewhere"],
    "never":    ["the", "finished", "started", "recorded", "played",
                 "quiet", "still", "was", "here", "there"],
    "somewhere":["in", "the", "else", "between", "behind", "a",
                 "near", "quiet", "dark", "below"],
    "nowhere":  ["in", "the", "near", "particular", "visible",
                 "left", "and", "else", "close"],
    "again":    ["the", "and", "it", "in", "from", "without",
                 "until", "slowly", "quietly", "always"],
    "slowly":   ["the", "it", "fades", "plays", "runs", "fills",
                 "dissolves", "and", "in", "into", "through"],
    "quietly":  ["the", "it", "plays", "runs", "fades", "in",
                 "and", "hums", "fills", "loops"],
    "almost":   ["the", "there", "quiet", "still", "nothing",
                 "empty", "gone", "finished", "white", "over"],
    "here":     ["the", "and", "is", "was", "in", "it", "where",
                 "still", "again", "now", "before"],
    "there":    ["was", "is", "the", "and", "in", "it", "where",
                 "still", "again", "nothing", "something"],
    "then":     ["the", "it", "nothing", "everything", "silence",
                 "again", "quiet", "gone", "still", "white"],
    "gone":     ["the", "and", "now", "before", "is", "was",
                 "from", "quiet", "already", "again"],
    "already":  ["the", "gone", "playing", "running", "over",
                 "there", "here", "quiet", "was", "removed"],
    "anyway":   ["the", "it", "and", "this", "nothing", "plays",
                 "continues", "runs", "is", "was"],
    "endlessly":["the", "it", "and", "in", "plays", "loops",
                 "runs", "through", "into", "without"],
    "back":     ["the", "to", "and", "in", "again", "into",
                 "from", "through", "it", "around"],
    "away":     ["the", "from", "and", "in", "slowly", "into",
                 "quietly", "it", "now", "again"],
    "out":      ["the", "of", "and", "in", "again", "it",
                 "here", "there", "now", "slowly"],

    # misc / dead ends that restart
    "nothing":  ["was", "is", "plays", "left", "remains", "happens",
                 "in", "the", "but", "and", "recorded", "changed"],
    "everything":["was", "is", "plays", "fades", "loops", "left",
                  "in", "the", "and", "removed", "recorded", "else"],
    "something":["was", "is", "plays", "in", "the", "and", "left",
                 "there", "like", "about", "from", "recorded"],
    "someone":  ["was", "is", "left", "in", "the", "and", "came",
                 "there", "recorded", "removed", "played"],
}


def generate_chain(starter=None, length=None, rng=None):
    """
    walk the transition table from a starter word, picking the next
    word each step. like mashing the middle predictive text suggestion
    on your phone.

    Args:
        starter: first word (or None for random)
        length: number of words (or None for random 5-15)
        rng: random.Random instance (or None for unseeded)

    Returns:
        string of chained words
    """
    if rng is None:
        rng = random.Random()

    if length is None:
        length = rng.randint(5, 15)

    # pick a starter
    if starter is None or starter not in CHAIN:
        starter = rng.choice(list(CHAIN.keys()))

    words = [starter]
    current = starter

    for _ in range(length - 1):
        if current in CHAIN:
            current = rng.choice(CHAIN[current])
            words.append(current)
        else:
            # dead end — pick a random anchor and keep going
            current = rng.choice(["the", "a", "it", "and", "in"])
            words.append(current)

    return ' '.join(words)


# =============================================================================
# SEPARATORS
#
# how fragments join together.
# =============================================================================

SEPARATORS = [
    " / ",
    ". ",
    " — ",
    "\n",
    " · ",
    ", ",
    "  //  ",
]


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def generate_title(rng=None, session=None):
    """
    generate a single cryptic title.

    Args:
        rng: random.Random instance (or None for unseeded)
        session: dict of session/render info (optional, unused for titles currently)

    Returns:
        title string, lowercase
    """
    if rng is None:
        rng = random.Random()

    generator = _weighted_choice(rng, TITLE_GENERATORS)
    return generator(rng).lower()


def generate_description(rng=None, session=None):
    """
    generate a cryptic description from fragments.

    purely observational and cryptic — no technical details, no session
    data, no fps, no duration, no mode info. just vibes.

    Args:
        rng: random.Random instance (or None for unseeded)
        session: ignored (kept for API compatibility)

    Returns:
        description string
    """
    if rng is None:
        rng = random.Random()

    # pick a structure — the key is *variable length*.
    # some descriptions are two words. some are a paragraph.
    roll = rng.random()

    if roll < 0.1:
        # ultra-short: just a state or an object
        return rng.choice(STATES + OBJECTS).lower()

    elif roll < 0.2:
        # very short: 2-4 word fragment
        short = [
            f"{rng.choice(STATES)} {rng.choice(OBJECTS)}",
            f"{rng.choice(ACTIONS)} and {rng.choice(ACTIONS)}",
            f"{rng.choice(OBJECTS)}, {rng.choice(STATES)}",
            f"not {rng.choice(STATES)}",
            f"almost {rng.choice(STATES)}",
            f"still {rng.choice(STATES)}",
            f"the {rng.choice(OBJECTS)}",
            f"no {rng.choice(OBJECTS)}",
            f"{rng.choice(TEMPORAL)} the {rng.choice(OBJECTS)}",
        ]
        return rng.choice(short).lower()

    elif roll < 0.35:
        # single cryptic or observational line
        if rng.random() < 0.5:
            return _desc_cryptic(rng).lower()
        return _desc_observational(rng).lower()

    elif roll < 0.5:
        # short chain (3-7 words)
        return generate_chain(length=rng.randint(3, 7), rng=rng).lower()

    elif roll < 0.65:
        # medium chain (8-15 words)
        return generate_chain(rng=rng).lower()

    elif roll < 0.78:
        # observational + cryptic combo
        fragments = [_desc_observational(rng)]
        if rng.random() < 0.6:
            fragments.append(_desc_cryptic(rng))
        sep = rng.choice(SEPARATORS)
        return sep.join(fragments).lower()

    elif roll < 0.88:
        # multi-fragment: 2-4 cryptic/observational lines
        count = rng.randint(2, 4)
        fragments = []
        for _ in range(count):
            if rng.random() < 0.5:
                fragments.append(_desc_cryptic(rng))
            else:
                fragments.append(_desc_observational(rng))
        sep = rng.choice(SEPARATORS)
        return sep.join(fragments).lower()

    elif roll < 0.95:
        # long: chain + cryptic aside(s)
        chain = generate_chain(length=rng.randint(10, 20), rng=rng)
        extras = [_desc_cryptic(rng) for _ in range(rng.randint(1, 2))]
        sep = rng.choice(SEPARATORS)
        return sep.join([chain] + extras).lower()

    else:
        # sprawl: multiple chains and fragments woven together
        parts = []
        for _ in range(rng.randint(2, 4)):
            if rng.random() < 0.5:
                parts.append(generate_chain(length=rng.randint(5, 12), rng=rng))
            elif rng.random() < 0.5:
                parts.append(_desc_cryptic(rng))
            else:
                parts.append(_desc_observational(rng))
        sep = rng.choice(SEPARATORS)
        return sep.join(parts).lower()


def generate_metadata(session=None, rng=None, seed=None):
    """
    generate a complete metadata dict (title + description).

    this is the main entry point for other scripts.

    Args:
        session: dict of render session info
        rng: random.Random instance (or None to create one)
        seed: seed for the rng (ignored if rng is provided)

    Returns:
        dict with 'title' and 'description' keys
    """
    if rng is None:
        rng = random.Random(seed)

    return {
        'title': generate_title(rng, session),
        'description': generate_description(rng, session),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='generate cryptic titles and descriptions for video art',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/text/metadata.py
  python scripts/text/metadata.py --count 20
  python scripts/text/metadata.py --seed 42
  python scripts/text/metadata.py --titles-only --count 50
  python scripts/text/metadata.py --session '{"mode":"fixed","layers":3,"duration":187,"seed":42}'
  python scripts/text/metadata.py --chain --count 10
  python scripts/text/metadata.py --chain --starter "the" --length 12
        """
    )

    parser.add_argument('--count', '-n', type=int, default=1,
                        help='how many to generate (default: 1)')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for reproducibility')
    parser.add_argument('--session', type=str, default=None,
                        help='JSON string of session info (mode, layers, duration, etc)')
    parser.add_argument('--titles-only', action='store_true',
                        help='only output titles')
    parser.add_argument('--chain', action='store_true',
                        help='generate predictive text chains instead')
    parser.add_argument('--starter', type=str, default=None,
                        help='starter word for chain mode')
    parser.add_argument('--length', type=int, default=None,
                        help='chain length in words (default: random 5-15)')
    parser.add_argument('--json', action='store_true',
                        help='output as JSON')

    args = parser.parse_args()

    session = json.loads(args.session) if args.session else {}
    rng = random.Random(args.seed)

    results = []
    for _ in range(args.count):
        if args.chain:
            results.append(generate_chain(
                starter=args.starter, length=args.length, rng=rng
            ))
        elif args.titles_only:
            results.append(generate_title(rng, session))
        else:
            results.append(generate_metadata(session, rng))

    if args.json:
        print(json.dumps(results, indent=2))
    elif args.chain or args.titles_only:
        for r in results:
            print(r)
    else:
        for i, r in enumerate(results):
            if i > 0:
                print()
            print(f"  {r['title']}")
            print(f"  {r['description']}")


if __name__ == '__main__':
    main()
