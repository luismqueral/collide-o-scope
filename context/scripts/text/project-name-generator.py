"""
project-name-generator.py - Evocative Project Name Generator

Creates poetic, memorable project names for generative video art.
Uses linguistic patterns to generate names that feel intentional yet mysterious.

NAMING PATTERNS:
- "adjective-noun" (e.g., "velvet-cascade")
- "noun-preposition-noun" (e.g., "rivers-through-static")  
- "gerund-noun" (e.g., "dissolving-meridian")
- "noun-noun" compound (e.g., "ghost-frequency")
- "the-noun-of-noun" (e.g., "the-weight-of-light")

Usage:
    python scripts/text/project-name-generator.py            # Generate one name
    python scripts/text/project-name-generator.py --count 10 # Generate 10 names
    python scripts/text/project-name-generator.py --style poetic  # Use specific style
"""

import random
import argparse
import os

# =============================================================================
# WORD BANKS - Curated for aesthetic resonance
# =============================================================================

# Adjectives that evoke texture, light, and atmosphere
ADJECTIVES = [
    # textures
    "velvet", "liquid", "crystalline", "hollow", "fractured", "silken",
    "molten", "granular", "woven", "tangled", "scattered", "layered",
    # light/color qualities
    "luminous", "iridescent", "phosphorescent", "dim", "blazing", "pale",
    "saturated", "washed", "bleached", "glowing", "smoldering", "flickering",
    # atmospheric
    "distant", "submerged", "floating", "suspended", "drifting", "hovering",
    "buried", "emerging", "fading", "rising", "falling", "dissolving",
    # abstract qualities
    "quiet", "restless", "patient", "urgent", "slow", "sudden",
    "soft", "sharp", "blurred", "clear", "hazy", "vivid",
    # evocative
    "forgotten", "remembered", "imagined", "witnessed", "dreamed", "lost",
    "found", "hidden", "revealed", "broken", "mended", "unfinished"
]

# Nouns that suggest motion, nature, and abstraction
NOUNS = [
    # natural phenomena
    "cascade", "meridian", "eclipse", "aurora", "nebula", "tide",
    "current", "drift", "undertow", "storm", "haze", "vapor",
    # light/time
    "dusk", "twilight", "daybreak", "midnight", "noon", "shadow",
    "gleam", "flicker", "pulse", "glow", "flash", "shimmer",
    # water/fluid
    "river", "stream", "pool", "wave", "ripple", "eddy",
    "spill", "flow", "surge", "flood", "drip", "rain",
    # earth/space  
    "canyon", "ridge", "plateau", "crater", "orbit", "void",
    "hollow", "cave", "shore", "cliff", "peak", "valley",
    # abstract/technical
    "frequency", "static", "signal", "noise", "loop", "interval",
    "threshold", "boundary", "edge", "frame", "layer", "trace",
    # poetic
    "memory", "echo", "ghost", "remnant", "fragment", "shard",
    "thread", "weight", "distance", "silence", "warmth", "cold"
]

# Gerunds for action-based names
GERUNDS = [
    "dissolving", "drifting", "falling", "floating", "flickering", "fading",
    "emerging", "burning", "freezing", "melting", "breaking", "mending",
    "dreaming", "waking", "sleeping", "breathing", "pulsing", "humming",
    "spinning", "spiraling", "unwinding", "unfolding", "scattering", "gathering"
]

# Prepositions for relational names
PREPOSITIONS = [
    "through", "between", "beneath", "beyond", "within", "without",
    "against", "toward", "across", "among", "before", "after"
]

# Abstract concepts for philosophical names
CONCEPTS = [
    "time", "space", "light", "dark", "warmth", "cold",
    "weight", "stillness", "motion", "silence", "sound", "touch",
    "absence", "presence", "distance", "closeness", "loss", "return"
]

# =============================================================================
# NAME GENERATION PATTERNS
# =============================================================================

def pattern_adjective_noun():
    """velvet-cascade, liquid-meridian"""
    return f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"

def pattern_noun_noun():
    """ghost-frequency, shadow-river"""
    return f"{random.choice(NOUNS)}-{random.choice(NOUNS)}"

def pattern_gerund_noun():
    """dissolving-meridian, falling-static"""
    return f"{random.choice(GERUNDS)}-{random.choice(NOUNS)}"

def pattern_noun_preposition_noun():
    """rivers-through-static, light-beneath-water"""
    return f"{random.choice(NOUNS)}-{random.choice(PREPOSITIONS)}-{random.choice(NOUNS)}"

def pattern_the_noun_of_noun():
    """the-weight-of-light, the-edge-of-silence"""
    return f"the-{random.choice(NOUNS)}-of-{random.choice(CONCEPTS)}"

def pattern_adjective_noun_numeral():
    """liquid-cascade-003, fractured-signal-017"""
    num = random.randint(1, 99)
    return f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}-{num:03d}"

def pattern_minimal():
    """single evocative word with optional number"""
    word = random.choice(NOUNS + CONCEPTS)
    if random.random() > 0.5:
        return f"{word}-{random.randint(1, 99):02d}"
    return word

def pattern_double_adjective():
    """slow-quiet-echo, pale-distant-glow"""
    adj1, adj2 = random.sample(ADJECTIVES, 2)
    return f"{adj1}-{adj2}-{random.choice(NOUNS)}"

# All available patterns
PATTERNS = {
    'adjective-noun': pattern_adjective_noun,
    'noun-noun': pattern_noun_noun,
    'gerund-noun': pattern_gerund_noun,
    'noun-prep-noun': pattern_noun_preposition_noun,
    'the-of': pattern_the_noun_of_noun,
    'numbered': pattern_adjective_noun_numeral,
    'minimal': pattern_minimal,
    'double-adj': pattern_double_adjective,
}

# Style presets (weighted pattern selections)
STYLES = {
    'poetic': ['adjective-noun', 'gerund-noun', 'the-of', 'double-adj'],
    'minimal': ['minimal', 'noun-noun', 'adjective-noun'],
    'numbered': ['numbered', 'adjective-noun', 'minimal'],
    'descriptive': ['noun-prep-noun', 'the-of', 'double-adj', 'gerund-noun'],
    'random': list(PATTERNS.keys()),
}

# =============================================================================
# MAIN GENERATOR
# =============================================================================

def generate_project_name(style='random', pattern=None):
    """
    Generate a single project name.
    
    Args:
        style: Name style preset ('poetic', 'minimal', 'numbered', 'descriptive', 'random')
        pattern: Specific pattern to use (overrides style)
        
    Returns:
        A kebab-case project name string
    """
    if pattern and pattern in PATTERNS:
        generator = PATTERNS[pattern]
    else:
        available_patterns = STYLES.get(style, STYLES['random'])
        pattern_name = random.choice(available_patterns)
        generator = PATTERNS[pattern_name]
    
    return generator()


def generate_unique_names(count, style='random', existing_names=None):
    """
    Generate multiple unique project names.
    
    Args:
        count: Number of names to generate
        style: Name style preset
        existing_names: Set of names to avoid duplicating
        
    Returns:
        List of unique project names
    """
    existing = set(existing_names or [])
    names = []
    attempts = 0
    max_attempts = count * 10  # Prevent infinite loops
    
    while len(names) < count and attempts < max_attempts:
        name = generate_project_name(style)
        if name not in existing and name not in names:
            names.append(name)
        attempts += 1
    
    return names


def get_existing_project_names(projects_dir='projects'):
    """
    Get list of existing project folder names.
    
    Args:
        projects_dir: Path to projects directory
        
    Returns:
        Set of existing project names
    """
    if not os.path.exists(projects_dir):
        return set()
    
    return {
        name for name in os.listdir(projects_dir)
        if os.path.isdir(os.path.join(projects_dir, name))
        and not name.startswith('_')  # Skip template
        and name != 'archive'  # Skip archive
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate evocative project names for video art',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Styles:
  poetic      - Lyrical, abstract names (velvet-cascade, the-weight-of-light)
  minimal     - Simple, clean names (echo-02, ghost-frequency)
  numbered    - Names with numeric suffixes (liquid-meridian-017)
  descriptive - Longer relational names (rivers-through-static)
  random      - Mix of all patterns (default)

Patterns:
  adjective-noun    velvet-cascade
  noun-noun         ghost-frequency
  gerund-noun       dissolving-meridian
  noun-prep-noun    rivers-through-static
  the-of            the-weight-of-light
  numbered          liquid-cascade-003
  minimal           echo-02
  double-adj        slow-quiet-echo
        """
    )
    
    parser.add_argument('--count', '-n', type=int, default=1,
                        help='Number of names to generate (default: 1)')
    parser.add_argument('--style', '-s', choices=list(STYLES.keys()), default='random',
                        help='Name style preset (default: random)')
    parser.add_argument('--pattern', '-p', choices=list(PATTERNS.keys()),
                        help='Specific pattern to use (overrides style)')
    parser.add_argument('--unique', '-u', action='store_true',
                        help='Avoid names that already exist in projects/')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output as JSON array')
    
    args = parser.parse_args()
    
    existing = get_existing_project_names() if args.unique else None
    names = generate_unique_names(args.count, args.style, existing)
    
    if args.json:
        import json
        print(json.dumps(names))
    else:
        for name in names:
            print(name)


if __name__ == '__main__':
    main()

