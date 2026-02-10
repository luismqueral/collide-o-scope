# Color Keying Approaches for Video Blender

This document outlines different strategies for selecting which colors become transparent in the video overlay system.

---

## Current Approach: Fixed Color

**How it works:**
- Single hardcoded color (currently white `0xFFFFFF`)
- Same color keyed out for all videos, all runs

**Pros:** Simple, predictable, fast
**Cons:** Repetitive results, doesn't adapt to content

```python
COLORKEY_HEX = '0xFFFFFF'  # Always white
```

---

## Approach 1: K-Means Clustering (ML-Based)

**How it works:**
1. Extract a random frame from each video
2. Use K-Means to find N dominant colors in that frame
3. Randomly select M of those colors to key out
4. Each video gets keyed based on ITS OWN colors

**Implementation:** Already exists in `blend-video-alt-color-key.py`

**Pros:**
- Adapts to actual video content
- Creates organic, unpredictable results
- Colors that exist in footage become transparent
- Debug images show what was detected

**Cons:**
- Requires numpy, sklearn, PIL dependencies
- Slower (frame extraction + ML processing)
- Can sometimes key out important parts of video

**Config options:**
```python
COLORS_PER_PALETTE = 4    # Colors to extract per video
COLORS_TO_KEY_OUT = 2     # How many to make transparent
COLOR_SIMILARITY = 0.3    # Tolerance for near-matches
```

---

## Approach 2: Random Preset Palette

**How it works:**
- Define a list of interesting colors to key
- Randomly pick one (or more) per video or per run
- Good middle ground between fixed and ML

**Pros:**
- Fast (no analysis needed)
- Curated colors = predictable aesthetic
- Easy to tune for specific looks

**Cons:**
- Still somewhat random/uncontrolled
- May not match video content well

**Suggested palettes:**
```python
# Classic - high contrast
PALETTE_CLASSIC = ['0xFFFFFF', '0x000000']

# Neutrals - grays, beiges
PALETTE_NEUTRALS = ['0xFFFFFF', '0x000000', '0x808080', '0xC0C0C0', '0xF5F5DC']

# Warm - sunset/fire tones to key out
PALETTE_WARM = ['0xFFFFFF', '0xFF6B35', '0xFFA500', '0xFFD700']

# Cool - blues/greens to key out  
PALETTE_COOL = ['0x000000', '0x0066CC', '0x00CED1', '0x228B22']

# Skin tones - for surreal people videos
PALETTE_SKIN = ['0xFFDFC4', '0xF0C8A0', '0xDEB887', '0xD2956B']

# RGB primaries
PALETTE_RGB = ['0xFF0000', '0x00FF00', '0x0000FF']

# Pastels
PALETTE_PASTEL = ['0xFFB6C1', '0xADD8E6', '0x98FB98', '0xFFFFE0']
```

**Config:**
```python
COLOR_MODE = 'palette'
COLOR_PALETTE = PALETTE_CLASSIC
COLORS_PER_VIDEO = 1  # How many colors to key per overlay layer
RANDOMIZE_PER_VIDEO = True  # Different color per layer, or same for all?
```

---

## Approach 3: Luminance-Based Keying

**How it works:**
- Analyze video brightness
- Key out either DARKS or LIGHTS based on mode
- Can auto-detect: key out whichever is more prevalent

**Pros:**
- Works well for high-contrast footage
- Predictable results based on brightness
- No color bias

**Cons:**
- Less creative/artistic control
- May not work well for mid-tone videos

**Modes:**
```python
LUMINANCE_MODE = 'lights'  # Key out bright areas
LUMINANCE_MODE = 'darks'   # Key out dark areas
LUMINANCE_MODE = 'auto'    # Analyze and pick
LUMINANCE_THRESHOLD = 0.7  # 0-1, what counts as "light" or "dark"
```

**FFMPEG approach:**
```bash
# Key out lights (near white)
colorkey=0xFFFFFF:0.4:0.1

# Key out darks (near black)  
colorkey=0x000000:0.4:0.1

# Or use lumakey filter directly
lumakey=threshold=0.7:tolerance=0.1
```

---

## Approach 4: Multi-Color Keying

**How it works:**
- Apply MULTIPLE colorkey filters in sequence
- Each filter removes a different color
- More holes = more abstract layered look

**Pros:**
- Very abstract/artistic results
- More transparency = more layer blending
- Can combine approaches (e.g., white + one random color)

**Cons:**
- Can make videos too transparent
- Harder to control final look

**FFMPEG filter chain:**
```bash
colorkey=0xFFFFFF:0.3:0.1,colorkey=0x000000:0.2:0.1,colorkey=0x808080:0.15:0.1
```

**Config:**
```python
MULTI_KEY_COLORS = ['0xFFFFFF', '0x000000']  # Always key these
MULTI_KEY_RANDOM = 1  # Plus N random colors from palette
```

---

## Approach 5: Hybrid / Smart Mode

**How it works:**
Combine multiple approaches with fallbacks:

1. Try K-Means to find dominant color
2. If that color is "boring" (gray/neutral), use palette instead
3. Always include white OR black as baseline
4. Add one video-specific color

**Pros:**
- Best of all worlds
- Adapts but with guardrails
- Consistent baseline + variety

**Config:**
```python
COLOR_MODE = 'hybrid'
ALWAYS_KEY = ['0xFFFFFF']           # Always key white
USE_KMEANS = True                    # Try to find video-specific color
KMEANS_FALLBACK = PALETTE_CLASSIC    # If K-Means fails or is boring
BORING_THRESHOLD = 30                # Color variance below this = boring
```

---

## Approach 6: Content-Aware Presets

**How it works:**
- Analyze video filename or metadata for hints
- Apply appropriate palette based on content type

**Examples:**
```python
CONTENT_PRESETS = {
    'outdoor': PALETTE_COOL,      # Sky/nature - key blues/greens
    'indoor': PALETTE_WARM,       # Interior - key warm tones
    'night': ['0x000000'],        # Dark footage - key blacks
    'bright': ['0xFFFFFF'],       # Bright footage - key whites
    'default': PALETTE_CLASSIC
}
```

---

## Approach 7: ML Background Removal (rembg)

**How it works:**
1. Extract a random frame from the video
2. Run through rembg (U²-Net ML model) to detect foreground/background
3. Use the mask to isolate either background or foreground pixels
4. Run K-Means on those pixels to find dominant colors
5. Key out those colors from the video

**Pros:**
- Semantically aware (knows what's "background" vs "subject")
- Creates interesting subject isolation effects
- Can target foreground OR background
- Combines ML detection with color keying

**Cons:**
- Slower (ML model inference per video)
- Requires additional dependency: `pip install rembg onnxruntime`
- First run downloads ~170MB model
- Works best on videos with clear subjects

**Config options:**
```python
COLOR_MODE = 'rembg'
REMBG_TARGET = 'background'   # 'background', 'foreground', or 'random'
REMBG_COLORS_RANGE = (2, 4)   # How many background colors to key
```

**Use cases:**
- Key out sky/environment, keep subjects
- Invert: key out people, keep backgrounds
- Create ghostly floating subject effects
- Blend environments from multiple videos

---

## Recommended Implementation

For `blend-video-alt.py`, I suggest implementing **Approach 2 (Palette) + Approach 4 (Multi-Key)** as a starting point:

```python
# =============================================================================
# COLOR KEYING CONFIGURATION
# =============================================================================

# Color selection mode: 'fixed', 'palette', 'multi', 'kmeans', 'hybrid'
COLOR_MODE = 'palette'

# For 'fixed' mode - single color
COLORKEY_HEX = '0xFFFFFF'

# For 'palette' mode - random selection from these
COLOR_PALETTES = {
    'classic': ['0xFFFFFF', '0x000000'],
    'neutrals': ['0xFFFFFF', '0x000000', '0x808080', '0xC0C0C0'],
    'warm': ['0xFFFFFF', '0xFF6B35', '0xFFA500'],
    'cool': ['0x000000', '0x0066CC', '0x00CED1'],
    'vibrant': ['0xFF0000', '0x00FF00', '0x0000FF', '0xFFFF00', '0xFF00FF'],
}
ACTIVE_PALETTE = 'classic'

# For 'multi' mode - key multiple colors per layer
MULTI_KEY_COUNT = 2  # How many colors to key out per video

# Randomization
RANDOMIZE_PER_VIDEO = True   # Different colors per layer
RANDOMIZE_PER_RUN = True     # Different colors each time script runs

# Similarity/blend (applies to all modes)
SIMILARITY = 0.3
BLEND = 0.1
```

---

## Currently Implemented

All of these modes are now available in `blend-video-alt.py`:

| Mode | Description | Dependencies |
|------|-------------|--------------|
| `fixed` | Single hardcoded color | None |
| `kmeans` | ML-based dominant color extraction | numpy, sklearn, PIL |
| `luminance` | Key out brights or darks | numpy, PIL |
| `rembg` | ML background removal + color keying | rembg, onnxruntime |

**Set the mode:**
```python
COLOR_MODE = 'rembg'  # Options: 'fixed', 'kmeans', 'luminance', 'rembg'
```

All modes now include randomized similarity and blend ranges for more variety.

