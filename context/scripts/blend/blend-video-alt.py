"""
blend-video-alt.py - Two-Video Overlay with Random Cropping

An alternative (and more refined) version of the video blender. This version:
- Uses exactly 2 videos
- Crops random segments from each video (not always starting at 0:00)
- Properly loops the audio to match the video length
- Uses better FFMPEG encoding settings

WHAT IT DOES:
1. Picks 2 random videos from library/video
2. Calculates a random start time within each video
3. Trims each video to the desired length from that start point
4. Applies colorkey to the TOP video (making white transparent)
5. Overlays the keyed video on the bottom video
6. Adds looped audio from library/audio
7. Outputs with x264 encoding

KEY DIFFERENCE FROM blend-video.py:
- Random start times mean you get different parts of each video each time
- The trim filter extracts a specific segment rather than using the whole video
"""

import os
import random
import subprocess
import json
from datetime import datetime
import time

# Optional imports for K-Means mode (graceful fallback if not installed)
try:
    import numpy as np
    from PIL import Image
    from sklearn.cluster import KMeans
    KMEANS_AVAILABLE = True
except ImportError:
    KMEANS_AVAILABLE = False
    print("Note: numpy/sklearn/PIL not installed. K-Means mode unavailable.")

# Optional import for rembg (background removal)
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("Note: rembg not installed. Background removal mode unavailable. Install with: pip install rembg")

# =============================================================================
# CONFIGURATION - Modify these settings to change the output
# =============================================================================

# Where to find input videos
VIDEO_INPUT = 'library/video'

# Where to save output videos
OUTPUT_DIRECTORY = 'projects/archive/output'

# Cache file for video metadata (resolution, duration, audio)
VIDEO_CACHE_FILE = 'video_metadata_cache.json'

# Output video dimensions (width, height) as a tuple
OUTPUT_SIZE = (1920, 1080)

# =============================================================================
# COLOR KEYING MODE
# =============================================================================
# Options: 'fixed', 'kmeans', 'luminance', 'rembg', or 'random'
#   - 'fixed': Use COLORKEY_HEX (classic white/black keying)
#   - 'kmeans': ML-based extraction of dominant colors from each video
#   - 'luminance': Key out darks or lights based on brightness
#   - 'rembg': ML background removal - keys out detected background colors
#   - 'random': Randomly pick from RANDOM_MODE_CHOICES each run
COLOR_MODE = 'random'

# Which modes to randomly choose from when COLOR_MODE = 'random'
RANDOM_MODE_CHOICES = ['luminance', 'rembg']

# --- FIXED MODE SETTINGS ---
# Format: '0xRRGGBB' - this removes white pixels from the top video
COLORKEY_HEX = '0xFFFFFF'

# --- K-MEANS MODE SETTINGS (randomized ranges!) ---
# How many dominant colors to extract from each video frame
KMEANS_COLORS_RANGE = (3, 6)  # Random between 3-6 colors

# How many of those colors to key out per video
KMEANS_KEY_RANGE = (1, 3)  # Random between 1-3 colors keyed

# Filter out colors too similar to each other (squared RGB distance)
KMEANS_DISTINCT_THRESHOLD = 2000

# --- LUMINANCE MODE SETTINGS ---
# Options: 'lights', 'darks', 'auto', 'random'
#   - 'lights': Key out bright areas (whites)
#   - 'darks': Key out dark areas (blacks)
#   - 'auto': Analyze video and key out whichever is more prevalent
#   - 'random': Randomly choose lights or darks each run
LUMINANCE_TARGET = 'random'

# Luminance threshold (0.0-1.0) - what counts as "light" or "dark"
LUMINANCE_THRESHOLD_RANGE = (0.6, 0.9)  # Randomized

# --- REMBG (BACKGROUND REMOVAL) MODE SETTINGS ---
# What to key out: 'background' or 'foreground'
#   - 'background': Remove detected backgrounds (keep subjects)
#   - 'foreground': Remove detected subjects (keep backgrounds)
#   - 'random': Randomly choose each run
REMBG_TARGET = 'background'

# How many background colors to sample and key
REMBG_COLORS_RANGE = (2, 4)  # Randomized

# --- COMMON SETTINGS (all modes) ---
# How similar a pixel must be to the key color to be removed (0.0-1.0)
SIMILARITY_RANGE = (0.2, 0.4)  # Randomized for variety

# Edge blending for the colorkey (0.0-1.0)
# Lower = harder edges, Higher = softer/feathered edges
BLEND_RANGE = (0.0, 0.05)  # Hard crispy edges

# Exact length for output video in seconds
# Set to None to use a random length from VIDEO_LENGTH_RANGE
VIDEO_LENGTH = None  # Random for production

# Random length range (min, max) in seconds
# Only used if VIDEO_LENGTH is None
VIDEO_LENGTH_RANGE = (120, 300)  # 2-5 minutes

# Output frame rate (frames per second)
# 12 fps = classic stop motion look (jerky, handmade feel)
# 15-16 fps = slightly smoother stop motion
# 24 fps = film standard, 30 fps = video standard
FRAME_RATE = 30  # Back to standard for testing

# Number of videos to blend together
NUM_VIDEOS = 3

# Use audio from source videos instead of external audio files
USE_SOURCE_AUDIO = True

# =============================================================================
# AUDIO SETTINGS
# =============================================================================
# How many audio sources to mix together (randomly selected from videos with audio)
NUM_AUDIO_SOURCES = (1, 3)  # Random between 1-3 audio tracks

# Normalize audio levels (prevents quiet/loud inconsistency)
NORMALIZE_AUDIO = True

# Randomize stereo panning for each audio source
# Each source gets a random position in the stereo field
RANDOM_PANNING = True

# =============================================================================
# COLOR CORRECTION
# =============================================================================
# Auto-normalize: stretches color levels to use full range (fixes washed out/dark videos)
AUTO_NORMALIZE = True

# Random color shifts: adds slight variation to each video layer
COLOR_SHIFT_ENABLED = True

# Hue shift range in degrees (-180 to 180, subtle = -15 to 15)
HUE_SHIFT_RANGE = (-12, 12)

# Saturation multiplier range (1.0 = no change, 0.5 = half, 1.5 = 50% more)
SATURATION_RANGE = (0.85, 1.25)

# Brightness offset range (-1.0 to 1.0, subtle = -0.1 to 0.1)
BRIGHTNESS_RANGE = (-0.08, 0.08)

# Contrast multiplier range (1.0 = no change)
CONTRAST_RANGE = (0.9, 1.15)

# Gamma range (1.0 = no change, <1 = brighter mids, >1 = darker mids)
GAMMA_RANGE = (0.9, 1.1)

# HD Mode - only use videos that are 720p or higher
HD_ONLY = True

# Minimum resolution (height in pixels) for HD mode
# 720 = 720p HD, 1080 = 1080p Full HD
MIN_RESOLUTION = 720

# =============================================================================
# DEBUG MODE
# =============================================================================
# When True, adds a debug panel to the RIGHT of the video with technical info
# Shows: color mode, keyed colors, source videos, similarity/blend values
DEBUG_MODE = False  # Off for batch runs

# Save debug frames showing what rembg detected for each video
# Saves to debug_output/ folder: original frame, mask, and keyed preview
SAVE_DEBUG_FRAMES = False

# Debug panel settings (appears on the right side of the video)
DEBUG_PANEL_WIDTH = 280  # Width of the debug panel in pixels (wider for more info)
DEBUG_FONT = '/System/Library/Fonts/Supplemental/Courier New.ttf'
DEBUG_FONT_SIZE = 10
DEBUG_FONT_COLOR = 'black'  # Dark text on white background
DEBUG_PANEL_BG = 'white'  # Panel background color
DEBUG_LINE_HEIGHT = 14  # Spacing between lines
DEBUG_COLOR_SWATCH_SIZE = 12  # Size of color swatches in pixels

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
# Set to a specific integer to reproduce exact outputs, or None for random
RANDOM_SEED = None  # e.g., 12345 to reproduce, None for random each time


def get_video_duration(video_path):
    """
    Get the duration of a video file in seconds using ffprobe.
    
    ffprobe is FFMPEG's companion tool for reading media file info.
    We ask it for just the duration value, which it returns as a float.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Duration in seconds as a float, or 0 if there's an error
    """
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',  # Suppress ffprobe's usual output
                '-show_entries', 'format=duration',  # Only get duration
                '-of', 'default=noprint_wrappers=1:nokey=1',  # Clean output format
                video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        duration = float(result.stdout)
        return duration
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return 0


def get_video_resolution(video_path):
    """
    Get the resolution (width, height) of a video file using ffprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (width, height) in pixels, or (0, 0) if there's an error
    """
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0:s=x',
                video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        dims = result.stdout.strip().split('x')
        if len(dims) == 2:
            return (int(dims[0]), int(dims[1]))
        return (0, 0)
    except Exception as e:
        print(f"Error getting video resolution: {e}")
        return (0, 0)


# =============================================================================
# VIDEO METADATA CACHING
# =============================================================================

def load_video_cache(cache_file):
    """Load cached video metadata from JSON file."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_video_cache(cache_file, cache_data):
    """Save video metadata cache to JSON file."""
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)


def build_video_cache(folder, cache_file):
    """
    Build/update the video metadata cache.
    
    Only scans new videos that aren't already in the cache.
    Returns the full cache with all video metadata.
    """
    cache = load_video_cache(cache_file)
    
    # Find all video files
    all_videos = [
        os.path.join(folder, file) 
        for file in os.listdir(folder) 
        if file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
    ]
    
    # Check which videos need scanning
    new_videos = [v for v in all_videos if v not in cache]
    
    if new_videos:
        print(f"Scanning {len(new_videos)} new videos (cached: {len(cache)})...")
        for i, video_path in enumerate(new_videos):
            if (i + 1) % 50 == 0:
                print(f"  Scanned {i + 1}/{len(new_videos)}...")
            
            width, height = get_video_resolution(video_path)
            duration = get_video_duration(video_path)
            has_audio = video_has_audio(video_path)
            
            cache[video_path] = {
                'width': width,
                'height': height,
                'duration': duration,
                'has_audio': has_audio
            }
        
        # Save updated cache
        save_video_cache(cache_file, cache)
        print(f"Cache updated: {len(cache)} total videos")
    else:
        print(f"Using cached metadata for {len(cache)} videos")
    
    # Remove entries for deleted videos
    existing_videos = set(all_videos)
    stale_entries = [v for v in cache if v not in existing_videos]
    if stale_entries:
        for v in stale_entries:
            del cache[v]
        save_video_cache(cache_file, cache)
        print(f"Removed {len(stale_entries)} stale cache entries")
    
    return cache


def select_random_videos(folder, crop_length, num_videos, hd_only=False, min_resolution=720):
    """
    Select random videos and calculate random start times for each.
    
    Uses cached metadata for fast filtering - no ffprobe calls during selection!
    
    Args:
        folder: Directory containing video files
        crop_length: How long the output video will be (in seconds)
        num_videos: How many videos to select
        hd_only: If True, only select videos with height >= min_resolution
        min_resolution: Minimum height in pixels (default 720 for HD)
        
    Returns:
        List of tuples: [(video_path, start_time), ...]
        or None if there aren't enough videos
    """
    # Build/load the video metadata cache
    cache = build_video_cache(folder, VIDEO_CACHE_FILE)
    
    # Get all cached video paths that still exist
    all_videos = [v for v in cache.keys() if os.path.exists(v)]
    
    # Filter by resolution if HD mode is enabled (using cached data!)
    if hd_only:
        videos = [v for v in all_videos if cache[v]['height'] >= min_resolution]
        print(f"HD Mode: {len(videos)} HD videos (>= {min_resolution}p) out of {len(all_videos)} total")
    else:
        videos = all_videos
    
    if len(videos) < num_videos:
        print(f"Not enough videos found. Need {num_videos}, found {len(videos)}.")
        if hd_only:
            print("Try setting HD_ONLY = False to include lower resolution videos.")
        return None

    # Split into videos with and without audio (using cached data!)
    videos_with_audio = [v for v in videos if cache[v]['has_audio']]
    
    # Ensure at least one video with audio is selected (for output audio)
    if videos_with_audio:
        # Pick one video with audio first
        guaranteed_audio = [random.choice(videos_with_audio)]
        remaining_pool = [v for v in videos if v not in guaranteed_audio]
        
        # Fill remaining slots from the full pool
        if len(remaining_pool) >= num_videos - 1:
            additional = random.sample(remaining_pool, num_videos - 1)
            selected_videos = guaranteed_audio + additional
        else:
            selected_videos = guaranteed_audio + remaining_pool
        
        # Shuffle so the audio source isn't always first
        random.shuffle(selected_videos)
    else:
        # No videos have audio - just pick randomly
        print("Warning: No videos with audio found!")
        selected_videos = random.sample(videos, num_videos)
    
    video_segments = []

    for video in selected_videos:
        # Use cached duration!
        duration = cache[video]['duration']
        
        # Calculate the maximum start time that still allows crop_length of footage
        # If video is shorter than crop_length, start at 0
        if duration <= crop_length:
            start_time = 0
        else:
            # Random start point, ensuring we have enough video left
            start_time = random.randint(0, int(duration - crop_length))
        
        video_segments.append((video, start_time))

    return video_segments


def video_has_audio(video_path):
    """
    Check if a video file has an audio stream.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        True if the video has audio, False otherwise
    """
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-select_streams', 'a',
                '-show_entries', 'stream=codec_type',
                '-of', 'csv=p=0',
                video_path
            ],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())
    except:
        return False


# =============================================================================
# COLOR EXTRACTION FUNCTIONS
# =============================================================================

def extract_frame_from_video(video_path, output_path, timestamp=None):
    """
    Extract a single frame from a video at a specific or random timestamp.
    
    Args:
        video_path: Path to the source video
        output_path: Where to save the extracted frame
        timestamp: Specific time in seconds, or None for random
    """
    if timestamp is None:
        duration = get_video_duration(video_path)
        timestamp = random.uniform(0, max(0, duration - 1))
    
    subprocess.run([
        'ffmpeg', '-y', '-v', 'error',
        '-ss', str(timestamp),
        '-i', video_path,
        '-frames:v', '1',
        output_path
    ], check=True)


def extract_dominant_colors_kmeans(video_path, num_colors, distinct_threshold=2000):
    """
    Extract dominant colors from a video frame using K-Means clustering.
    
    Args:
        video_path: Path to the video file
        num_colors: Number of dominant colors to extract
        distinct_threshold: Min squared RGB distance between colors
        
    Returns:
        List of hex color strings like ['0xRRGGBB', ...]
    """
    if not KMEANS_AVAILABLE:
        print("K-Means not available, falling back to white")
        return ['0xFFFFFF']
    
    # Extract a random frame
    temp_frame = f"/tmp/colorkey_frame_{random.randint(1000,9999)}.jpg"
    try:
        extract_frame_from_video(video_path, temp_frame)
        
        # Load and resize image for faster processing
        image = Image.open(temp_frame).convert('RGB')
        image = image.resize((100, 100))
        
        # Convert to numpy array of pixels
        pixels = np.array(image).reshape(-1, 3)
        
        # Run K-Means clustering
        kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=None)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Filter out similar colors
        distinct_colors = [colors[0]]
        for color in colors[1:]:
            is_distinct = all(
                np.sum((color - dc)**2) > distinct_threshold 
                for dc in distinct_colors
            )
            if is_distinct:
                distinct_colors.append(color)
        
        # Convert to hex strings
        hex_colors = [
            '0x{:02X}{:02X}{:02X}'.format(int(c[0]), int(c[1]), int(c[2]))
            for c in distinct_colors
        ]
        
        return hex_colors
        
    except Exception as e:
        print(f"K-Means extraction failed: {e}, falling back to white")
        return ['0xFFFFFF']
    finally:
        if os.path.exists(temp_frame):
            os.remove(temp_frame)


def get_luminance_color(video_path, target='random', threshold=0.7):
    """
    Get a color to key based on video luminance analysis.
    
    Args:
        video_path: Path to the video file
        target: 'lights', 'darks', 'auto', or 'random'
        threshold: Luminance threshold (0-1)
        
    Returns:
        Hex color string and adjusted similarity value
    """
    if target == 'random':
        target = random.choice(['lights', 'darks'])
    
    if target == 'auto' and KMEANS_AVAILABLE:
        # Analyze frame brightness
        temp_frame = f"/tmp/luma_frame_{random.randint(1000,9999)}.jpg"
        try:
            extract_frame_from_video(video_path, temp_frame)
            image = Image.open(temp_frame).convert('L')  # Grayscale
            pixels = np.array(image).flatten()
            avg_brightness = np.mean(pixels) / 255.0
            target = 'lights' if avg_brightness > 0.5 else 'darks'
        except:
            target = 'lights'
        finally:
            if os.path.exists(temp_frame):
                os.remove(temp_frame)
    elif target == 'auto':
        target = 'lights'
    
    if target == 'lights':
        # Key out whites/brights
        brightness = int(255 * threshold)
        color = '0x{:02X}{:02X}{:02X}'.format(brightness, brightness, brightness)
    else:
        # Key out blacks/darks
        brightness = int(255 * (1 - threshold))
        color = '0x{:02X}{:02X}{:02X}'.format(brightness, brightness, brightness)
    
    print(f"Luminance mode: keying {target} with color {color}")
    return color


def extract_colors_rembg(video_path, target='background', num_colors=3, video_index=0, timestamp=None):
    """
    Use rembg to identify background/foreground and extract colors from that region.
    
    Args:
        video_path: Path to the video file
        target: 'background' to key backgrounds, 'foreground' to key subjects
        num_colors: How many colors to extract from the target region
        video_index: Which video layer this is (for debug file naming)
        timestamp: Optional timestamp for debug file naming
        
    Returns:
        List of hex color strings from the target region
    """
    if not REMBG_AVAILABLE:
        print("rembg not available, falling back to white")
        return ['0xFFFFFF']
    
    if not KMEANS_AVAILABLE:
        print("numpy/sklearn needed for rembg color extraction, falling back to white")
        return ['0xFFFFFF']
    
    if target == 'random':
        target = random.choice(['background', 'foreground'])
    
    temp_frame = f"/tmp/rembg_frame_{random.randint(1000,9999)}.png"
    
    try:
        # Extract a random frame
        extract_frame_from_video(video_path, temp_frame)
        
        # Load original image (keep full size for debug output)
        original_full = Image.open(temp_frame).convert('RGBA')
        
        # Run rembg to get foreground with alpha (full size for debug)
        foreground_full = rembg_remove(original_full)
        
        # Save debug frames if enabled
        if SAVE_DEBUG_FRAMES:
            debug_dir = 'debug_output'
            os.makedirs(debug_dir, exist_ok=True)
            ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save original frame
            original_full.save(f"{debug_dir}/{ts}_v{video_index}_1_original.png")
            
            # Save foreground with transparency
            foreground_full.save(f"{debug_dir}/{ts}_v{video_index}_2_foreground.png")
            
            # Save mask (white=foreground, black=background)
            alpha_full = np.array(foreground_full)[:, :, 3]
            mask_img = Image.fromarray(alpha_full)
            mask_img.save(f"{debug_dir}/{ts}_v{video_index}_3_mask.png")
            
            # Save visualization of what will be keyed (background highlighted)
            original_rgb = np.array(original_full.convert('RGB'))
            keyed_preview = original_rgb.copy()
            if target == 'background':
                # Tint background areas magenta
                keyed_preview[alpha_full < 128] = (keyed_preview[alpha_full < 128] * 0.5 + np.array([255, 0, 255]) * 0.5).astype(np.uint8)
            else:
                # Tint foreground areas magenta
                keyed_preview[alpha_full >= 128] = (keyed_preview[alpha_full >= 128] * 0.5 + np.array([255, 0, 255]) * 0.5).astype(np.uint8)
            Image.fromarray(keyed_preview).save(f"{debug_dir}/{ts}_v{video_index}_4_keyed_preview.png")
            
            print(f"    Saved debug frames to {debug_dir}/{ts}_v{video_index}_*.png")
        
        # Resize for faster K-Means processing
        original = original_full.resize((150, 150))
        foreground = foreground_full.resize((150, 150))
        
        # Get alpha channel as mask
        alpha = np.array(foreground)[:, :, 3]
        original_pixels = np.array(original.convert('RGB'))
        
        if target == 'background':
            # Get pixels where alpha is LOW (background)
            mask = alpha < 128
            region_name = "background"
        else:
            # Get pixels where alpha is HIGH (foreground)
            mask = alpha >= 128
            region_name = "foreground"
        
        # Extract pixels from target region
        target_pixels = original_pixels[mask]
        
        if len(target_pixels) < 100:
            print(f"  Not enough {region_name} pixels detected, using K-Means fallback")
            return extract_dominant_colors_kmeans(video_path, num_colors, 2000)
        
        # Run K-Means on target region to find dominant colors
        kmeans = KMeans(n_clusters=min(num_colors, len(target_pixels) // 10), n_init=10)
        kmeans.fit(target_pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        hex_colors = [
            '0x{:02X}{:02X}{:02X}'.format(int(c[0]), int(c[1]), int(c[2]))
            for c in colors
        ]
        
        print(f"  rembg: detected {region_name}, keying colors: {hex_colors}")
        return hex_colors
        
    except Exception as e:
        print(f"rembg extraction failed: {e}, falling back to K-Means")
        return extract_dominant_colors_kmeans(video_path, num_colors, 2000)
    finally:
        if os.path.exists(temp_frame):
            os.remove(temp_frame)


def get_color_correction_filter(video_index):
    """
    Generate color correction filter string for a video layer.
    
    Applies:
    1. normalize (if AUTO_NORMALIZE) - stretches levels to full range
    2. Random color shifts (if COLOR_SHIFT_ENABLED) - hue, sat, brightness, contrast, gamma
    
    Args:
        video_index: Which video layer (for logging)
        
    Returns:
        Tuple of (filter_string, debug_info_dict)
    """
    filters = []
    debug = {}
    
    # Auto-normalize: stretch histogram to use full range
    if AUTO_NORMALIZE:
        filters.append("normalize")
        debug['normalized'] = True
    
    # Random color shifts
    if COLOR_SHIFT_ENABLED:
        hue_shift = random.uniform(*HUE_SHIFT_RANGE)
        saturation = random.uniform(*SATURATION_RANGE)
        brightness = random.uniform(*BRIGHTNESS_RANGE)
        contrast = random.uniform(*CONTRAST_RANGE)
        gamma = random.uniform(*GAMMA_RANGE)
        
        # hue filter for hue shift and saturation
        # Note: hue 'h' is in degrees, 's' is a multiplier
        filters.append(f"hue=h={hue_shift:.1f}:s={saturation:.2f}")
        
        # eq filter for brightness, contrast, gamma
        filters.append(f"eq=brightness={brightness:.3f}:contrast={contrast:.2f}:gamma={gamma:.2f}")
        
        debug['hue'] = round(hue_shift, 1)
        debug['sat'] = round(saturation, 2)
        debug['bright'] = round(brightness, 3)
        debug['contrast'] = round(contrast, 2)
        debug['gamma'] = round(gamma, 2)
    
    filter_string = ','.join(filters) if filters else None
    return filter_string, debug


def get_colorkey_settings(video_path, mode, video_index):
    """
    Get colorkey color(s) and settings based on the configured mode.
    
    Args:
        video_path: Path to the video being processed
        mode: 'fixed', 'kmeans', or 'luminance'
        video_index: Which video layer this is (for logging)
        
    Returns:
        List of (color_hex, similarity, blend) tuples for this video
    """
    # Randomize similarity and blend within configured ranges
    similarity = random.uniform(*SIMILARITY_RANGE)
    blend = random.uniform(*BLEND_RANGE)
    
    if mode == 'fixed':
        return [(COLORKEY_HEX, similarity, blend)]
    
    elif mode == 'kmeans':
        if not KMEANS_AVAILABLE:
            print("K-Means not available, using fixed mode")
            return [(COLORKEY_HEX, similarity, blend)]
        
        # Randomize how many colors to extract and key
        num_colors = random.randint(*KMEANS_COLORS_RANGE)
        num_to_key = random.randint(*KMEANS_KEY_RANGE)
        
        colors = extract_dominant_colors_kmeans(
            video_path, num_colors, KMEANS_DISTINCT_THRESHOLD
        )
        
        # Randomly select which colors to key out
        colors_to_key = random.sample(colors, min(num_to_key, len(colors)))
        
        print(f"  Video {video_index}: K-Means found {len(colors)} colors, keying {len(colors_to_key)}: {colors_to_key}")
        
        return [(c, similarity, blend) for c in colors_to_key]
    
    elif mode == 'luminance':
        threshold = random.uniform(*LUMINANCE_THRESHOLD_RANGE)
        color = get_luminance_color(video_path, LUMINANCE_TARGET, threshold)
        return [(color, similarity, blend)]
    
    elif mode == 'rembg':
        if not REMBG_AVAILABLE:
            print("rembg not available, falling back to K-Means")
            return get_colorkey_settings(video_path, 'kmeans', video_index)
        
        num_colors = random.randint(*REMBG_COLORS_RANGE)
        colors = extract_colors_rembg(video_path, REMBG_TARGET, num_colors, video_index)
        
        print(f"  Video {video_index}: rembg targeting {REMBG_TARGET}, keying {len(colors)} colors")
        return [(c, similarity, blend) for c in colors]
    
    else:
        print(f"Unknown color mode '{mode}', using fixed")
        return [(COLORKEY_HEX, similarity, blend)]


def overlay_videos(input_folder, output_folder, output_size, colorkey_hex,
                   video_length, length_range, frame_rate, 
                   num_videos, use_source_audio=True, hd_only=False, min_resolution=720):
    """
    Main function that creates the overlaid video composition.
    
    This builds a complex FFMPEG filter chain that:
    1. Trims each video to start at a random point
    2. Applies colorkey to all videos except the base layer (using COLOR_MODE)
    3. Scales all videos to the target size
    4. Overlays all videos on top of each other
    5. Uses audio from one of the source videos (randomly chosen)
    
    Args:
        input_folder: Directory with source videos
        output_folder: Directory for output
        output_size: Tuple of (width, height)
        colorkey_hex: Color to key out for 'fixed' mode (e.g., '0xFFFFFF')
        video_length: Exact length or None for random
        length_range: (min, max) tuple for random length
        frame_rate: Output FPS
        num_videos: Number of videos to blend together
        use_source_audio: If True, use audio from source videos
        hd_only: If True, only use videos >= min_resolution
        min_resolution: Minimum height in pixels for HD mode (default 720)
    """
    
    # Set up random seed for reproducibility
    if RANDOM_SEED is not None:
        seed = RANDOM_SEED
    else:
        seed = int(time.time() * 1000) % (2**31)  # Generate seed from current time
    random.seed(seed)
    print(f"Random seed: {seed}")
    
    # Determine the output duration
    crop_length = video_length if video_length else random.randint(*length_range)

    # Select our source materials (with optional HD filtering)
    videos = select_random_videos(input_folder, crop_length, num_videos, hd_only, min_resolution)
    
    if not videos:
        return  # Exit if we don't have enough source material
    
    # Find which videos have audio and randomly pick multiple sources
    videos_with_audio = [i for i in range(len(videos)) if video_has_audio(videos[i][0])]
    if videos_with_audio:
        # Determine how many audio sources to use
        num_audio = random.randint(*NUM_AUDIO_SOURCES)
        num_audio = min(num_audio, len(videos_with_audio))  # Can't use more than available
        
        # Randomly select audio sources
        audio_source_indices = random.sample(videos_with_audio, num_audio)
        
        # Generate random panning values for each source (-1.0 = full left, 1.0 = full right)
        audio_panning = {}
        for idx in audio_source_indices:
            if RANDOM_PANNING:
                audio_panning[idx] = random.uniform(-0.8, 0.8)  # Leave some center
            else:
                audio_panning[idx] = 0.0  # Center
        
        print(f"Using {len(audio_source_indices)} audio source(s):")
        for idx in audio_source_indices:
            pan_str = f"pan={audio_panning[idx]:.2f}" if RANDOM_PANNING else ""
            print(f"  - {os.path.basename(videos[idx][0])} {pan_str}")
    else:
        audio_source_indices = []
        audio_panning = {}
        print("No source videos have audio - output will be silent")

    # Generate a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_{timestamp}.mp4"
    output_path = os.path.join(output_folder, output_filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # =============================================================================
    # BUILD THE FILTER COMPLEX DYNAMICALLY
    # 
    # For N videos:
    # 1. Trim and scale video 0 as base layer (no colorkey)
    # 2. For videos 1 to N-1: trim, colorkey(s), scale
    # 3. Overlay all layers on top of each other
    # 4. Loop audio
    # =============================================================================
    
    # Handle random mode selection
    if COLOR_MODE == 'random':
        active_mode = random.choice(RANDOM_MODE_CHOICES)
        print(f"\nColor Mode: random -> selected '{active_mode}'")
    else:
        active_mode = COLOR_MODE
        print(f"\nColor Mode: {active_mode}")
    
    filter_parts = []
    
    # Track color correction info for debug
    color_corrections = {}
    
    # Get color correction for base layer
    cc_filter_0, cc_debug_0 = get_color_correction_filter(0)
    color_corrections[0] = cc_debug_0
    
    # Process the base layer (video 0) - no colorkey
    # trim -> reset pts -> LOOP to fill duration -> reset pts again -> [color correction] -> scale
    base_filter = (
        f"[0:v]trim=start={videos[0][1]}:duration={crop_length},setpts=PTS-STARTPTS,"
        f"loop=loop=-1:size={crop_length * frame_rate},setpts=N/({frame_rate}*TB),"
    )
    if cc_filter_0:
        base_filter += f"{cc_filter_0},"
    base_filter += f"scale={output_size[0]}:{output_size[1]},setsar=1[v0]"
    filter_parts.append(base_filter)
    
    # Load video cache for resolution info
    video_cache = load_video_cache(VIDEO_CACHE_FILE)
    
    # Collect comprehensive debug info
    debug_info = {
        'seed': seed,
        'timestamp': timestamp,
        'mode': active_mode,
        'rembg_target': REMBG_TARGET if active_mode == 'rembg' else None,
        'duration': crop_length,
        'frame_rate': frame_rate,
        'output_size': output_size,
        'hd_only': hd_only,
        'min_resolution': min_resolution,
        'crf': 23,
        'preset': 'veryfast',
        'audio_sources': audio_source_indices,
        'videos': [],
        'colorkeys': []
    }
    
    # Collect per-video info
    for i, (video_path, start_time) in enumerate(videos):
        vid_name = os.path.basename(video_path)
        cached = video_cache.get(video_path, {})
        debug_info['videos'].append({
            'name': vid_name[:28],
            'resolution': f"{cached.get('width', '?')}x{cached.get('height', '?')}",
            'start_offset': start_time,
            'has_audio': cached.get('has_audio', False),
            'is_audio_source': i in audio_source_indices
        })
    
    # Process overlay layers (videos 1 to N-1) - with colorkey(s)
    # trim -> reset pts -> LOOP -> [color correction] -> colorkey(s) -> scale
    for i in range(1, num_videos):
        video_path = videos[i][0]
        
        # Get color correction for this layer
        cc_filter_i, cc_debug_i = get_color_correction_filter(i)
        color_corrections[i] = cc_debug_i
        
        # Get colorkey settings based on mode (may return multiple colors)
        colorkey_settings = get_colorkey_settings(video_path, active_mode, i)
        
        # Store colorkey info for debug overlay
        debug_info['colorkeys'].append({
            'layer': i,
            'colors': [c[0] for c in colorkey_settings],
            'sim': round(colorkey_settings[0][1], 2) if colorkey_settings else 0,
            'blend': round(colorkey_settings[0][2], 2) if colorkey_settings else 0
        })
        
        # Build colorkey filter chain (can be multiple colorkeys chained)
        colorkey_filters = ','.join([
            f"colorkey=color={color}:similarity={sim}:blend={bld}"
            for color, sim, bld in colorkey_settings
        ])
        
        # Build full filter chain for this layer
        layer_filter = (
            f"[{i}:v]trim=start={videos[i][1]}:duration={crop_length},setpts=PTS-STARTPTS,"
            f"loop=loop=-1:size={crop_length * frame_rate},setpts=N/({frame_rate}*TB),"
        )
        if cc_filter_i:
            layer_filter += f"{cc_filter_i},"
        layer_filter += f"{colorkey_filters},"
        layer_filter += f"scale={output_size[0]}:{output_size[1]},setsar=1[v{i}]"
        
        filter_parts.append(layer_filter)
    
    # Add color correction info to debug
    debug_info['color_correction'] = color_corrections
    if AUTO_NORMALIZE or COLOR_SHIFT_ENABLED:
        print(f"Color Correction: normalize={AUTO_NORMALIZE}, shifts={COLOR_SHIFT_ENABLED}")
    
    # Build the overlay chain
    # [v0][v1]overlay -> [temp1]
    # [temp1][v2]overlay -> [temp2]  (if 3+ videos)
    # Final output goes to [video_raw] if DEBUG_MODE, else [video]
    final_label = "[video_raw]" if DEBUG_MODE else "[video]"
    
    if num_videos == 2:
        filter_parts.append(f"[v0][v1]overlay=(W-w)/2:(H-h)/2{final_label}")
    else:
        # First overlay
        filter_parts.append("[v0][v1]overlay=(W-w)/2:(H-h)/2[temp1]")
        # Subsequent overlays
        for i in range(2, num_videos):
            if i == num_videos - 1:
                # Last one outputs to final_label
                filter_parts.append(f"[temp{i-1}][v{i}]overlay=(W-w)/2:(H-h)/2{final_label}")
            else:
                filter_parts.append(f"[temp{i-1}][v{i}]overlay=(W-w)/2:(H-h)/2[temp{i}]")
    
    # Add debug panel to the right side of the video
    if DEBUG_MODE:
        # Calculate panel dimensions
        panel_width = DEBUG_PANEL_WIDTH
        total_width = output_size[0] + panel_width
        
        # First, pad the video to add white space on the right
        filter_parts.append(
            f"[video_raw]pad={total_width}:{output_size[1]}:0:0:color={DEBUG_PANEL_BG}[video_padded]"
        )
        
        # Collect all text lines and color swatches to render
        # Format: ('text', y_pos, is_header, None) or ('swatch', y_pos, False, hex_color)
        debug_elements = []
        y = 8  # Starting y position
        x_pos = output_size[0] + 8  # Left padding in panel
        swatch_size = DEBUG_COLOR_SWATCH_SIZE
        
        # === OUTPUT SECTION ===
        debug_elements.append(('text', 'OUTPUT', y, True, None))
        y += DEBUG_LINE_HEIGHT + 2
        
        debug_elements.append(('text', f"{debug_info['timestamp']}", y, False, None))
        y += DEBUG_LINE_HEIGHT
        debug_elements.append(('text', f"Seed: {debug_info['seed']}", y, False, None))
        y += DEBUG_LINE_HEIGHT
        debug_elements.append(('text', f"Size: {output_size[0]}x{output_size[1]}", y, False, None))
        y += DEBUG_LINE_HEIGHT
        debug_elements.append(('text', f"Duration: {debug_info['duration']}s @ {debug_info['frame_rate']}fps", y, False, None))
        y += DEBUG_LINE_HEIGHT
        debug_elements.append(('text', f"CRF: {debug_info['crf']}  Preset: {debug_info['preset']}", y, False, None))
        y += DEBUG_LINE_HEIGHT
        if debug_info['hd_only']:
            debug_elements.append(('text', f"HD Only: {debug_info['min_resolution']}p+", y, False, None))
            y += DEBUG_LINE_HEIGHT
        
        y += 6  # Section gap
        
        # === SOURCES SECTION ===
        debug_elements.append(('text', 'SOURCES', y, True, None))
        y += DEBUG_LINE_HEIGHT + 2
        
        for i, vid in enumerate(debug_info['videos']):
            # Audio indicator
            audio_icon = "🔊" if vid['is_audio_source'] else ""
            name = vid['name'][:22] + '..' if len(vid['name']) > 24 else vid['name']
            debug_elements.append(('text', f"V{i}: {name} {audio_icon}", y, False, None))
            y += DEBUG_LINE_HEIGHT
            debug_elements.append(('text', f"    {vid['resolution']} @ {vid['start_offset']}s", y, False, None))
            y += DEBUG_LINE_HEIGHT
        
        y += 6  # Section gap
        
        # === COLORKEY SECTION ===
        mode_display = debug_info['mode']
        if debug_info['rembg_target']:
            mode_display += f" ({debug_info['rembg_target']})"
        debug_elements.append(('text', f"COLORKEY: {mode_display}", y, True, None))
        y += DEBUG_LINE_HEIGHT + 2
        
        for ck in debug_info['colorkeys']:
            debug_elements.append(('text', f"Layer {ck['layer']}: sim={ck['sim']} blend={ck['blend']}", y, False, None))
            y += DEBUG_LINE_HEIGHT + 2
            
            # Add color swatches with hex labels
            swatch_x = x_pos
            for color in ck['colors'][:4]:  # Max 4 colors per layer
                # Add colored rectangle
                debug_elements.append(('swatch', swatch_x, y, False, color))
                swatch_x += swatch_size + 4
            
            # Add hex values on next line
            y += swatch_size + 4
            colors_text = ' '.join(c[2:] for c in ck['colors'][:4])  # Remove '0x' prefix
            debug_elements.append(('text', colors_text, y, False, None))
            y += DEBUG_LINE_HEIGHT + 4
        
        # === RENDER ALL ELEMENTS ===
        current_label = "video_padded"
        element_idx = 0
        
        # Separate text and swatch elements
        text_elements = [(e[1], e[2], e[3]) for e in debug_elements if e[0] == 'text']
        swatch_elements = [(e[1], e[2], e[4]) for e in debug_elements if e[0] == 'swatch']
        
        # First, draw all color swatches (drawbox filters)
        for swatch_x, swatch_y, hex_color in swatch_elements:
            next_label = f"swatch{element_idx}"
            # Convert hex like '0xRRGGBB' to ffmpeg color format
            ffmpeg_color = hex_color.replace('0x', '#')
            filter_parts.append(
                f"[{current_label}]drawbox=x={swatch_x}:y={swatch_y}:"
                f"w={swatch_size}:h={swatch_size}:color={ffmpeg_color}:t=fill[{next_label}]"
            )
            current_label = next_label
            element_idx += 1
        
        # Then, draw all text
        for idx, (line_text, y_pos, is_header) in enumerate(text_elements):
            is_last = idx == len(text_elements) - 1
            next_label = "video" if is_last else f"txt{idx}"
            
            # Escape special characters for ffmpeg drawtext
            escaped_text = line_text.replace(':', '\\:').replace("'", "\\'")
            
            # Make headers slightly larger
            font_size = DEBUG_FONT_SIZE + 2 if is_header else DEBUG_FONT_SIZE
            
            filter_parts.append(
                f"[{current_label}]drawtext=text='{escaped_text}':"
                f"fontfile={DEBUG_FONT}:fontsize={font_size}:"
                f"fontcolor={DEBUG_FONT_COLOR}:x={x_pos}:y={y_pos}[{next_label}]"
            )
            current_label = next_label
    
    # Audio: process multiple audio sources with normalization and panning
    if audio_source_indices:
        audio_labels = []
        
        for idx in audio_source_indices:
            audio_video_path = videos[idx][0]
            audio_duration = get_video_duration(audio_video_path)
            audio_label = f"aud{idx}"
            
            # Build the filter chain for this audio source
            # Step 1: Trim or loop to match output duration
            if audio_duration < crop_length:
                # Short audio - pick random start, then loop
                # This avoids always hearing the same beginning of short clips
                max_start = max(0, audio_duration - 5)  # Leave at least 5 seconds before looping
                audio_start = random.uniform(0, max_start) if max_start > 0 else 0
                trim_filter = f"[{idx}:a]atrim=start={audio_start:.2f},asetpts=PTS-STARTPTS,aloop=loop=-1:size=2e+09,asetpts=N/SR/TB"
                print(f"    Audio {idx}: short ({audio_duration:.1f}s), starting at {audio_start:.1f}s then looping")
            else:
                # Long audio - trim from the same start point as video
                audio_start_time = videos[idx][1]
                trim_filter = f"[{idx}:a]atrim=start={audio_start_time}:duration={crop_length},asetpts=PTS-STARTPTS"
                print(f"    Audio {idx}: long ({audio_duration:.1f}s), starting at {audio_start_time}s")
            
            # Step 2: Apply panning (stereo positioning)
            pan_value = audio_panning.get(idx, 0.0)
            # Convert pan value (-1 to 1) to left/right channel gains
            # pan=-1: full left, pan=0: center, pan=1: full right
            left_gain = 1.0 - max(0, pan_value)   # Reduce left when panning right
            right_gain = 1.0 + min(0, pan_value)  # Reduce right when panning left
            pan_filter = f"pan=stereo|c0={left_gain:.2f}*c0|c1={right_gain:.2f}*c1"
            
            # Step 3: Normalize audio levels (EBU R128 loudness normalization)
            if NORMALIZE_AUDIO:
                norm_filter = "loudnorm=I=-16:TP=-1.5:LRA=11"
                filter_parts.append(f"{trim_filter},{pan_filter},{norm_filter}[{audio_label}]")
            else:
                filter_parts.append(f"{trim_filter},{pan_filter}[{audio_label}]")
            
            audio_labels.append(f"[{audio_label}]")
        
        # Mix all audio sources together
        if len(audio_labels) == 1:
            # Single source - just rename it
            filter_parts.append(f"{audio_labels[0]}acopy[audio]")
        else:
            # Multiple sources - mix them with equal weights
            mix_inputs = ''.join(audio_labels)
            # amix with normalized weights so combined audio isn't too loud
            filter_parts.append(f"{mix_inputs}amix=inputs={len(audio_labels)}:duration=longest:normalize=1[audio]")
    
    filter_complex = ";".join(filter_parts)

    # =============================================================================
    # BUILD THE FFMPEG COMMAND
    # =============================================================================
    
    ffmpeg_cmd = ['ffmpeg']
    
    # Add all video inputs
    for i, (video_path, start_time) in enumerate(videos):
        ffmpeg_cmd.extend(['-i', video_path])
    
    # Add filter and output settings
    ffmpeg_cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '[video]',   # Use the [video] output from our filter chain
    ])
    
    # Map audio if available
    if audio_source_indices:
        ffmpeg_cmd.extend(['-map', '[audio]'])
    else:
        ffmpeg_cmd.extend(['-an'])  # No audio
    
    ffmpeg_cmd.extend([
        '-t', str(crop_length),  # Total output duration
        '-r', str(frame_rate),   # Output frame rate
        '-c:v', 'libx264',   # Video codec: H.264
        '-crf', '23',        # Quality (lower = better, 18-28 is typical)
        '-preset', 'veryfast',  # Encoding speed (faster = larger file)
        '-c:a', 'aac',       # Audio codec: AAC
        '-b:a', '192k',      # Audio bitrate
        output_path
    ])

    # Execute the FFMPEG command
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video successfully created at {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during video creation: {e}")


# =============================================================================
# ENTRY POINT - Run the script
# =============================================================================

def main():
    """
    Main entry point for the video blender.
    Can be called directly or imported and called from batch scripts.
    """
    overlay_videos(
        VIDEO_INPUT, 
        OUTPUT_DIRECTORY, 
        OUTPUT_SIZE, 
        COLORKEY_HEX,  # Used for 'fixed' mode
        VIDEO_LENGTH, 
        VIDEO_LENGTH_RANGE, 
        FRAME_RATE, 
        NUM_VIDEOS,
        USE_SOURCE_AUDIO,
        HD_ONLY,
        MIN_RESOLUTION
    )


if __name__ == "__main__":
    main()
