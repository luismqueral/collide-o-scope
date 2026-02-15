"""
multi-layer.py - N-Video Overlay with Colorkey Compositing

The main video synthesis script. Layers multiple videos on top of each other
using color keying to create transparency between layers.

WHAT IT DOES:
1. Selects random videos from the library
2. Picks a random segment from each video (random start time)
3. Analyzes each video to determine which colors to key out
4. Builds an ffmpeg filter chain: trim → loop → color correct → colorkey → scale → overlay
5. Mixes audio from source videos with stereo panning
6. Outputs the final composited video

COLOR MODES:
  fixed     - use a hardcoded hex color (classic white/black keying)
  kmeans    - ML extracts dominant colors from each video frame
  luminance - keys out brights or darks based on frame analysis
  rembg     - ML background removal, keys detected bg/fg colors
  random    - randomly picks from random_mode_choices each run

USAGE:
  python scripts/blend/multi-layer.py
  python scripts/blend/multi-layer.py --preset kmeans-default
  python scripts/blend/multi-layer.py --fps 24 --num-videos 5
  python scripts/blend/multi-layer.py --preset classic-white --project dark-city-loop
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# add project root to path so we can import from scripts.utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.utils.config import load_config
from scripts.utils.cache import build_cache, load_cache, VIDEO_EXTENSIONS
from scripts.utils.random import create_rng
from scripts.utils.ffprobe import (
    get_video_duration,
    get_video_resolution,
    video_has_audio,
)
from scripts.text.metadata import generate_metadata, generate_response_title

# optional imports for ML color modes (graceful fallback if not installed)
try:
    import numpy as np
    from PIL import Image
    from sklearn.cluster import KMeans
    KMEANS_AVAILABLE = True
except ImportError:
    KMEANS_AVAILABLE = False
    print("Note: numpy/sklearn/PIL not installed. kmeans mode unavailable.")

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


# =============================================================================
# DEFAULTS
#
# These control video selection, compositing, color keying, audio mixing,
# and encoding. Any value can be overridden by a preset or CLI flag.
#
# Ranges like [min, max] = "pick randomly within this window each run."
# Single values = "use exactly this."
# =============================================================================

DEFAULTS = {

    # --- source selection ---

    # where to find scraped footage
    "video_input": "library/video",

    # how many videos to layer on top of each other
    # 2 = simple overlay, 3-5 = denser/more abstract
    "num_videos": 3,

    # only use videos with height >= this (in pixels)
    # 720 = HD, 1080 = Full HD, 0 = no filtering
    "min_resolution": 720,

    # --- timing ---

    # output duration in seconds
    # use [min, max] for random range, or a single number for exact
    "duration": [60, 120],

    # output frame rate
    # 12-18 = dreamy/choppy, 24 = film, 30 = video standard
    "fps": 30,

    # --- color keying ---

    # which algorithm picks the transparency color
    # "fixed"      = always use colorkey_hex
    # "kmeans"     = extract dominant colors from each video via ML
    # "luminance"  = key out brights or darks based on frame analysis
    # "rembg"      = ML background removal, key detected bg/fg colors
    # "random"     = randomly pick from random_mode_choices each run
    "mode": "fixed",

    # for "fixed" mode: the hex color to make transparent
    # 0xFFFFFF = white (skies, highlights), 0x000000 = black (shadows)
    "colorkey_hex": "0xFFFFFF",

    # for "random" mode: which modes to randomly choose between
    "random_mode_choices": ["luminance", "rembg"],

    # how close a pixel must be to the key color to become transparent
    # lower = stricter (less transparency), higher = looser (more)
    "similarity": [0.2, 0.4],

    # edge softness around transparent areas
    # 0.0 = hard pixelated edges, higher = feathered/soft
    "blend": [0.0, 0.05],

    # --- kmeans mode ---

    # how many dominant colors to extract from each video frame
    "kmeans_colors": [3, 6],

    # how many of those extracted colors to actually key out
    "kmeans_key_count": [1, 3],

    # minimum squared RGB distance between colors to count as distinct
    "kmeans_distinct_threshold": 2000,

    # --- luminance mode ---

    # what to key out: "lights", "darks", "auto", or "random"
    # auto = analyze frame brightness and key whichever dominates
    "luminance_target": "random",

    # brightness threshold (0.0-1.0) for what counts as "light" or "dark"
    "luminance_threshold": [0.6, 0.9],

    # --- rembg mode ---

    # what to key out: "background", "foreground", or "random"
    "rembg_target": "background",

    # how many colors to extract from the detected region
    "rembg_colors": [2, 4],

    # save debug frames (original, mask, preview) to output directory
    "save_debug_frames": False,

    # --- color correction (applied per layer before keying) ---

    # stretch color levels to use full range (fixes washed out footage)
    "auto_normalize": True,

    # random color shifts per layer for visual variety
    "color_shift": True,

    # hue rotation in degrees per layer
    # [-12, 12] = subtle variation, [-180, 180] = wild
    "hue_shift": [-12, 12],

    # saturation multiplier: 1.0 = unchanged, 0.5 = muted, 1.5 = vivid
    "saturation": [0.85, 1.25],

    # brightness offset: 0.0 = unchanged, negative = darker, positive = brighter
    "brightness": [-0.08, 0.08],

    # contrast multiplier: 1.0 = unchanged
    "contrast": [0.9, 1.15],

    # gamma: 1.0 = unchanged, <1 = brighter mids, >1 = darker mids
    "gamma": [0.9, 1.1],

    # --- audio ---

    # how many audio tracks to pull from source videos
    "audio_sources": [1, 3],

    # spread audio sources across the stereo field
    "stereo_panning": True,

    # auto-adjust volume levels (EBU R128 loudness standard)
    "normalize_audio": True,

    # --- encoding ---

    # output dimensions [width, height]
    "output_size": [1920, 1080],

    # where to save output
    "output_dir": "projects/archive/output",

    # h264 quality: 18 = high/large, 23 = balanced, 28 = small/lossy
    "crf": 23,

    # encoding speed: ultrafast, veryfast, fast, medium, slow
    # faster = bigger file, slower = smaller file at same quality
    "encoding_preset": "veryfast",

    # --- reproducibility ---

    # set to an integer to reproduce exact outputs
    # None = different result every run (seed is logged for replay)
    "seed": None,

    # optional comment text to steer title generation.
    # if set, the title absorbs words from this comment.
    "title_comment": None,
}


# =============================================================================
# VIDEO SELECTION
# =============================================================================

def select_videos(rng, config):
    """
    Select random videos from the library and assign random start times.

    Uses the metadata cache for fast filtering — no ffprobe calls during selection.
    Guarantees at least one video with audio is included (for output audio).

    Returns:
        List of (video_path, start_time) tuples, or None if not enough videos
    """
    video_input = config['video_input']
    if not os.path.isabs(video_input):
        video_input = os.path.join(PROJECT_ROOT, video_input)

    cache_file = os.path.join(PROJECT_ROOT, 'video_cache.json')
    cache = build_cache(video_input, cache_file)

    num_videos = config['num_videos']
    min_res = config['min_resolution']
    crop_length = config['_duration']  # resolved duration (single number)

    # filter by resolution
    all_videos = [v for v in cache.keys() if os.path.exists(v)]
    if min_res > 0:
        videos = [v for v in all_videos if cache[v]['height'] >= min_res]
        print(f"HD filter: {len(videos)} videos >= {min_res}p (of {len(all_videos)} total)")
    else:
        videos = all_videos

    if len(videos) < num_videos:
        print(f"Not enough videos. Need {num_videos}, found {len(videos)}.")
        return None

    # guarantee at least one video with audio
    videos_with_audio = [v for v in videos if cache[v]['has_audio']]

    if videos_with_audio:
        guaranteed = [rng.choice(videos_with_audio)]
        remaining = [v for v in videos if v not in guaranteed]
        additional = rng.sample(remaining, min(num_videos - 1, len(remaining)))
        selected = guaranteed + additional
        rng.shuffle(selected)
    else:
        print("Warning: no videos with audio found")
        selected = rng.sample(videos, num_videos)

    # assign random start times
    segments = []
    for video in selected:
        duration = cache[video]['duration']
        if duration <= crop_length:
            start = 0
        else:
            start = rng.randint(0, int(duration - crop_length))
        segments.append((video, start))

    return segments


# =============================================================================
# COLOR EXTRACTION (stays inline until Phase 5)
# =============================================================================

def extract_frame(video_path, output_path, rng, timestamp=None):
    """Extract a single frame from a video at a random or specific time."""
    if timestamp is None:
        duration = get_video_duration(video_path)
        timestamp = rng.uniform(0, max(0, duration - 1))

    subprocess.run([
        'ffmpeg', '-y', '-v', 'error',
        '-ss', str(timestamp),
        '-i', video_path,
        '-frames:v', '1',
        output_path
    ], check=True)


def get_colors_fixed(rng, config):
    """Fixed mode: return the configured hex color."""
    similarity = rng.from_range(config['similarity'])
    blend = rng.from_range(config['blend'])
    return [(config['colorkey_hex'], similarity, blend)]


def get_colors_kmeans(video_path, rng, config):
    """K-means mode: extract dominant colors from a video frame via ML clustering."""
    if not KMEANS_AVAILABLE:
        print("  K-means unavailable, falling back to fixed")
        return get_colors_fixed(rng, config)

    similarity = rng.from_range(config['similarity'])
    blend = rng.from_range(config['blend'])
    num_colors = rng.int_from_range(config['kmeans_colors'])
    num_to_key = rng.int_from_range(config['kmeans_key_count'])
    distinct_threshold = config['kmeans_distinct_threshold']

    temp_frame = f"/tmp/colorkey_frame_{rng.randint(1000, 9999)}.jpg"
    try:
        extract_frame(video_path, temp_frame, rng)

        image = Image.open(temp_frame).convert('RGB')
        image = image.resize((100, 100))
        pixels = np.array(image).reshape(-1, 3)

        kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=None)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)

        # filter out colors too similar to each other
        distinct = [colors[0]]
        for color in colors[1:]:
            if all(np.sum((color - dc)**2) > distinct_threshold for dc in distinct):
                distinct.append(color)

        hex_colors = [
            '0x{:02X}{:02X}{:02X}'.format(int(c[0]), int(c[1]), int(c[2]))
            for c in distinct
        ]

        keys = rng.sample(hex_colors, min(num_to_key, len(hex_colors)))
        print(f"  K-means: found {len(hex_colors)} colors, keying {keys}")
        return [(c, similarity, blend) for c in keys]

    except Exception as e:
        print(f"  K-means failed: {e}, falling back to fixed")
        return get_colors_fixed(rng, config)
    finally:
        if os.path.exists(temp_frame):
            os.remove(temp_frame)


def get_colors_luminance(video_path, rng, config):
    """Luminance mode: key out brights or darks based on frame brightness."""
    similarity = rng.from_range(config['similarity'])
    blend = rng.from_range(config['blend'])
    threshold = rng.from_range(config['luminance_threshold'])

    target = config['luminance_target']
    if target == 'random':
        target = rng.choice(['lights', 'darks'])

    if target == 'auto' and KMEANS_AVAILABLE:
        temp_frame = f"/tmp/luma_frame_{rng.randint(1000, 9999)}.jpg"
        try:
            extract_frame(video_path, temp_frame, rng)
            image = Image.open(temp_frame).convert('L')
            avg = np.mean(np.array(image).flatten()) / 255.0
            target = 'lights' if avg > 0.5 else 'darks'
        except Exception:
            target = 'lights'
        finally:
            if os.path.exists(temp_frame):
                os.remove(temp_frame)
    elif target == 'auto':
        target = 'lights'

    if target == 'lights':
        b = int(255 * threshold)
    else:
        b = int(255 * (1 - threshold))

    color = '0x{:02X}{:02X}{:02X}'.format(b, b, b)
    print(f"  Luminance: keying {target} with {color}")
    return [(color, similarity, blend)]


def get_colors_rembg(video_path, rng, config, video_index=0, debug_dir=None):
    """Rembg mode: use ML background removal to find colors to key out."""
    if not REMBG_AVAILABLE:
        print("  rembg unavailable, falling back to kmeans")
        return get_colors_kmeans(video_path, rng, config)
    if not KMEANS_AVAILABLE:
        print("  numpy/sklearn needed for rembg, falling back to fixed")
        return get_colors_fixed(rng, config)

    similarity = rng.from_range(config['similarity'])
    blend = rng.from_range(config['blend'])
    num_colors = rng.int_from_range(config['rembg_colors'])

    target = config['rembg_target']
    if target == 'random':
        target = rng.choice(['background', 'foreground'])

    temp_frame = f"/tmp/rembg_frame_{rng.randint(1000, 9999)}.png"
    try:
        extract_frame(video_path, temp_frame, rng)

        original_full = Image.open(temp_frame).convert('RGBA')
        foreground_full = rembg_remove(original_full)

        # save debug frames if enabled
        if config['save_debug_frames'] and debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_full.save(f"{debug_dir}/{ts}_v{video_index}_original.png")
            foreground_full.save(f"{debug_dir}/{ts}_v{video_index}_foreground.png")
            alpha = np.array(foreground_full)[:, :, 3]
            Image.fromarray(alpha).save(f"{debug_dir}/{ts}_v{video_index}_mask.png")
            print(f"    Debug frames saved to {debug_dir}/")

        # resize for faster clustering
        original = original_full.resize((150, 150))
        foreground = foreground_full.resize((150, 150))
        alpha = np.array(foreground)[:, :, 3]
        original_pixels = np.array(original.convert('RGB'))

        mask = alpha < 128 if target == 'background' else alpha >= 128
        target_pixels = original_pixels[mask]

        if len(target_pixels) < 100:
            print(f"  Not enough {target} pixels, falling back to kmeans")
            return get_colors_kmeans(video_path, rng, config)

        kmeans = KMeans(n_clusters=min(num_colors, len(target_pixels) // 10), n_init=10)
        kmeans.fit(target_pixels)
        colors = kmeans.cluster_centers_.astype(int)

        hex_colors = [
            '0x{:02X}{:02X}{:02X}'.format(int(c[0]), int(c[1]), int(c[2]))
            for c in colors
        ]
        print(f"  rembg: targeting {target}, keying {hex_colors}")
        return [(c, similarity, blend) for c in hex_colors]

    except Exception as e:
        print(f"  rembg failed: {e}, falling back to kmeans")
        return get_colors_kmeans(video_path, rng, config)
    finally:
        if os.path.exists(temp_frame):
            os.remove(temp_frame)


def get_colorkey_settings(video_path, mode, rng, config, video_index=0, debug_dir=None):
    """Dispatch to the appropriate color extraction mode."""
    if mode == 'fixed':
        return get_colors_fixed(rng, config)
    elif mode == 'kmeans':
        return get_colors_kmeans(video_path, rng, config)
    elif mode == 'luminance':
        return get_colors_luminance(video_path, rng, config)
    elif mode == 'rembg':
        return get_colors_rembg(video_path, rng, config, video_index, debug_dir)
    else:
        print(f"  Unknown mode '{mode}', using fixed")
        return get_colors_fixed(rng, config)


# =============================================================================
# COLOR CORRECTION
# =============================================================================

def get_color_correction_filter(rng, config):
    """
    Build ffmpeg filter string for per-layer color correction.

    Applies histogram normalization and random hue/saturation/brightness/contrast/gamma
    shifts to give each layer a slightly different character.

    Returns:
        (filter_string or None, debug_info_dict)
    """
    filters = []
    debug = {}

    if config['auto_normalize']:
        filters.append("normalize")
        debug['normalized'] = True

    if config['color_shift']:
        hue = rng.from_range(config['hue_shift'])
        sat = rng.from_range(config['saturation'])
        bright = rng.from_range(config['brightness'])
        contrast = rng.from_range(config['contrast'])
        gamma = rng.from_range(config['gamma'])

        # hue filter: h = degrees, s = saturation multiplier
        filters.append(f"hue=h={hue:.1f}:s={sat:.2f}")
        # eq filter: brightness, contrast, gamma
        filters.append(f"eq=brightness={bright:.3f}:contrast={contrast:.2f}:gamma={gamma:.2f}")

        debug.update({
            'hue': round(hue, 1), 'sat': round(sat, 2),
            'bright': round(bright, 3), 'contrast': round(contrast, 2),
            'gamma': round(gamma, 2),
        })

    filter_string = ','.join(filters) if filters else None
    return filter_string, debug


# =============================================================================
# MAIN BLEND
# =============================================================================

def blend(config):
    """
    Main synthesis function. Builds and executes the ffmpeg filter chain.

    Steps:
    1. Select source videos with random trim points
    2. Resolve color mode (handle 'random' selection)
    3. For each layer: build trim → loop → color correct → colorkey → scale chain
    4. Build overlay chain stacking all layers
    5. Build audio chain with panning and normalization
    6. Execute ffmpeg
    """
    rng = create_rng(seed=config['seed'])
    print(f"Seed: {rng.seed_value}")

    # resolve duration from range
    config['_duration'] = rng.int_from_range(config['duration'])
    crop_length = config['_duration']
    fps = config['fps']
    num_videos = config['num_videos']
    output_size = config['output_size']

    # select source videos
    videos = select_videos(rng, config)
    if not videos:
        return None

    # resolve output path
    output_dir = config['output_dir']
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"output_{timestamp}.mp4")
    debug_dir = os.path.join(os.path.dirname(output_path), '..', 'debug') if config['save_debug_frames'] else None

    # resolve color mode
    mode = config['mode']
    if mode == 'random':
        mode = rng.choice(config['random_mode_choices'])
        print(f"\nColor mode: random -> '{mode}'")
    else:
        print(f"\nColor mode: {mode}")

    # =========================================================================
    # BUILD FILTER CHAIN
    # =========================================================================

    filter_parts = []

    # --- base layer (video 0): no colorkey ---
    cc_filter, _ = get_color_correction_filter(rng, config)
    base = (
        f"[0:v]trim=start={videos[0][1]}:duration={crop_length},setpts=PTS-STARTPTS,"
        f"loop=loop=-1:size={crop_length * fps},setpts=N/({fps}*TB),"
    )
    if cc_filter:
        base += f"{cc_filter},"
    base += f"scale={output_size[0]}:{output_size[1]},setsar=1[v0]"
    filter_parts.append(base)

    # --- overlay layers (videos 1 to N-1): with colorkey ---
    colorkey_info = []
    for i in range(1, num_videos):
        video_path = videos[i][0]
        print(f"\nLayer {i}: {os.path.basename(video_path)}")

        cc_filter, _ = get_color_correction_filter(rng, config)
        ck_settings = get_colorkey_settings(video_path, mode, rng, config, i, debug_dir)

        colorkey_info.append({
            'layer': i,
            'colors': [c[0] for c in ck_settings],
            'similarity': round(ck_settings[0][1], 2) if ck_settings else 0,
            'blend': round(ck_settings[0][2], 2) if ck_settings else 0,
        })

        # build colorkey filter chain (may be multiple colorkeys chained)
        ck_filters = ','.join([
            f"colorkey=color={color}:similarity={sim}:blend={bld}"
            for color, sim, bld in ck_settings
        ])

        layer = (
            f"[{i}:v]trim=start={videos[i][1]}:duration={crop_length},setpts=PTS-STARTPTS,"
            f"loop=loop=-1:size={crop_length * fps},setpts=N/({fps}*TB),"
        )
        if cc_filter:
            layer += f"{cc_filter},"
        layer += f"{ck_filters},"
        layer += f"scale={output_size[0]}:{output_size[1]},setsar=1[v{i}]"
        filter_parts.append(layer)

    # --- overlay chain: stack all layers ---
    if num_videos == 2:
        filter_parts.append(f"[v0][v1]overlay=(W-w)/2:(H-h)/2[video]")
    else:
        filter_parts.append("[v0][v1]overlay=(W-w)/2:(H-h)/2[temp1]")
        for i in range(2, num_videos):
            label = "[video]" if i == num_videos - 1 else f"[temp{i}]"
            filter_parts.append(f"[temp{i-1}][v{i}]overlay=(W-w)/2:(H-h)/2{label}")

    # --- audio chain ---
    videos_with_audio = [i for i in range(len(videos)) if video_has_audio(videos[i][0])]
    audio_source_indices = []
    audio_panning = {}

    if videos_with_audio:
        num_audio = rng.int_from_range(config['audio_sources'])
        num_audio = min(num_audio, len(videos_with_audio))
        audio_source_indices = rng.sample(videos_with_audio, num_audio)

        for idx in audio_source_indices:
            if config['stereo_panning']:
                audio_panning[idx] = rng.uniform(-0.8, 0.8)
            else:
                audio_panning[idx] = 0.0

        print(f"\nAudio: {len(audio_source_indices)} source(s)")
        for idx in audio_source_indices:
            pan = audio_panning[idx]
            print(f"  - {os.path.basename(videos[idx][0])} (pan={pan:.2f})")

        audio_labels = []
        for idx in audio_source_indices:
            audio_path = videos[idx][0]
            audio_dur = get_video_duration(audio_path)
            label = f"aud{idx}"

            # trim or loop audio to match output duration
            if audio_dur < crop_length:
                max_start = max(0, audio_dur - 5)
                a_start = rng.uniform(0, max_start) if max_start > 0 else 0
                trim = f"[{idx}:a]atrim=start={a_start:.2f},asetpts=PTS-STARTPTS,aloop=loop=-1:size=2e+09,asetpts=N/SR/TB"
            else:
                a_start = videos[idx][1]
                trim = f"[{idx}:a]atrim=start={a_start}:duration={crop_length},asetpts=PTS-STARTPTS"

            # stereo panning
            pan_val = audio_panning.get(idx, 0.0)
            left_gain = 1.0 - max(0, pan_val)
            right_gain = 1.0 + min(0, pan_val)
            pan = f"pan=stereo|c0={left_gain:.2f}*c0|c1={right_gain:.2f}*c1"

            # loudness normalization
            if config['normalize_audio']:
                norm = "loudnorm=I=-16:TP=-1.5:LRA=11"
                filter_parts.append(f"{trim},{pan},{norm}[{label}]")
            else:
                filter_parts.append(f"{trim},{pan}[{label}]")

            audio_labels.append(f"[{label}]")

        # mix audio sources
        if len(audio_labels) == 1:
            filter_parts.append(f"{audio_labels[0]}acopy[audio]")
        else:
            inputs = ''.join(audio_labels)
            filter_parts.append(f"{inputs}amix=inputs={len(audio_labels)}:duration=longest:normalize=1[audio]")
    else:
        print("\nNo source videos have audio — output will be silent")

    # =========================================================================
    # BUILD AND EXECUTE FFMPEG
    # =========================================================================

    filter_complex = ";".join(filter_parts)

    cmd = ['ffmpeg']
    for video_path, _ in videos:
        cmd.extend(['-i', video_path])

    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '[video]',
    ])

    if audio_source_indices:
        cmd.extend(['-map', '[audio]'])
    else:
        cmd.extend(['-an'])

    # generate metadata and embed in MP4 container
    meta = generate_metadata(rng=rng)
    title_comment = config.get('title_comment')
    if title_comment:
        meta['title'] = generate_response_title(title_comment, rng=rng)
    title = meta['title']
    description = meta['description']

    cmd.extend([
        '-metadata', f'title={title}',
        '-metadata', f'comment={description}',
        '-metadata', 'artist=luis queral',
        '-t', str(crop_length),
        '-r', str(fps),
        '-c:v', 'libx264',
        '-crf', str(config['crf']),
        '-preset', config['encoding_preset'],
        '-c:a', 'aac',
        '-b:a', '192k',
        output_path
    ])

    print(f"\nRendering: {output_path}")
    print(f"Title: {title}")
    print(f"Duration: {crop_length}s @ {fps}fps, {output_size[0]}x{output_size[1]}")
    print(f"Videos: {num_videos}, Mode: {mode}, CRF: {config['crf']}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\nDone: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"\nFFmpeg error: {e}")
        return None


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-layer video compositing with colorkey',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--preset', default=None,
                        help='Preset name (looked up in project presets, then universal)')
    parser.add_argument('--project', default=None,
                        help='Project directory name (for project-level presets and output)')

    parser.add_argument('--mode', default=None,
                        choices=['fixed', 'kmeans', 'luminance', 'rembg', 'random'],
                        help='Color keying mode')
    parser.add_argument('--num-videos', type=int, default=None,
                        help='Number of videos to layer')
    parser.add_argument('--fps', type=int, default=None,
                        help='Output frame rate')
    parser.add_argument('--duration', type=int, default=None,
                        help='Exact output duration in seconds')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory')
    parser.add_argument('--title-comment', default=None,
                        help='comment text to steer title generation')

    return parser.parse_args()


def main():
    args = parse_args()

    # build CLI overrides from flags that were explicitly set
    cli = {}
    if args.mode is not None: cli['mode'] = args.mode
    if args.num_videos is not None: cli['num_videos'] = args.num_videos
    if args.fps is not None: cli['fps'] = args.fps
    if args.duration is not None: cli['duration'] = args.duration
    if args.seed is not None: cli['seed'] = args.seed
    if args.output_dir is not None: cli['output_dir'] = args.output_dir
    if args.title_comment is not None: cli['title_comment'] = args.title_comment

    config = load_config(
        DEFAULTS,
        preset_name=args.preset,
        preset_category='blend',
        project_dir=args.project,
        cli_overrides=cli,
    )

    blend(config)


if __name__ == "__main__":
    main()
