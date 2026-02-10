"""
video-chain-generator.py - Generate chained image-to-video sequences

Creates a continuous video by chaining AI-generated video segments.
Each segment uses the last frame of the previous segment as input.

Uses fal-ai/minimax/hailuo-02-fast/image-to-video model.

Usage:
  python video-chain-generator.py --input frame.png                    # Start from image
  python video-chain-generator.py --input frame.png --target-duration 60  # 1 minute chain
  python video-chain-generator.py --input frame.png --prompt "flowing abstract motion"
"""

import os
import sys
import subprocess
import json
import base64
import shutil
import urllib.request
from datetime import datetime
from pathlib import Path
import time
import tempfile

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: PIL or numpy not installed")
    print("Install with: pip install pillow numpy")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULTS = {
    'target_duration': 60,    # Target total duration in seconds
    'prompt': '',             # Optional prompt for video generation
    'output_dir': 'output',
}

# Available models
MODELS = {
    'minimax': {
        'id': 'fal-ai/minimax/hailuo-02-fast/image-to-video',
        'name': 'MiniMax Hailuo-02-Fast',
        'approx_duration': 6,
    },
    'seedance': {
        'id': 'fal-ai/bytedance/seedance/v1/pro/fast/image-to-video',
        'name': 'Seedance 1.0 Pro Fast',
        'approx_duration': 5,
    },
    'kling': {
        'id': 'fal-ai/kling-video/v1.5/pro/image-to-video',
        'name': 'Kling 1.5 Pro',
        'approx_duration': 5,
    },
    'luma': {
        'id': 'fal-ai/luma-dream-machine/image-to-video',
        'name': 'Luma Dream Machine',
        'approx_duration': 5,
    },
}

DEFAULT_MODEL = 'minimax'

# Intensity modifiers for prompt evolution
INTENSITY_LEVELS = [
    "",  # Level 0: base prompt only
    ", subtle movement",
    ", growing intensity, slight distortion",
    ", building tension, reality shifting",
    ", intensifying, unstable, surreal",
    ", dramatic transformation, warping",
    ", intense chaos, dissolving boundaries",
    ", climactic, reality breaking apart",
    ", apocalyptic intensity, maximum distortion",
    ", transcendent chaos, beyond reality",
]


def get_evolved_prompt(base_prompt, segment_num, total_segments, intensity_curve='linear'):
    """
    Generate an evolved prompt based on segment position.
    
    intensity_curve options:
    - 'linear': steady increase
    - 'exponential': slow start, rapid end
    - 'late_bloom': mostly calm, intense ending
    - 'front_loaded': intense at start, calms down
    - 'chaotic': random intensity each segment
    """
    if total_segments <= 1:
        return base_prompt
    
    # Calculate progress (0.0 to 1.0)
    progress = (segment_num - 1) / (total_segments - 1)
    
    # Apply curve
    if intensity_curve == 'exponential':
        # Slow start, rapid acceleration at end
        progress = progress ** 2
    elif intensity_curve == 'late_bloom':
        # Stay low until 60%, then ramp up fast
        if progress < 0.6:
            progress = progress * 0.3
        else:
            progress = 0.18 + (progress - 0.6) * 2.05
        progress = min(progress, 1.0)
    elif intensity_curve == 'front_loaded':
        # Start intense, calm down over time (invert the progress)
        progress = 1.0 - progress
    elif intensity_curve == 'chaotic':
        # Random intensity each segment
        import random
        progress = random.random()
    # 'linear' is default - no modification needed
    
    # Map progress to intensity level (0-9)
    intensity_index = int(progress * (len(INTENSITY_LEVELS) - 1))
    intensity_index = min(intensity_index, len(INTENSITY_LEVELS) - 1)
    
    intensity_modifier = INTENSITY_LEVELS[intensity_index]
    
    return base_prompt + intensity_modifier


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def encode_image_to_base64(image_path):
    """Encode image to base64 data URI."""
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    ext = Path(image_path).suffix.lower()
    mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
    
    return f"data:{mime_type};base64,{image_data}"


def get_video_duration(video_path):
    """Get video duration in seconds."""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ], capture_output=True, text=True)
        return float(result.stdout.strip())
    except:
        return 0


def get_video_frame_count(video_path):
    """Get total frame count of a video."""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_read_frames',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ], capture_output=True, text=True)
        return int(result.stdout.strip())
    except:
        return 0


def extract_last_frame(video_path, output_path):
    """
    Extract the last frame from a video.
    Uses ffmpeg to seek to the end and grab the final frame.
    """
    duration = get_video_duration(video_path)
    
    if duration <= 0:
        print(f"Warning: Could not determine video duration for {video_path}")
        # Fallback: try to extract using frame count
        frame_count = get_video_frame_count(video_path)
        if frame_count > 0:
            # Use select filter to get last frame
            cmd = [
                'ffmpeg', '-y', '-v', 'error',
                '-i', video_path,
                '-vf', f'select=eq(n\\,{frame_count-1})',
                '-vframes', '1',
                '-q:v', '2',
                output_path
            ]
        else:
            # Last resort: just grab a frame near the end
            cmd = [
                'ffmpeg', '-y', '-v', 'error',
                '-sseof', '-0.1',  # 0.1 seconds before end
                '-i', video_path,
                '-frames:v', '1',
                '-q:v', '2',
                output_path
            ]
    else:
        # Seek to just before the end
        seek_time = max(0, duration - 0.05)  # 50ms before end
        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-ss', str(seek_time),
            '-i', video_path,
            '-frames:v', '1',
            '-q:v', '2',
            output_path
        ]
    
    subprocess.run(cmd, check=True)
    
    if os.path.exists(output_path):
        return True
    return False


def download_video(url, output_path):
    """Download video from URL."""
    urllib.request.urlretrieve(url, output_path)


def concatenate_videos(video_paths, output_path, fps=None):
    """
    Concatenate multiple videos into a single video.
    Uses ffmpeg concat demuxer for seamless joining.
    """
    # Create a temporary file list with absolute paths
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for video_path in video_paths:
            # Use absolute path and escape single quotes
            abs_path = os.path.abspath(video_path)
            escaped_path = abs_path.replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
        concat_list = f.name
    
    try:
        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list,
            '-c:v', 'libx264',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
        ]
        
        if fps:
            cmd.extend(['-r', str(fps)])
        
        cmd.append(output_path)
        
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(concat_list)


# =============================================================================
# FAL.AI VIDEO GENERATION
# =============================================================================

def generate_video_segment(image_path, output_path, model_key='minimax', prompt=None, segment_num=1):
    """
    Generate a single video segment from an image using fal.ai.
    """
    try:
        import fal_client
    except ImportError:
        print("Error: fal-client not installed. Run: pip install fal-client")
        sys.exit(1)
    
    if not os.environ.get('FAL_KEY'):
        print("Error: FAL_KEY not set in environment")
        sys.exit(1)
    
    model_config = MODELS.get(model_key, MODELS[DEFAULT_MODEL])
    
    # Encode image
    image_uri = encode_image_to_base64(image_path)
    
    # Build arguments
    arguments = {
        "image_url": image_uri,
    }
    
    # Add prompt (required for most models)
    if prompt:
        arguments["prompt"] = prompt
    else:
        # Default abstract prompt
        arguments["prompt"] = "slow organic motion, dreamlike movement, flowing transformation"
    
    print(f"  → Calling fal.ai API for segment {segment_num} ({model_config['name']})...")
    start_time = time.time()
    
    try:
        result = fal_client.subscribe(
            model_config['id'],
            arguments=arguments,
            with_logs=False,
        )
    except Exception as e:
        print(f"  ✗ API error: {e}")
        return None
    
    elapsed = time.time() - start_time
    print(f"  → API completed in {elapsed:.1f}s")
    
    # Extract video URL from result
    video_url = None
    
    if isinstance(result, dict):
        # Try different possible response structures
        if 'video' in result:
            video_url = result['video'].get('url') if isinstance(result['video'], dict) else result['video']
        elif 'video_url' in result:
            video_url = result['video_url']
        elif 'output' in result:
            video_url = result['output'].get('url') if isinstance(result['output'], dict) else result['output']
    
    if not video_url:
        print(f"  ✗ Could not extract video URL from response")
        print(f"  Response keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
        return None
    
    # Download video
    print(f"  → Downloading video...")
    download_video(video_url, output_path)
    
    duration = get_video_duration(output_path)
    print(f"  ✓ Segment {segment_num} complete ({duration:.1f}s)")
    
    return duration


# =============================================================================
# MAIN CHAIN GENERATOR
# =============================================================================

def generate_video_chain(input_image, target_duration, prompt, output_dir, work_dir, model_key='minimax', evolve_prompts=False, intensity_curve='linear'):
    """
    Generate a chain of videos, each starting from the last frame of the previous.
    
    evolve_prompts: If True, prompts gradually intensify over the duration
    intensity_curve: 'linear', 'exponential', or 'late_bloom'
    """
    model_config = MODELS.get(model_key, MODELS[DEFAULT_MODEL])
    
    # Estimate number of segments needed
    estimated_segments = int(target_duration / model_config['approx_duration']) + 1
    
    print(f"\n{'=' * 60}")
    print("VIDEO CHAIN GENERATOR")
    print(f"{'=' * 60}")
    print(f"Model: {model_config['name']}")
    print(f"Target duration: {target_duration}s")
    print(f"Estimated segments: ~{estimated_segments}")
    if prompt:
        print(f"Base prompt: {prompt}")
    if evolve_prompts:
        print(f"Prompt evolution: ENABLED ({intensity_curve} curve)")
        print(f"  Start: \"{prompt}\"")
        print(f"  End:   \"{prompt}{INTENSITY_LEVELS[-1]}\"")
    print(f"{'=' * 60}\n")
    
    # Track generated segments
    segments = []
    total_duration = 0
    current_frame = input_image
    segment_num = 0
    
    # Copy initial frame to work dir
    initial_frame = os.path.join(work_dir, "frame_000_initial.png")
    img = Image.open(input_image)
    img.save(initial_frame, 'PNG')
    current_frame = initial_frame
    
    print(f"Starting from: {os.path.basename(input_image)}")
    print(f"Image size: {img.size}")
    print()
    
    # Generate segments until we reach target duration
    while total_duration < target_duration:
        segment_num += 1
        
        print(f"[Segment {segment_num}] Generating...")
        print(f"  Input: {os.path.basename(current_frame)}")
        
        # Generate video segment
        segment_path = os.path.join(work_dir, f"segment_{segment_num:03d}.mp4")
        
        # Get evolved prompt if evolution is enabled
        if evolve_prompts and prompt:
            current_prompt = get_evolved_prompt(prompt, segment_num, estimated_segments, intensity_curve)
            print(f"  Prompt: {current_prompt}")
        else:
            current_prompt = prompt
        
        duration = generate_video_segment(
            current_frame,
            segment_path,
            model_key=model_key,
            prompt=current_prompt,
            segment_num=segment_num
        )
        
        if duration is None or duration <= 0:
            print(f"  ✗ Failed to generate segment {segment_num}")
            if segment_num == 1:
                print("First segment failed. Cannot continue.")
                return None
            print("Continuing with existing segments...")
            break
        
        segments.append({
            'path': segment_path,
            'duration': duration,
            'input_frame': current_frame,
        })
        
        total_duration += duration
        
        print(f"  Total duration so far: {total_duration:.1f}s / {target_duration}s")
        print()
        
        # Check if we've reached target
        if total_duration >= target_duration:
            print("Target duration reached!")
            break
        
        # Extract last frame for next segment
        next_frame = os.path.join(work_dir, f"frame_{segment_num:03d}_lastframe.png")
        
        print(f"  → Extracting last frame...")
        success = extract_last_frame(segment_path, next_frame)
        
        if not success or not os.path.exists(next_frame):
            print(f"  ✗ Failed to extract last frame from segment {segment_num}")
            print("Continuing with existing segments...")
            break
        
        current_frame = next_frame
        print(f"  ✓ Last frame extracted for next iteration")
        print()
    
    if not segments:
        print("No segments generated!")
        return None
    
    # Concatenate all segments
    print(f"\n{'=' * 60}")
    print("CONCATENATING SEGMENTS")
    print(f"{'=' * 60}")
    print(f"Total segments: {len(segments)}")
    print(f"Total duration: {total_duration:.1f}s")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output = os.path.join(output_dir, f"chain_{timestamp}.mp4")
    
    video_paths = [s['path'] for s in segments]
    concatenate_videos(video_paths, final_output)
    
    final_duration = get_video_duration(final_output)
    print(f"Final video duration: {final_duration:.1f}s")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model': model_config['id'],
        'model_name': model_config['name'],
        'target_duration': target_duration,
        'actual_duration': final_duration,
        'num_segments': len(segments),
        'prompt': prompt,
        'input_image': os.path.basename(input_image),
        'segments': [
            {
                'segment': i + 1,
                'duration': s['duration'],
            }
            for i, s in enumerate(segments)
        ]
    }
    
    metadata_path = os.path.join(output_dir, f"chain_{timestamp}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        'output': final_output,
        'metadata_path': metadata_path,
        'duration': final_duration,
        'segments': len(segments),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate chained image-to-video sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Generate 1 minute chain from an image
  python video-chain-generator.py --input frame.png
  
  # Generate 2 minute chain with custom prompt
  python video-chain-generator.py --input frame.png --target-duration 120 --prompt "flowing abstract motion"
  
  # Quick test - generate ~30 seconds
  python video-chain-generator.py --input frame.png --target-duration 30

HOW IT WORKS:
  1. Takes your input image
  2. Generates a ~6 second video using MiniMax Hailuo-02-Fast
  3. Extracts the last frame from that video
  4. Uses that frame as input for the next video
  5. Repeats until target duration is reached
  6. Concatenates all segments into final video
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Starting image path (PNG or JPG)')
    parser.add_argument('--target-duration', '-d', type=int, default=DEFAULTS['target_duration'],
                        help=f"Target total duration in seconds (default: {DEFAULTS['target_duration']})")
    parser.add_argument('--prompt', '-p', type=str, default='',
                        help='Optional prompt to guide video generation')
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL,
                        choices=list(MODELS.keys()),
                        help=f"Model to use (default: {DEFAULT_MODEL}). Options: {', '.join(MODELS.keys())}")
    parser.add_argument('--evolve', action='store_true',
                        help='Enable prompt evolution - prompts gradually intensify over duration')
    parser.add_argument('--intensity-curve', type=str, default='linear',
                        choices=['linear', 'exponential', 'late_bloom', 'front_loaded', 'chaotic'],
                        help='Intensity curve for prompt evolution (default: linear). front_loaded=intense start, calms down')
    parser.add_argument('--output-dir', '-o', type=str, default=DEFAULTS['output_dir'],
                        help=f"Output directory (default: {DEFAULTS['output_dir']})")
    parser.add_argument('--keep-segments', action='store_true',
                        help='Keep individual segment files')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = f"chain_work_{timestamp}"
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        result = generate_video_chain(
            args.input,
            args.target_duration,
            args.prompt,
            args.output_dir,
            work_dir,
            model_key=args.model,
            evolve_prompts=args.evolve,
            intensity_curve=args.intensity_curve
        )
        
        if result:
            print(f"\n{'=' * 60}")
            print("✓ CHAIN COMPLETE!")
            print(f"{'=' * 60}")
            print(f"Output: {result['output']}")
            print(f"Duration: {result['duration']:.1f}s")
            print(f"Segments: {result['segments']}")
            print(f"Metadata: {result['metadata_path']}")
            print(f"{'=' * 60}")
            
            if args.keep_segments:
                segments_dir = os.path.join(args.output_dir, f"chain_{timestamp}_segments")
                shutil.copytree(work_dir, segments_dir)
                print(f"\nSegments saved to: {segments_dir}")
        else:
            print("\n✗ Chain generation failed")
            sys.exit(1)
            
    finally:
        if not args.keep_segments and os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            print(f"\nCleaned up: {work_dir}")


if __name__ == "__main__":
    main()

