"""
blend-to-luma-chain.py - Generate blended video, then chain through Luma AI

Pipeline:
1. Generate a 7-second blended video
2. Extract last frame → feed to Luma
3. Extract last frame of Luma output → feed to Luma again
4. Concatenate all segments into final video

Usage:
  python blend-to-luma-chain.py                    # Default 2 Luma iterations
  python blend-to-luma-chain.py --iterations 3    # 3 Luma passes
  python blend-to-luma-chain.py --skip-blend --input video.mp4  # Use existing video
"""

import os
import sys
import subprocess
import random
import json
import base64
import shutil
from datetime import datetime
from pathlib import Path
import time

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import fal_client
except ImportError:
    print("Error: fal-client not installed. Run: pip install fal-client")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULTS = {
    'blend_fps': 18,
    'blend_duration': 7,
    'blend_size': (800, 800),
    'num_videos': 3,
    'luma_iterations': 2,
    'luma_duration': 5,
    'video_input': 'library/video',
    'output_dir': 'projects/archive/output',
}

# Default prompt for Luma chain
DEFAULT_PROMPT = (
    "[Tracking shot] slow orbital drift around submerged stone object, "
    "unfiltered textures with visible pixel shimmer, "
    "late 90s console rendering with nearest-neighbor sampling, "
    "murky aquatic void, sediment haze obscuring edges, "
    "hypnotic digital archaeology, ambient drone"
)

LUMA_MODEL = {
    'id': 'fal-ai/luma-dream-machine/image-to-video',
    'name': 'Luma Dream Machine',
}


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


def get_video_resolution(video_path):
    """Get video resolution (width, height)."""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0:s=x',
            video_path
        ], capture_output=True, text=True)
        dims = result.stdout.strip().split('x')
        if len(dims) == 2:
            return (int(dims[0]), int(dims[1]))
        return (0, 0)
    except:
        return (0, 0)


def extract_last_frame(video_path, output_path):
    """Extract the last frame from a video."""
    duration = get_video_duration(video_path)
    # Go slightly before end to ensure we get a frame
    timestamp = max(0, duration - 0.1)
    
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-ss', str(timestamp),
        '-i', video_path,
        '-frames:v', '1',
        '-q:v', '2',
        output_path
    ]
    subprocess.run(cmd, check=True)
    print(f"  Extracted last frame at {timestamp:.2f}s → {os.path.basename(output_path)}")
    return output_path


def get_random_videos(folder, num_videos, min_duration):
    """Select random videos that are long enough."""
    videos = [
        os.path.join(folder, f) 
        for f in os.listdir(folder) 
        if f.endswith(('.mp4', '.mov', '.avi', '.webm'))
    ]
    
    valid = []
    for v in videos:
        dur = get_video_duration(v)
        if dur >= min_duration:
            valid.append((v, dur))
    
    if len(valid) < num_videos:
        num_videos = min(num_videos, len(valid))
    
    selected = random.sample(valid, num_videos)
    
    result = []
    for video_path, duration in selected:
        max_start = max(0, duration - min_duration)
        start_time = random.uniform(0, max_start) if max_start > 0 else 0
        result.append((video_path, start_time))
    
    return result


# =============================================================================
# BLEND VIDEO GENERATION
# =============================================================================

def generate_blend_video(output_path, video_folder, fps, duration, size, num_videos):
    """Generate a blended video using colorkey compositing."""
    print("\n" + "=" * 60)
    print("STEP 1: GENERATING BLEND VIDEO")
    print("=" * 60)
    
    videos = get_random_videos(video_folder, num_videos, duration)
    
    if not videos:
        print("Error: No suitable videos found")
        return None
    
    print(f"Selected {len(videos)} videos:")
    for i, (path, start) in enumerate(videos):
        print(f"  {i+1}. {os.path.basename(path)} (start: {start:.1f}s)")
    
    # Build filter complex
    filter_parts = []
    
    # Base layer
    filter_parts.append(
        f"[0:v]trim=start={videos[0][1]}:duration={duration},setpts=PTS-STARTPTS,"
        f"loop=loop=-1:size={duration * fps},setpts=N/({fps}*TB),"
        f"scale={size[0]}:{size[1]},setsar=1[v0]"
    )
    
    # Overlay layers with colorkey
    for i in range(1, len(videos)):
        if random.random() > 0.5:
            brightness = random.randint(200, 255)
        else:
            brightness = random.randint(0, 50)
        color = f"0x{brightness:02X}{brightness:02X}{brightness:02X}"
        similarity = random.uniform(0.2, 0.4)
        blend = random.uniform(0.0, 0.05)
        
        filter_parts.append(
            f"[{i}:v]trim=start={videos[i][1]}:duration={duration},setpts=PTS-STARTPTS,"
            f"loop=loop=-1:size={duration * fps},setpts=N/({fps}*TB),"
            f"colorkey=color={color}:similarity={similarity}:blend={blend},"
            f"scale={size[0]}:{size[1]},setsar=1[v{i}]"
        )
    
    # Overlay chain
    if len(videos) == 2:
        filter_parts.append("[v0][v1]overlay=(W-w)/2:(H-h)/2[video]")
    else:
        filter_parts.append("[v0][v1]overlay=(W-w)/2:(H-h)/2[temp1]")
        for i in range(2, len(videos)):
            if i == len(videos) - 1:
                filter_parts.append(f"[temp{i-1}][v{i}]overlay=(W-w)/2:(H-h)/2[video]")
            else:
                filter_parts.append(f"[temp{i-1}][v{i}]overlay=(W-w)/2:(H-h)/2[temp{i}]")
    
    filter_complex = ";".join(filter_parts)
    
    cmd = ['ffmpeg', '-y']
    for video_path, _ in videos:
        cmd.extend(['-i', video_path])
    
    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '[video]',
        '-an',
        '-t', str(duration),
        '-r', str(fps),
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'veryfast',
        output_path
    ])
    
    print(f"\nGenerating {duration}s blend video at {fps}fps...")
    subprocess.run(cmd, check=True)
    print(f"✓ Blend video saved: {output_path}")
    
    return output_path


# =============================================================================
# LUMA VIDEO GENERATION
# =============================================================================

def generate_luma_video(image_path, output_path, prompt, iteration):
    """Generate video from image using Luma Dream Machine."""
    print(f"\n--- Luma Iteration {iteration} ---")
    
    if not os.environ.get('FAL_KEY'):
        print("Error: FAL_KEY not set")
        sys.exit(1)
    
    image_uri = encode_image_to_base64(image_path)
    
    arguments = {
        "image_url": image_uri,
        "prompt": prompt,
        "aspect_ratio": "4:3",
    }
    
    print(f"  Prompt: {prompt[:60]}...")
    print("  Calling Luma API...")
    start_time = time.time()
    
    try:
        result = fal_client.subscribe(
            LUMA_MODEL['id'],
            arguments=arguments,
            with_logs=True,
        )
    except Exception as e:
        print(f"  Error: {e}")
        return None
    
    elapsed = time.time() - start_time
    print(f"  API completed in {elapsed:.1f}s")
    
    # Extract video URL
    video_url = None
    if isinstance(result, dict):
        if 'video' in result:
            video_url = result['video'].get('url') if isinstance(result['video'], dict) else result['video']
        elif 'video_url' in result:
            video_url = result['video_url']
    
    if not video_url:
        print(f"  Error: Could not get video URL")
        return None
    
    # Download
    import urllib.request
    urllib.request.urlretrieve(video_url, output_path)
    print(f"  ✓ Saved: {os.path.basename(output_path)}")
    
    return output_path


# =============================================================================
# CONCATENATE VIDEOS
# =============================================================================

def concatenate_videos(video_paths, output_path, target_fps=18, target_size=(800, 800)):
    """Concatenate multiple videos into one, normalizing fps and size."""
    print("\n" + "=" * 60)
    print("FINAL: CONCATENATING ALL SEGMENTS")
    print("=" * 60)
    
    # Build filter for each input
    filter_parts = []
    for i, _ in enumerate(video_paths):
        filter_parts.append(
            f"[{i}:v]fps={target_fps},scale={target_size[0]}:{target_size[1]},setsar=1[v{i}]"
        )
    
    # Concat all
    concat_inputs = "".join([f"[v{i}]" for i in range(len(video_paths))])
    filter_parts.append(f"{concat_inputs}concat=n={len(video_paths)}:v=1:a=0[out]")
    
    filter_complex = ";".join(filter_parts)
    
    cmd = ['ffmpeg', '-y', '-v', 'error']
    for vp in video_paths:
        cmd.extend(['-i', vp])
    
    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'fast',
        output_path
    ])
    
    subprocess.run(cmd, check=True)
    
    total_duration = sum(get_video_duration(vp) for vp in video_paths)
    print(f"✓ Final video: {output_path}")
    print(f"  Total duration: {total_duration:.1f}s ({len(video_paths)} segments)")
    
    return output_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Blend video → Luma chain pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--skip-blend', action='store_true',
                        help='Skip blend, use --input video instead')
    parser.add_argument('--input', '-i', type=str,
                        help='Input video (when using --skip-blend)')
    parser.add_argument('--iterations', '-n', type=int, default=DEFAULTS['luma_iterations'],
                        help=f"Number of Luma iterations (default: {DEFAULTS['luma_iterations']})")
    parser.add_argument('--prompt', '-p', type=str, default=DEFAULT_PROMPT,
                        help='Prompt for Luma generation')
    parser.add_argument('--blend-duration', type=int, default=DEFAULTS['blend_duration'],
                        help=f"Blend video duration (default: {DEFAULTS['blend_duration']}s)")
    parser.add_argument('--video-input', type=str, default=DEFAULTS['video_input'],
                        help=f"Source video folder (default: {DEFAULTS['video_input']})")
    parser.add_argument('--output-dir', '-o', type=str, default=DEFAULTS['output_dir'],
                        help=f"Output directory (default: {DEFAULTS['output_dir']})")
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = f"luma_chain_work_{timestamp}"
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("BLEND → LUMA CHAIN PIPELINE")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"Luma iterations: {args.iterations}")
    print(f"Prompt: {args.prompt[:50]}...")
    
    all_segments = []
    
    try:
        # STEP 1: Generate or load initial video
        if args.skip_blend:
            if not args.input or not os.path.exists(args.input):
                print("Error: --skip-blend requires valid --input")
                sys.exit(1)
            initial_video = args.input
            print(f"\nUsing existing video: {initial_video}")
        else:
            initial_video = os.path.join(work_dir, "blend_segment.mp4")
            generate_blend_video(
                initial_video,
                args.video_input,
                DEFAULTS['blend_fps'],
                args.blend_duration,
                DEFAULTS['blend_size'],
                DEFAULTS['num_videos']
            )
        
        all_segments.append(initial_video)
        current_video = initial_video
        
        # STEP 2+: Luma chain
        print("\n" + "=" * 60)
        print(f"STEP 2: LUMA CHAIN ({args.iterations} iterations)")
        print("=" * 60)
        
        for i in range(args.iterations):
            # Extract last frame
            frame_path = os.path.join(work_dir, f"frame_{i}.png")
            extract_last_frame(current_video, frame_path)
            
            # Generate Luma video
            luma_video = os.path.join(work_dir, f"luma_{i}.mp4")
            result = generate_luma_video(frame_path, luma_video, args.prompt, i + 1)
            
            if not result:
                print(f"Luma iteration {i+1} failed, stopping chain")
                break
            
            all_segments.append(luma_video)
            current_video = luma_video
        
        # FINAL: Concatenate all segments
        final_output = os.path.join(args.output_dir, f"luma_chain_{timestamp}.mp4")
        concatenate_videos(all_segments, final_output, DEFAULTS['blend_fps'], DEFAULTS['blend_size'])
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'pipeline': 'blend-to-luma-chain',
            'luma_iterations': args.iterations,
            'prompt': args.prompt,
            'segments': len(all_segments),
            'output': final_output,
        }
        
        metadata_path = os.path.join(args.output_dir, f"luma_chain_{timestamp}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✓ PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"Output: {final_output}")
        print(f"Segments: {len(all_segments)} (1 blend + {len(all_segments)-1} Luma)")
        print("=" * 60)
        
    finally:
        # Cleanup
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            print(f"\nCleaned up: {work_dir}")


if __name__ == "__main__":
    main()



