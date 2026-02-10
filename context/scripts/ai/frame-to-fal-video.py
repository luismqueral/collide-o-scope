"""
frame-to-fal-video.py - Generate video, extract frame, feed to fal video models

Pipeline:
1. Generate a blended video using blend-video-alt logic at 12fps
2. Extract a random frame from the generated video
3. Feed that frame into fal.ai image-to-video models (Kling, Minimax, etc.)

Usage:
  python frame-to-fal-video.py                    # Full pipeline with defaults
  python frame-to-fal-video.py --model kling      # Use Kling 2.5 model
  python frame-to-fal-video.py --model minimax    # Use Minimax video model
  python frame-to-fal-video.py --duration 5       # Generate 5 second AI video
  python frame-to-fal-video.py --skip-blend --input frame.png  # Use existing frame
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
    # Blend video settings
    'blend_fps': 12,           # FPS for generated blend video
    'blend_duration': 30,      # Duration of blend video (seconds)
    'blend_size': (768, 768),  # Size of blend video
    'num_videos': 3,           # Number of videos to blend
    
    # AI video generation settings
    'ai_duration': 5,          # Duration of AI-generated video (seconds)
    'model': 'minimax',         # Default AI model (cheapest option)
    'prompt': '',              # Optional prompt for video generation
    
    # Paths
    'video_input': 'library/video',
    'output_dir': 'projects/archive/output',
}

# Available fal.ai video models
FAL_VIDEO_MODELS = {
    'kling': {
        'id': 'fal-ai/kling-video/v1.5/pro/image-to-video',
        'name': 'Kling 1.5 Pro',
        'max_duration': 10,
        'supports_prompt': True,
    },
    'kling-turbo': {
        'id': 'fal-ai/kling-video/v1/standard/image-to-video',
        'name': 'Kling 1.0 Standard (Turbo)',
        'max_duration': 5,
        'supports_prompt': True,
    },
    'minimax': {
        'id': 'fal-ai/minimax/video-01/image-to-video',
        'name': 'MiniMax Video-01',
        'max_duration': 6,
        'supports_prompt': True,
    },
    'minimax-fast': {
        'id': 'fal-ai/minimax/video-01-live/image-to-video',
        'name': 'MiniMax Video-01 Live (Fast)',
        'max_duration': 6,
        'supports_prompt': True,
    },
    'luma': {
        'id': 'fal-ai/luma-dream-machine/image-to-video',
        'name': 'Luma Dream Machine',
        'max_duration': 5,
        'supports_prompt': True,
    },
    'runway': {
        'id': 'fal-ai/runway/gen3/alpha/image-to-video',
        'name': 'Runway Gen-3 Alpha',
        'max_duration': 10,
        'supports_prompt': True,
    },
    'hunyuan': {
        'id': 'fal-ai/hunyuan-video/image-to-video',
        'name': 'Hunyuan Video',
        'max_duration': 5,
        'supports_prompt': True,
    },
    'wan': {
        'id': 'fal-ai/wan/image-to-video',
        'name': 'Wan',
        'max_duration': 5,
        'supports_prompt': True,
    },
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def encode_image_to_base64(image_path):
    """Encode image to base64 data URI."""
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Detect format
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


def extract_random_frame(video_path, output_path):
    """Extract a random frame from a video."""
    duration = get_video_duration(video_path)
    
    # Pick random timestamp (avoid first and last 10%)
    margin = duration * 0.1
    timestamp = random.uniform(margin, duration - margin)
    
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-ss', str(timestamp),
        '-i', video_path,
        '-frames:v', '1',
        '-q:v', '2',
        output_path
    ]
    
    subprocess.run(cmd, check=True)
    print(f"Extracted frame at {timestamp:.2f}s from {os.path.basename(video_path)}")
    
    return timestamp


# =============================================================================
# BLEND VIDEO GENERATION (inline from blend-video-alt)
# =============================================================================

def get_random_videos(folder, num_videos, min_duration, hd_only=False, min_resolution=720):
    """Select random videos that are long enough and optionally HD."""
    videos = [
        os.path.join(folder, f) 
        for f in os.listdir(folder) 
        if f.endswith(('.mp4', '.mov', '.avi', '.webm'))
    ]
    
    # Filter by duration and optionally by resolution
    valid = []
    hd_count = 0
    for v in videos:
        dur = get_video_duration(v)
        if dur >= min_duration:
            if hd_only:
                width, height = get_video_resolution(v)
                if height >= min_resolution:
                    valid.append((v, dur, height))
                    hd_count += 1
            else:
                valid.append((v, dur, 0))
    
    if hd_only:
        print(f"HD Mode: Found {hd_count} videos >= {min_resolution}p")
    
    if len(valid) < num_videos:
        print(f"Warning: Only {len(valid)} videos meet requirements")
        num_videos = min(num_videos, len(valid))
    
    # Select random videos
    selected = random.sample(valid, num_videos)
    
    # Calculate random start times
    result = []
    for video_path, duration, height in selected:
        max_start = max(0, duration - min_duration)
        start_time = random.uniform(0, max_start) if max_start > 0 else 0
        result.append((video_path, start_time))
    
    return result


def generate_blend_video(output_path, video_folder, fps, duration, size, num_videos, hd_only=False, min_resolution=720):
    """
    Generate a blended video using the blend-video-alt approach.
    Returns the path to the generated video.
    """
    print("\n" + "=" * 60)
    print("STEP 1: GENERATING BLEND VIDEO")
    print("=" * 60)
    
    if hd_only:
        print(f"[HD Mode: Only using videos >= {min_resolution}p]")
    
    # Select videos
    videos = get_random_videos(video_folder, num_videos, duration, hd_only, min_resolution)
    
    if not videos:
        print("Error: No suitable videos found")
        return None
    
    print(f"Selected {len(videos)} videos:")
    for i, (path, start) in enumerate(videos):
        print(f"  {i+1}. {os.path.basename(path)} (start: {start:.1f}s)")
    
    # Build filter complex
    filter_parts = []
    
    # Process base layer (no colorkey)
    filter_parts.append(
        f"[0:v]trim=start={videos[0][1]}:duration={duration},setpts=PTS-STARTPTS,"
        f"loop=loop=-1:size={duration * fps},setpts=N/({fps}*TB),"
        f"scale={size[0]}:{size[1]},setsar=1[v0]"
    )
    
    # Process overlay layers with luminance-based colorkey
    for i in range(1, len(videos)):
        # Random luminance keying (lights or darks)
        if random.random() > 0.5:
            # Key out lights
            brightness = random.randint(200, 255)
            color = f"0x{brightness:02X}{brightness:02X}{brightness:02X}"
        else:
            # Key out darks
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
    
    # Build overlay chain
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
    
    # Build ffmpeg command
    cmd = ['ffmpeg', '-y']
    
    for video_path, _ in videos:
        cmd.extend(['-i', video_path])
    
    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '[video]',
        '-an',  # No audio for this intermediate step
        '-t', str(duration),
        '-r', str(fps),
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'veryfast',
        output_path
    ])
    
    print(f"\nGenerating {duration}s video at {fps}fps...")
    subprocess.run(cmd, check=True)
    print(f"✓ Blend video saved: {output_path}")
    
    return output_path


# =============================================================================
# FAL.AI VIDEO GENERATION
# =============================================================================

def generate_fal_video(image_path, output_path, model_key, duration, prompt=None):
    """
    Generate a video from an image using fal.ai models.
    """
    try:
        import fal_client
    except ImportError:
        print("Error: fal-client not installed. Run: pip install fal-client")
        sys.exit(1)
    
    if not os.environ.get('FAL_KEY'):
        print("Error: FAL_KEY not set in environment")
        sys.exit(1)
    
    model_info = FAL_VIDEO_MODELS.get(model_key)
    if not model_info:
        print(f"Error: Unknown model '{model_key}'")
        print(f"Available models: {', '.join(FAL_VIDEO_MODELS.keys())}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("STEP 3: GENERATING AI VIDEO")
    print("=" * 60)
    print(f"Model: {model_info['name']}")
    print(f"Duration: {min(duration, model_info['max_duration'])}s (max: {model_info['max_duration']}s)")
    
    # Encode image
    image_uri = encode_image_to_base64(image_path)
    
    # Build arguments based on model
    arguments = {
        "image_url": image_uri,
    }
    
    # Add duration if supported
    actual_duration = min(duration, model_info['max_duration'])
    if model_key in ['kling', 'kling-turbo']:
        arguments["duration"] = "5" if actual_duration <= 5 else "10"
    elif model_key == 'minimax':
        # MiniMax uses prompt_optimizer
        arguments["prompt_optimizer"] = True
    elif model_key == 'luma':
        arguments["aspect_ratio"] = "4:3"  # Closest to square that Luma supports
    elif model_key == 'runway':
        arguments["duration"] = actual_duration
    
    # Add prompt if provided and supported
    if prompt and model_info.get('supports_prompt'):
        arguments["prompt"] = prompt
        print(f"Prompt: {prompt}")
    elif not prompt:
        # Default Seaman Dreamcast aesthetic prompt
        default_prompt = (
            "[Tracking shot] slow lateral pan across murky aquarium interior, "
            "late 90s 3D graphics, glass distortion with low-poly tank geometry, "
            "sediment particles floating in green-brown water, "
            "unfiltered rock textures shimmering, institutional ambient hum, "
            "observation without interaction, Seaman Dreamcast aesthetic"
        )
        arguments["prompt"] = default_prompt
        print(f"Default prompt: {arguments['prompt']}")
    
    print("\nCalling fal.ai API...")
    start_time = time.time()
    
    try:
        result = fal_client.subscribe(
            model_info['id'],
            arguments=arguments,
            with_logs=True,
        )
    except Exception as e:
        print(f"Error calling fal.ai: {e}")
        return None
    
    elapsed = time.time() - start_time
    print(f"API call completed in {elapsed:.1f}s")
    
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
        print(f"Error: Could not extract video URL from response")
        print(f"Response: {json.dumps(result, indent=2)[:500]}")
        return None
    
    # Download video
    print(f"Downloading video...")
    import urllib.request
    urllib.request.urlretrieve(video_url, output_path)
    print(f"✓ AI video saved: {output_path}")
    
    return output_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate blend video → extract frame → create AI video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODELS:
  kling        Kling 1.5 Pro (up to 10s)
  kling-turbo  Kling 1.0 Standard (up to 5s, faster)
  minimax      MiniMax Video-01 (up to 6s)
  luma         Luma Dream Machine (up to 5s)
  runway       Runway Gen-3 Turbo (up to 10s)
  hunyuan      Hunyuan Video (up to 5s)
  wan          Wan 2.1 (up to 5s)

EXAMPLES:
  # Full pipeline with Kling
  python frame-to-fal-video.py --model kling
  
  # Use MiniMax with custom prompt
  python frame-to-fal-video.py --model minimax --prompt "flowing abstract motion"
  
  # Skip blend generation, use existing frame
  python frame-to-fal-video.py --skip-blend --input my_frame.png
  
  # Longer blend video before extraction
  python frame-to-fal-video.py --blend-duration 60
        """
    )
    
    # Pipeline control
    parser.add_argument('--skip-blend', action='store_true',
                        help='Skip blend video generation, use --input frame instead')
    parser.add_argument('--input', '-i', type=str,
                        help='Input frame path (when using --skip-blend)')
    parser.add_argument('--skip-fal', action='store_true',
                        help='Skip fal video generation (just generate blend + extract frame)')
    
    # Blend video settings
    parser.add_argument('--blend-fps', type=int, default=DEFAULTS['blend_fps'],
                        help=f"FPS for blend video (default: {DEFAULTS['blend_fps']})")
    parser.add_argument('--blend-duration', type=int, default=DEFAULTS['blend_duration'],
                        help=f"Duration of blend video in seconds (default: {DEFAULTS['blend_duration']})")
    parser.add_argument('--blend-size', type=int, nargs=2, default=list(DEFAULTS['blend_size']),
                        help=f"Size of blend video WxH (default: {DEFAULTS['blend_size'][0]} {DEFAULTS['blend_size'][1]})")
    parser.add_argument('--num-videos', type=int, default=DEFAULTS['num_videos'],
                        help=f"Number of videos to blend (default: {DEFAULTS['num_videos']})")
    parser.add_argument('--hd', action='store_true',
                        help='Only use HD videos (>= 720p) for blending')
    parser.add_argument('--min-resolution', type=int, default=720,
                        help='Minimum video height in pixels for HD mode (default: 720)')
    
    # AI video settings
    parser.add_argument('--model', '-m', type=str, default=DEFAULTS['model'],
                        choices=list(FAL_VIDEO_MODELS.keys()),
                        help=f"AI video model (default: {DEFAULTS['model']})")
    parser.add_argument('--duration', '-d', type=int, default=DEFAULTS['ai_duration'],
                        help=f"AI video duration in seconds (default: {DEFAULTS['ai_duration']})")
    parser.add_argument('--prompt', '-p', type=str, default='',
                        help='Optional prompt for video generation')
    
    # Paths
    parser.add_argument('--video-input', type=str, default=DEFAULTS['video_input'],
                        help=f"Input video folder (default: {DEFAULTS['video_input']})")
    parser.add_argument('--output-dir', '-o', type=str, default=DEFAULTS['output_dir'],
                        help=f"Output directory (default: {DEFAULTS['output_dir']})")
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = f"frame2fal_work_{timestamp}"
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    blend_size = tuple(args.blend_size)
    
    print("=" * 60)
    print("FRAME-TO-FAL VIDEO PIPELINE")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"Work directory: {work_dir}")
    
    try:
        # STEP 1: Generate or load frame
        if args.skip_blend:
            if not args.input:
                print("Error: --skip-blend requires --input")
                sys.exit(1)
            if not os.path.exists(args.input):
                print(f"Error: Input file not found: {args.input}")
                sys.exit(1)
            
            frame_path = args.input
            print(f"\nUsing existing frame: {frame_path}")
        else:
            # Generate blend video
            blend_path = os.path.join(work_dir, "blend_video.mp4")
            
            result = generate_blend_video(
                blend_path,
                args.video_input,
                args.blend_fps,
                args.blend_duration,
                blend_size,
                args.num_videos,
                hd_only=args.hd,
                min_resolution=args.min_resolution
            )
            
            if not result:
                print("Error: Blend video generation failed")
                sys.exit(1)
            
            # STEP 2: Extract random frame
            print("\n" + "=" * 60)
            print("STEP 2: EXTRACTING RANDOM FRAME")
            print("=" * 60)
            
            frame_path = os.path.join(work_dir, "extracted_frame.png")
            frame_timestamp = extract_random_frame(blend_path, frame_path)
            
            # Copy blend video to output
            output_blend = os.path.join(args.output_dir, f"blend_{timestamp}.mp4")
            shutil.copy(blend_path, output_blend)
            print(f"✓ Blend video copied to: {output_blend}")
        
        # STEP 3: Generate AI video (unless skipped)
        if args.skip_fal:
            print("\n[Skipping fal video generation]")
            output_frame = os.path.join(args.output_dir, f"frame_{timestamp}.png")
            shutil.copy(frame_path, output_frame)
            print(f"✓ Frame saved to: {output_frame}")
        else:
            output_video = os.path.join(args.output_dir, f"fal_{args.model}_{timestamp}.mp4")
            
            result = generate_fal_video(
                frame_path,
                output_video,
                args.model,
                args.duration,
                args.prompt if args.prompt else None
            )
            
            if not result:
                print("Error: AI video generation failed")
                sys.exit(1)
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'pipeline': 'frame-to-fal-video',
                'model': args.model,
                'model_name': FAL_VIDEO_MODELS[args.model]['name'],
                'ai_duration': args.duration,
                'prompt': args.prompt if args.prompt else 'auto-generated',
                'blend_settings': {
                    'fps': args.blend_fps,
                    'duration': args.blend_duration,
                    'size': blend_size,
                    'num_videos': args.num_videos,
                } if not args.skip_blend else None,
                'outputs': {
                    'ai_video': output_video,
                    'blend_video': output_blend if not args.skip_blend else None,
                }
            }
            
            metadata_path = os.path.join(args.output_dir, f"fal_{args.model}_{timestamp}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✓ PIPELINE COMPLETE!")
        print("=" * 60)
        
        if not args.skip_fal:
            print(f"AI Video: {output_video}")
            print(f"Metadata: {metadata_path}")
        if not args.skip_blend:
            print(f"Blend Video: {output_blend}")
        
        print("=" * 60)
        
    finally:
        # Cleanup work directory
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            print(f"\nCleaned up: {work_dir}")


if __name__ == "__main__":
    main()

