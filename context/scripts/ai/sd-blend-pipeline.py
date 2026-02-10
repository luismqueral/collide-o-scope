"""
sd-blend-pipeline.py - Blend Videos + Stable Diffusion Frame-by-Frame

Two-stage pipeline:
1. Generate a blended video using blend-video-alt logic (luminance colorkey compositing)
2. Process each frame through Stable Diffusion for stylization

Usage:
  python sd-blend-pipeline.py --prompt "dreamlike textures" --duration 10 --fps 14
  python sd-blend-pipeline.py --prompt "oil painting" --strength 0.6 --model sdxl
"""

import os
import sys
import subprocess
import shutil
import random
import json
import base64
import time
from datetime import datetime
from pathlib import Path

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
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULTS = {
    'fps': 14,
    'duration': 10,
    'width': 768,
    'height': 768,
    'num_videos': 3,
    'strength': 0.5,
    'guidance_scale': 7.5,
    'backend': 'replicate',
    'model': 'sdxl',
    'rate_limit': 1.5,
    'video_input': 'library/video',
    'output_dir': 'projects/archive/output',
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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
        num_videos = len(valid)
    
    selected = random.sample(valid, num_videos)
    
    result = []
    for video_path, duration in selected:
        max_start = max(0, duration - min_duration)
        start_time = random.uniform(0, max_start) if max_start > 0 else 0
        result.append((video_path, start_time))
    
    return result


def generate_blend_video(output_path, video_folder, fps, duration, size, num_videos):
    """Generate a blended video using luminance colorkey compositing."""
    print("\n" + "=" * 60)
    print("STAGE 1: GENERATING BLEND VIDEO")
    print("=" * 60)
    
    videos = get_random_videos(video_folder, num_videos, duration)
    
    if not videos:
        print("Error: No suitable videos found")
        return None
    
    print(f"Selected {len(videos)} videos:")
    for i, (path, start) in enumerate(videos):
        print(f"  {i+1}. {os.path.basename(path)} (start: {start:.1f}s)")
    
    # Build filter complex with luminance colorkey
    filter_parts = []
    
    # Base layer (no colorkey)
    filter_parts.append(
        f"[0:v]trim=start={videos[0][1]}:duration={duration},setpts=PTS-STARTPTS,"
        f"loop=loop=-1:size={duration * fps},setpts=N/({fps}*TB),"
        f"scale={size[0]}:{size[1]},setsar=1[v0]"
    )
    
    # Overlay layers with random luminance keying
    for i in range(1, len(videos)):
        # Random: key out lights or darks
        if random.random() > 0.5:
            brightness = random.randint(200, 255)
            color = f"0x{brightness:02X}{brightness:02X}{brightness:02X}"
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
    
    print(f"\nGenerating {duration}s blend at {fps}fps, {size[0]}x{size[1]}...")
    subprocess.run(cmd, check=True)
    print(f"✓ Blend video: {output_path}")
    
    return output_path


def extract_frames(video_path, output_dir, fps, duration, width, height):
    """Extract frames from video."""
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-i', video_path,
        '-t', str(duration),
        '-vf', f'fps={fps},scale={width}:{height}',
        f'{output_dir}/frame_%05d.png'
    ]
    
    subprocess.run(cmd, check=True)
    frames = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    return frames


def download_image(url, output_path):
    """Download image from URL and ensure PNG format."""
    import urllib.request
    temp_path = output_path + '.tmp'
    urllib.request.urlretrieve(url, temp_path)
    
    # Convert to PNG if needed (APIs often return JPEGs)
    try:
        img = Image.open(temp_path)
        img.save(output_path, 'PNG')
        os.remove(temp_path)
    except Exception:
        os.rename(temp_path, output_path)


def process_frame_replicate(image_path, prompt, negative_prompt, strength, guidance_scale, seed, width, height, model_id, max_retries=3):
    """Process a single frame through Replicate API with retry logic."""
    import replicate
    
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    image_uri = f"data:image/png;base64,{image_data}"
    
    input_params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": image_uri,
        "prompt_strength": strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": 30,
        "width": width,
        "height": height,
    }
    if seed is not None:
        input_params["seed"] = seed
    
    for attempt in range(max_retries):
        try:
            output = replicate.run(model_id, input=input_params)
            if isinstance(output, list):
                return output[0] if output else None
            return output
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "throttled" in error_str.lower():
                wait_time = 15 * (attempt + 1)  # 15s, 30s, 45s
                print(f"\n  Rate limited, waiting {wait_time}s...", end='', flush=True)
                time.sleep(wait_time)
                continue
            else:
                print(f"\nReplicate error: {e}")
                return None
    
    print(f"\n  Max retries exceeded")
    return None


def process_frame_fal(image_path, prompt, negative_prompt, strength, guidance_scale, seed, width, height):
    """Process a single frame through fal.ai API."""
    import fal_client
    
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    try:
        result = fal_client.subscribe(
            "fal-ai/fast-sdxl/image-to-image",
            arguments={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image_url": f"data:image/png;base64,{image_data}",
                "strength": strength,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "num_inference_steps": 25,
            }
        )
        
        if result and 'images' in result and len(result['images']) > 0:
            return result['images'][0]['url']
        return None
    except Exception as e:
        print(f"\nFal error: {e}")
        return None


def reassemble_video(frames_dir, output_path, fps):
    """Reassemble frames into video."""
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-framerate', str(fps),
        '-i', f'{frames_dir}/frame_%05d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_path
    ]
    subprocess.run(cmd, check=True)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Blend videos + SD frame-by-frame stylization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-p', '--prompt', type=str, required=True,
                        help='Style prompt for SD processing')
    parser.add_argument('-n', '--negative-prompt', type=str, 
                        default='blurry, low quality, watermark, text',
                        help='Negative prompt')
    parser.add_argument('--duration', type=float, default=DEFAULTS['duration'],
                        help=f"Duration in seconds (default: {DEFAULTS['duration']})")
    parser.add_argument('--fps', type=int, default=DEFAULTS['fps'],
                        help=f"Frames per second (default: {DEFAULTS['fps']})")
    parser.add_argument('--width', type=int, default=DEFAULTS['width'])
    parser.add_argument('--height', type=int, default=DEFAULTS['height'])
    parser.add_argument('--num-videos', type=int, default=DEFAULTS['num_videos'],
                        help=f"Number of videos to blend (default: {DEFAULTS['num_videos']})")
    parser.add_argument('-s', '--strength', type=float, default=DEFAULTS['strength'],
                        help=f"Transform strength 0-1 (default: {DEFAULTS['strength']})")
    parser.add_argument('-g', '--guidance-scale', type=float, default=DEFAULTS['guidance_scale'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('-m', '--model', type=str, default=DEFAULTS['model'],
                        choices=['sd15', 'sdxl', 'flux-schnell'])
    parser.add_argument('-b', '--backend', type=str, default='fal',
                        choices=['fal', 'replicate'],
                        help='API backend (default: fal)')
    parser.add_argument('--rate-limit', type=float, default=DEFAULTS['rate_limit'],
                        help='Seconds between API calls')
    parser.add_argument('--video-input', type=str, default=DEFAULTS['video_input'])
    parser.add_argument('-o', '--output-dir', type=str, default=DEFAULTS['output_dir'])
    parser.add_argument('--keep-temp', action='store_true')
    parser.add_argument('--comparison', action='store_true',
                        help='Create side-by-side comparison')
    
    args = parser.parse_args()
    
    # Model IDs
    model_ids = {
        'sd15': 'stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf',
        'sdxl': 'stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc',
        'flux-schnell': 'black-forest-labs/flux-schnell',
    }
    
    model_id = model_ids.get(args.model, model_ids['sdxl'])
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = f"sdblend_work_{timestamp}"
    blend_path = os.path.join(work_dir, "blend_source.mp4")
    frames_dir = os.path.join(work_dir, "frames")
    processed_dir = os.path.join(work_dir, "processed")
    
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_frames = int(args.fps * args.duration)
    
    # Use consistent seed
    base_seed = args.seed if args.seed else random.randint(0, 2**31 - 1)
    
    print("=" * 60)
    print("SD BLEND PIPELINE")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Duration: {args.duration}s @ {args.fps}fps = {total_frames} frames")
    print(f"Size: {args.width}x{args.height}")
    print(f"Strength: {args.strength}")
    print(f"Backend: {args.backend} / Model: {args.model}")
    print(f"Seed: {base_seed}")
    print("=" * 60)
    
    try:
        # STAGE 1: Generate blend video
        result = generate_blend_video(
            blend_path,
            args.video_input,
            args.fps,
            args.duration,
            (args.width, args.height),
            args.num_videos
        )
        
        if not result:
            print("Error: Blend generation failed")
            sys.exit(1)
        
        # STAGE 2: Extract frames
        print("\n" + "=" * 60)
        print("STAGE 2: EXTRACTING FRAMES")
        print("=" * 60)
        
        frame_files = extract_frames(
            blend_path, frames_dir,
            args.fps, args.duration,
            args.width, args.height
        )
        print(f"Extracted {len(frame_files)} frames")
        
        # STAGE 3: Process through SD
        print("\n" + "=" * 60)
        print("STAGE 3: STABLE DIFFUSION PROCESSING")
        print("=" * 60)
        print(f"Model: {args.model}")
        print(f"Rate limit: {args.rate_limit}s between calls")
        
        os.makedirs(processed_dir, exist_ok=True)
        
        start_time = time.time()
        processed = 0
        failed = 0
        
        for i, frame_file in enumerate(frame_files):
            input_path = os.path.join(frames_dir, frame_file)
            output_path = os.path.join(processed_dir, frame_file)
            
            if args.backend == 'fal':
                result = process_frame_fal(
                    input_path,
                    args.prompt,
                    args.negative_prompt,
                    args.strength,
                    args.guidance_scale,
                    base_seed,
                    args.width,
                    args.height
                )
            else:
                result = process_frame_replicate(
                    input_path,
                    args.prompt,
                    args.negative_prompt,
                    args.strength,
                    args.guidance_scale,
                    base_seed,
                    args.width,
                    args.height,
                    model_id
                )
            
            if result:
                if isinstance(result, str) and result.startswith('http'):
                    download_image(result, output_path)
                else:
                    download_image(str(result), output_path)
                processed += 1
            else:
                shutil.copy(input_path, output_path)
                failed += 1
            
            elapsed = time.time() - start_time
            per_frame = elapsed / (i + 1)
            eta = per_frame * (len(frame_files) - i - 1)
            print(f"  [{i+1}/{len(frame_files)}] {(i+1)/len(frame_files)*100:.0f}% - {per_frame:.1f}s/frame - ETA: {eta:.0f}s", end='\r')
            
            if args.rate_limit > 0 and i < len(frame_files) - 1:
                time.sleep(args.rate_limit)
        
        print(f"\n  Processed: {processed}, Failed: {failed}")
        
        # STAGE 4: Reassemble
        print("\n" + "=" * 60)
        print("STAGE 4: ASSEMBLING VIDEO")
        print("=" * 60)
        
        output_name = f"sdblend_{args.model}_{timestamp}.mp4"
        output_path = os.path.join(args.output_dir, output_name)
        reassemble_video(processed_dir, output_path, args.fps)
        
        # Also save the blend source
        blend_output = os.path.join(args.output_dir, f"sdblend_{args.model}_{timestamp}_source.mp4")
        shutil.copy(blend_path, blend_output)
        
        # Create comparison if requested
        if args.comparison:
            comparison_dir = os.path.join(work_dir, "comparison")
            os.makedirs(comparison_dir, exist_ok=True)
            
            for frame_file in frame_files:
                orig = Image.open(os.path.join(frames_dir, frame_file))
                proc = Image.open(os.path.join(processed_dir, frame_file)).resize(orig.size)
                comp = Image.new('RGB', (orig.width * 2, orig.height))
                comp.paste(orig, (0, 0))
                comp.paste(proc, (orig.width, 0))
                comp.save(os.path.join(comparison_dir, frame_file))
            
            comp_output = os.path.join(args.output_dir, f"sdblend_{args.model}_{timestamp}_comparison.mp4")
            reassemble_video(comparison_dir, comp_output, args.fps)
            print(f"✓ Comparison: {comp_output}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'prompt': args.prompt,
            'strength': args.strength,
            'model': args.model,
            'fps': args.fps,
            'duration': args.duration,
            'seed': base_seed,
            'num_videos_blended': args.num_videos,
        }
        meta_path = os.path.join(args.output_dir, f"sdblend_{args.model}_{timestamp}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✓ COMPLETE!")
        print("=" * 60)
        print(f"Output: {output_path}")
        print(f"Source: {blend_output}")
        print(f"Metadata: {meta_path}")
        print("=" * 60)
        
    finally:
        if not args.keep_temp and os.path.exists(work_dir):
            print("\nCleaning up...")
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()

