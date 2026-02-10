"""
sd-batch-frames.py - Extract frames from video and process through Stable Diffusion

Pipeline:
1. Convert video to target FPS
2. Extract all frames
3. Select random subset of frames
4. Send each to fal.ai SD for img2img style transfer
5. Save processed frames

Usage:
  python sd-batch-frames.py --input video.mp4 --prompt "3d dreamcast, 1997 low poly"
  python sd-batch-frames.py --input video.mp4 --num-frames 20 --fps 14
"""

import os
import sys
import subprocess
import shutil
import random
import base64
import json
import urllib.request
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
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULTS = {
    'fps': 14,
    'num_frames': 20,
    'strength': 0.65,        # How much to transform (0-1)
    'guidance_scale': 7.5,   # How strictly to follow prompt (5-15 typical)
    'num_steps': 25,         # Inference steps (15-50 typical)
    'scheduler': None,       # None = API default, or specify scheduler name
    'output_dir': 'projects/archive/output',
    'rate_limit': 0.5,       # Seconds between API calls
}

# Available schedulers for fal.ai SDXL
SCHEDULERS = [
    'DPM++ 2M',
    'DPM++ 2M Karras',
    'DPM++ 2M SDE',
    'DPM++ 2M SDE Karras',
    'Euler',
    'Euler A',  # Euler Ancestral
]


def encode_image(image_path):
    """Encode image to base64 data URI."""
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    ext = Path(image_path).suffix.lower()
    mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
    return f"data:{mime_type};base64,{image_data}"


def download_image(url, output_path):
    """Download image from URL and ensure it's saved as PNG."""
    temp_path = output_path + '.tmp'
    urllib.request.urlretrieve(url, temp_path)
    
    try:
        img = Image.open(temp_path)
        img.save(output_path, 'PNG')
        os.remove(temp_path)
    except Exception:
        os.rename(temp_path, output_path)


def get_video_info(video_path):
    """Get video duration and frame count."""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration,nb_frames,r_frame_rate',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ], capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        stream = data.get('streams', [{}])[0]
        fmt = data.get('format', {})
        
        width = stream.get('width', 0)
        height = stream.get('height', 0)
        duration = float(fmt.get('duration', stream.get('duration', 0)))
        
        return {
            'width': width,
            'height': height,
            'duration': duration,
        }
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None


def extract_frames(video_path, output_dir, fps):
    """Extract all frames from video at target FPS."""
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-i', video_path,
        '-vf', f'fps={fps}',
        f'{output_dir}/frame_%05d.png'
    ]
    
    subprocess.run(cmd, check=True)
    
    frames = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    return frames


def process_frame_sd(input_path, output_path, prompt, strength, seed, 
                      negative_prompt="", guidance_scale=7.5, num_steps=25, scheduler=None):
    """Process a frame through Stable Diffusion via fal.ai."""
    try:
        import fal_client
    except ImportError:
        print("Error: fal-client not installed. Run: pip install fal-client")
        return False
    
    if not os.environ.get('FAL_KEY'):
        print("Error: FAL_KEY not set in environment")
        return False
    
    image_uri = encode_image(input_path)
    
    arguments = {
        "image_url": image_uri,
        "prompt": prompt,
        "negative_prompt": negative_prompt or "blurry, low quality, watermark, photorealistic, modern",
        "strength": strength,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }
    
    # Add scheduler if specified
    if scheduler:
        arguments["scheduler"] = scheduler
    
    try:
        result = fal_client.subscribe(
            "fal-ai/fast-sdxl/image-to-image",
            arguments=arguments
        )
        
        if result and 'images' in result and len(result['images']) > 0:
            download_image(result['images'][0]['url'], output_path)
            return True
    except Exception as e:
        print(f"\nError: {e}")
    
    return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract and process frames through SD')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input video path')
    parser.add_argument('--prompt', '-p', type=str, required=True, help='SD prompt')
    parser.add_argument('--negative-prompt', type=str, default='', help='Negative prompt')
    parser.add_argument('--fps', type=int, default=DEFAULTS['fps'], help='Target FPS for extraction')
    parser.add_argument('--num-frames', '-n', type=int, default=DEFAULTS['num_frames'], 
                        help='Number of random frames to process')
    parser.add_argument('--strength', '-s', type=float, default=DEFAULTS['strength'],
                        help='SD strength (0-1, higher = more transformation)')
    parser.add_argument('--guidance-scale', '-g', type=float, default=DEFAULTS['guidance_scale'],
                        help='How strictly to follow prompt (5-15 typical)')
    parser.add_argument('--num-steps', type=int, default=DEFAULTS['num_steps'],
                        help='Inference steps (15-50 typical, more = higher quality)')
    parser.add_argument('--scheduler', type=str, default=DEFAULTS['scheduler'],
                        choices=SCHEDULERS + [None],
                        help=f'Noise scheduler: {", ".join(SCHEDULERS)}')
    parser.add_argument('--rate-limit', type=float, default=DEFAULTS['rate_limit'],
                        help='Seconds between API calls')
    parser.add_argument('--output-dir', '-o', default=DEFAULTS['output_dir'])
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary files')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (auto-generated if not set)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Get video info
    info = get_video_info(args.input)
    if not info:
        print("Error: Could not get video info")
        sys.exit(1)
    
    print("=" * 60)
    print("SD BATCH FRAME PROCESSOR")
    print("=" * 60)
    print(f"Input: {os.path.basename(args.input)}")
    print(f"Duration: {info['duration']:.1f}s")
    print(f"Target FPS: {args.fps}")
    print(f"Frames to process: {args.num_frames}")
    print(f"Prompt: {args.prompt}")
    print(f"Strength: {args.strength}")
    print(f"Guidance: {args.guidance_scale}")
    print(f"Steps: {args.num_steps}")
    print(f"Scheduler: {args.scheduler or 'default'}")
    print("=" * 60)
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = f"sd_batch_work_{timestamp}"
    frames_dir = os.path.join(work_dir, "frames")
    output_frames_dir = os.path.join(work_dir, "output")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)
    
    seed = args.seed if args.seed else random.randint(0, 2**31 - 1)
    print(f"Seed: {seed}")
    
    try:
        # Step 1: Extract all frames
        print(f"\n[1/3] Extracting frames at {args.fps}fps...")
        all_frames = extract_frames(args.input, frames_dir, args.fps)
        print(f"  Extracted {len(all_frames)} frames")
        
        # Step 2: Select random subset
        num_to_select = min(args.num_frames, len(all_frames))
        selected_indices = sorted(random.sample(range(len(all_frames)), num_to_select))
        selected_frames = [all_frames[i] for i in selected_indices]
        
        print(f"\n[2/3] Selected {len(selected_frames)} random frames")
        print(f"  Frame indices: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''}")
        
        # Step 3: Process each selected frame
        print(f"\n[3/3] Processing frames through SD...")
        start_time = time.time()
        processed = 0
        failed = 0
        results = []
        
        for i, frame_name in enumerate(selected_frames):
            input_path = os.path.join(frames_dir, frame_name)
            output_path = os.path.join(output_frames_dir, frame_name)
            
            success = process_frame_sd(
                input_path, output_path, 
                args.prompt, args.strength, seed,
                args.negative_prompt,
                args.guidance_scale, args.num_steps, args.scheduler
            )
            
            if success:
                processed += 1
                results.append({
                    'frame': frame_name,
                    'index': selected_indices[i],
                    'output': output_path,
                    'status': 'success'
                })
            else:
                failed += 1
                results.append({
                    'frame': frame_name,
                    'index': selected_indices[i],
                    'status': 'failed'
                })
            
            # Progress
            elapsed = time.time() - start_time
            per_frame = elapsed / (i + 1)
            eta = per_frame * (len(selected_frames) - i - 1)
            print(f"  [{i+1}/{len(selected_frames)}] {frame_name} - {per_frame:.1f}s/frame - ETA: {eta:.0f}s")
            
            # Rate limiting
            if args.rate_limit > 0 and i < len(selected_frames) - 1:
                time.sleep(args.rate_limit)
        
        print(f"\n  Done! Processed: {processed}, Failed: {failed}")
        
        # Copy processed frames to output directory
        output_subdir = os.path.join(args.output_dir, f"sd_frames_{timestamp}")
        os.makedirs(output_subdir, exist_ok=True)
        
        for result in results:
            if result['status'] == 'success':
                src = result['output']
                dst = os.path.join(output_subdir, result['frame'])
                shutil.copy(src, dst)
        
        # Also copy originals for comparison
        originals_dir = os.path.join(output_subdir, "originals")
        os.makedirs(originals_dir, exist_ok=True)
        for frame_name in selected_frames:
            src = os.path.join(frames_dir, frame_name)
            dst = os.path.join(originals_dir, frame_name)
            shutil.copy(src, dst)
        
        # Save metadata
        metadata = {
            'input': args.input,
            'prompt': args.prompt,
            'negative_prompt': args.negative_prompt,
            'fps': args.fps,
            'strength': args.strength,
            'guidance_scale': args.guidance_scale,
            'num_steps': args.num_steps,
            'scheduler': args.scheduler,
            'seed': seed,
            'total_frames': len(all_frames),
            'processed_frames': processed,
            'failed_frames': failed,
            'selected_indices': selected_indices,
            'results': results,
        }
        
        meta_path = os.path.join(output_subdir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✓ COMPLETE!")
        print("=" * 60)
        print(f"Processed frames: {output_subdir}")
        print(f"Originals: {originals_dir}")
        print(f"Metadata: {meta_path}")
        print("=" * 60)
        
    finally:
        if not args.keep_temp and os.path.exists(work_dir):
            print("\nCleaning up...")
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()

