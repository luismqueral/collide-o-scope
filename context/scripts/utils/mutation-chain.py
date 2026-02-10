#!/usr/bin/env python3
"""
mutation-chain.py - Exquisite Corpse Style Image Evolution

Takes a single starting frame and repeatedly processes it through
Stable Diffusion, feeding each output as input to the next iteration.

The image slowly mutates/evolves away from the original, creating
a visual game of telephone.

USAGE:
------
  # Start from first frame of a video, generate 100 mutations
  python mutation-chain.py --input video.mp4 --iterations 100

  # Start from random frame
  python mutation-chain.py --input video.mp4 --iterations 100 --random-start

  # Start from a specific image
  python mutation-chain.py --input image.png --iterations 100

  # With parameter drift over time
  python mutation-chain.py --input video.mp4 --iterations 100 \
    --strength-start 0.3 --strength-end 0.6

"""

import os
import sys
import time
import argparse
import subprocess
import shutil
import random
import json
import urllib.request
from pathlib import Path
from datetime import datetime

try:
    from PIL import Image
    import fal_client
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("Install with: pip install pillow fal-client")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# FRAME EXTRACTION
# =============================================================================

def get_video_info(video_path):
    """Get video duration and frame count."""
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames,duration,r_frame_rate',
        '-of', 'json',
        video_path
    ], capture_output=True, text=True)
    
    info = json.loads(result.stdout)
    stream = info['streams'][0]
    
    duration = float(stream.get('duration', 0))
    nb_frames = int(stream.get('nb_frames', 0))
    
    # Parse frame rate
    fps_str = stream.get('r_frame_rate', '30/1')
    if '/' in fps_str:
        num, den = fps_str.split('/')
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)
    
    return {
        'duration': duration,
        'nb_frames': nb_frames,
        'fps': fps
    }


def extract_frame(video_path, output_path, frame_number=0, random_frame=False):
    """Extract a single frame from video."""
    
    if random_frame:
        info = get_video_info(video_path)
        max_frame = max(1, info['nb_frames'] - 1)
        frame_number = random.randint(0, max_frame)
        print(f"  Selected random frame: {frame_number}/{info['nb_frames']}")
    
    # Calculate timestamp from frame number
    info = get_video_info(video_path)
    timestamp = frame_number / info['fps']
    
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-ss', str(timestamp),
        '-i', video_path,
        '-frames:v', '1',
        '-q:v', '2',
        output_path
    ]
    subprocess.run(cmd, check=True)
    
    return frame_number


# =============================================================================
# STABLE DIFFUSION PROCESSING
# =============================================================================

def upload_image(image_path):
    """Upload image to fal.ai and return URL."""
    url = fal_client.upload_file(image_path)
    return url


def process_with_sd(image_url, prompt, strength, guidance, seed=None, negative_prompt=None):
    """Process image through Stable Diffusion (SDXL)."""
    
    arguments = {
        "image_url": image_url,
        "prompt": prompt,
        "strength": strength,
        "guidance_scale": guidance,
        "num_inference_steps": 25,
    }
    
    if seed is not None:
        arguments["seed"] = seed
    
    if negative_prompt:
        arguments["negative_prompt"] = negative_prompt
    
    result = fal_client.subscribe(
        "fal-ai/fast-sdxl/image-to-image",
        arguments=arguments,
    )
    
    return result['images'][0]['url']


def correct_brightness(image_path, factor=1.1):
    """Slightly boost brightness to counteract darkening."""
    from PIL import ImageEnhance
    img = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(factor)
    img.save(image_path, 'PNG')


def download_image(url, output_path):
    """Download image and ensure it's saved as PNG."""
    temp_path = output_path + '.tmp'
    urllib.request.urlretrieve(url, temp_path)
    
    # Ensure PNG format
    img = Image.open(temp_path)
    img.save(output_path, 'PNG')
    os.remove(temp_path)


# =============================================================================
# PARAMETER INTERPOLATION
# =============================================================================

def interpolate(start, end, t):
    """Linear interpolation between start and end at position t (0-1)."""
    return start + (end - start) * t


def get_params_at_iteration(iteration, total_iterations, config):
    """Get interpolated parameters for a given iteration."""
    t = iteration / max(1, total_iterations - 1)
    
    strength = interpolate(
        config['strength_start'],
        config['strength_end'],
        t
    )
    
    guidance = interpolate(
        config['guidance_start'],
        config['guidance_end'],
        t
    )
    
    return strength, guidance


# =============================================================================
# VIDEO ASSEMBLY
# =============================================================================

def assemble_video(frames_dir, output_path, fps):
    """Assemble frames into video."""
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-framerate', str(fps),
        '-i', f'{frames_dir}/mutation_%05d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_path
    ]
    subprocess.run(cmd, check=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Mutation Chain - Exquisite Corpse Image Evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Basic mutation chain from video
  python mutation-chain.py -i video.mp4 -n 100 -p "dreamcast graphics"

  # Start from random frame with parameter drift
  python mutation-chain.py -i video.mp4 -n 150 --random-start \\
    --strength-start 0.3 --strength-end 0.7 \\
    -p "3d graphics 1997"

  # Start from existing image
  python mutation-chain.py -i image.png -n 50 -p "oil painting"
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input video or image file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output video path (auto-generated if not specified)')
    parser.add_argument('-n', '--iterations', type=int, default=100,
                        help='Number of mutation iterations (default: 100)')
    parser.add_argument('-p', '--prompt', default="abstract art",
                        help='Prompt for SD processing')
    
    # Starting frame options
    parser.add_argument('--random-start', action='store_true',
                        help='Start from a random frame instead of first')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Specific frame number to start from')
    
    # Parameter options (can drift over time)
    parser.add_argument('--strength-start', type=float, default=0.4,
                        help='Starting strength value (default: 0.4)')
    parser.add_argument('--strength-end', type=float, default=None,
                        help='Ending strength value (defaults to start value)')
    parser.add_argument('--guidance-start', type=float, default=6.0,
                        help='Starting guidance value (default: 6.0)')
    parser.add_argument('--guidance-end', type=float, default=None,
                        help='Ending guidance value (defaults to start value)')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (different seed each iteration if not set)')
    parser.add_argument('--negative-prompt', type=str, 
                        default="dark, black, dim, shadowy, muddy, blurry",
                        help='Negative prompt to avoid unwanted qualities')
    parser.add_argument('--brightness-correct', type=float, default=1.05,
                        help='Brightness boost per frame to counteract darkening (1.0=off, 1.05=subtle)')
    parser.add_argument('--fps', type=int, default=14,
                        help='Output video FPS (default: 14)')
    parser.add_argument('--rate-limit', type=float, default=0.5,
                        help='Delay between API calls in seconds')
    parser.add_argument('--keep-temp', action='store_true',
                        help='Keep temporary files')
    
    args = parser.parse_args()
    
    # Set defaults for end values
    if args.strength_end is None:
        args.strength_end = args.strength_start
    if args.guidance_end is None:
        args.guidance_end = args.guidance_start
    
    # Check API key
    if not os.getenv('FAL_KEY'):
        print("Error: FAL_KEY environment variable not set")
        sys.exit(1)
    
    # Generate output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"output/mutation_{timestamp}.mp4"
    
    # Setup
    work_dir = f"/tmp/mutation_chain_{int(time.time())}"
    frames_dir = os.path.join(work_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    print("=" * 60)
    print("MUTATION CHAIN")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Iterations: {args.iterations}")
    print(f"Prompt: {args.prompt}")
    print(f"Negative: {args.negative_prompt}")
    print(f"Strength: {args.strength_start} → {args.strength_end}")
    print(f"Guidance: {args.guidance_start} → {args.guidance_end}")
    print(f"Brightness correction: {args.brightness_correct}x per frame")
    print("=" * 60)
    
    config = {
        'strength_start': args.strength_start,
        'strength_end': args.strength_end,
        'guidance_start': args.guidance_start,
        'guidance_end': args.guidance_end,
    }
    
    try:
        # =====================================================================
        # STEP 1: Get starting frame
        # =====================================================================
        print("\n[1] Extracting starting frame...")
        
        input_path = Path(args.input)
        first_frame_path = os.path.join(work_dir, "start_frame.png")
        
        if input_path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            # Extract from video
            frame_num = extract_frame(
                args.input,
                first_frame_path,
                frame_number=args.start_frame,
                random_frame=args.random_start
            )
            print(f"  Extracted frame {frame_num}")
        else:
            # Assume it's an image
            img = Image.open(args.input)
            img.save(first_frame_path, 'PNG')
            print(f"  Using image directly")
        
        # =====================================================================
        # STEP 2: Run mutation chain
        # =====================================================================
        print(f"\n[2] Running {args.iterations} mutations...")
        
        current_image_path = first_frame_path
        
        for i in range(args.iterations):
            # Get interpolated parameters
            strength, guidance = get_params_at_iteration(i, args.iterations, config)
            
            # Use consistent seed if specified, otherwise random
            seed = args.seed if args.seed else random.randint(0, 2**32 - 1)
            
            # Upload current image
            image_url = upload_image(current_image_path)
            
            # Process through SD
            result_url = process_with_sd(
                image_url,
                args.prompt,
                strength,
                guidance,
                seed,
                args.negative_prompt
            )
            
            # Download result
            output_path = os.path.join(frames_dir, f"mutation_{i:05d}.png")
            download_image(result_url, output_path)
            
            # Brightness correction to counteract darkening
            if args.brightness_correct > 1.0:
                correct_brightness(output_path, args.brightness_correct)
            
            # This output becomes the input for next iteration
            current_image_path = output_path
            
            # Progress
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{args.iterations}] s={strength:.2f} g={guidance:.1f}")
            
            # Rate limiting
            if args.rate_limit > 0:
                time.sleep(args.rate_limit)
        
        # =====================================================================
        # STEP 3: Assemble video
        # =====================================================================
        print(f"\n[3] Assembling video at {args.fps} FPS...")
        
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        assemble_video(frames_dir, args.output, args.fps)
        
        print("\n" + "=" * 60)
        print("✓ COMPLETE!")
        print("=" * 60)
        print(f"Output: {args.output}")
        print(f"Frames: {args.iterations}")
        print(f"Duration: {args.iterations / args.fps:.1f}s at {args.fps} FPS")
        print("=" * 60)
        
    finally:
        if not args.keep_temp and os.path.exists(work_dir):
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()

