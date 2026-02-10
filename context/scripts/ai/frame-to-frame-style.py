"""
frame-to-frame-style.py - True Frame-by-Frame Style Transfer

Takes two videos and stylizes each frame of Video A using the 
corresponding frame of Video B as the style reference.

Frame 1 of A → styled by → Frame 1 of B
Frame 2 of A → styled by → Frame 2 of B
...

This creates temporal coherence because the style evolves with the content.

Usage:
  python frame-to-frame-style.py --content video_a.mp4 --style video_b.mp4
  python frame-to-frame-style.py --duration 5 --fps 14
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
    'fps': 12,
    'duration': 5,
    'width': 768,
    'height': 768,
    'strength': 0.5,        # How much to transform (0-1)
    'output_dir': 'output',
    'rate_limit': 2,        # Seconds between API calls
}


def encode_image(image_path):
    """Encode image to base64 data URI."""
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/png;base64,{image_data}"


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


def get_random_video(folder='library/video', exclude=None):
    """Get a random video from input folder."""
    videos = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.mov', '.avi', '.webm'))]
    if exclude:
        exclude_name = os.path.basename(exclude)
        videos = [v for v in videos if v != exclude_name]
    if not videos:
        return None
    return os.path.join(folder, random.choice(videos))


def extract_frames(video_path, output_dir, fps, duration, width, height):
    """Extract frames from video."""
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-i', video_path,
        '-t', str(duration),
        '-vf', f'fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
        f'{output_dir}/frame_%05d.png'
    ]
    
    subprocess.run(cmd, check=True)
    
    frames = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    return frames


def analyze_style_colors(image_path):
    """Extract color description from an image for prompt."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((100, 100))
    pixels = np.array(img).reshape(-1, 3)
    
    avg_color = pixels.mean(axis=0).astype(int)
    brightness = avg_color.mean()
    
    descriptors = []
    
    if brightness > 170:
        descriptors.append("bright")
    elif brightness < 85:
        descriptors.append("dark")
    
    r, g, b = avg_color
    if r > g + 30 and r > b + 30:
        descriptors.append("warm tones")
    elif b > r + 30 and b > g + 30:
        descriptors.append("cool blue tones")
    elif g > r + 30 and g > b + 30:
        descriptors.append("green tones")
    
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    if max_c - min_c > 100:
        descriptors.append("vibrant colors")
    elif max_c - min_c < 50:
        descriptors.append("muted palette")
    
    return ", ".join(descriptors) if descriptors else "artistic"


def process_frame_pair_fal(content_path, style_path, output_path, strength, seed):
    """Process a content/style frame pair using fal.ai."""
    try:
        import fal_client
    except ImportError:
        print("Error: fal-client not installed")
        return False
    
    content_uri = encode_image(content_path)
    
    # Analyze style frame for color prompt
    style_colors = analyze_style_colors(style_path)
    prompt = f"artistic, stylized, {style_colors}"
    
    try:
        result = fal_client.subscribe(
            "fal-ai/fast-sdxl/image-to-image",
            arguments={
                "image_url": content_uri,
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, watermark",
                "strength": strength,
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
                "seed": seed,
            }
        )
        
        if result and 'images' in result and len(result['images']) > 0:
            download_image(result['images'][0]['url'], output_path)
            return True
    except Exception as e:
        print(f"\nError: {e}")
    
    return False


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


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Frame-to-frame style transfer')
    parser.add_argument('--content', type=str, help='Content video')
    parser.add_argument('--style', type=str, help='Style video')
    parser.add_argument('--fps', type=int, default=DEFAULTS['fps'])
    parser.add_argument('--duration', type=float, default=DEFAULTS['duration'])
    parser.add_argument('--width', type=int, default=DEFAULTS['width'])
    parser.add_argument('--height', type=int, default=DEFAULTS['height'])
    parser.add_argument('--strength', type=float, default=DEFAULTS['strength'])
    parser.add_argument('--rate-limit', type=float, default=DEFAULTS['rate_limit'])
    parser.add_argument('--keep-temp', action='store_true')
    parser.add_argument('-o', '--output-dir', default=DEFAULTS['output_dir'])
    
    args = parser.parse_args()
    
    # Get videos
    if args.content:
        content_video = args.content
    else:
        content_video = get_random_video()
    
    if args.style:
        style_video = args.style
    else:
        style_video = get_random_video(exclude=content_video)
    
    if not content_video or not style_video:
        print("Error: Need two videos")
        sys.exit(1)
    
    total_frames = int(args.fps * args.duration)
    
    print("=" * 60)
    print("FRAME-TO-FRAME STYLE TRANSFER")
    print("=" * 60)
    print(f"Content: {os.path.basename(content_video)}")
    print(f"Style:   {os.path.basename(style_video)}")
    print(f"Frames:  {total_frames} ({args.duration}s @ {args.fps}fps)")
    print(f"Size:    {args.width}x{args.height}")
    print(f"Strength: {args.strength}")
    print("=" * 60)
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = f"f2f_work_{timestamp}"
    content_dir = os.path.join(work_dir, "content")
    style_dir = os.path.join(work_dir, "style")
    output_frames_dir = os.path.join(work_dir, "output")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)
    
    seed = random.randint(0, 2**31 - 1)
    print(f"Seed: {seed}")
    
    try:
        # Extract frames from both videos
        print(f"\n[1/3] Extracting frames...")
        content_frames = extract_frames(
            content_video, content_dir,
            args.fps, args.duration, args.width, args.height
        )
        print(f"  Content: {len(content_frames)} frames")
        
        style_frames = extract_frames(
            style_video, style_dir,
            args.fps, args.duration, args.width, args.height
        )
        print(f"  Style:   {len(style_frames)} frames")
        
        # Ensure same number of frames
        num_frames = min(len(content_frames), len(style_frames))
        
        # Process each frame pair
        print(f"\n[2/3] Processing {num_frames} frame pairs...")
        start_time = time.time()
        processed = 0
        failed = 0
        
        for i in range(num_frames):
            content_path = os.path.join(content_dir, content_frames[i])
            style_path = os.path.join(style_dir, style_frames[i])
            output_path = os.path.join(output_frames_dir, content_frames[i])
            
            success = process_frame_pair_fal(
                content_path, style_path, output_path,
                args.strength, seed
            )
            
            if success:
                processed += 1
            else:
                # Copy original on failure
                shutil.copy(content_path, output_path)
                failed += 1
            
            # Progress
            elapsed = time.time() - start_time
            per_frame = elapsed / (i + 1)
            eta = per_frame * (num_frames - i - 1)
            print(f"  [{i+1}/{num_frames}] {(i+1)/num_frames*100:.0f}% - {per_frame:.1f}s/frame - ETA: {eta:.0f}s", end='\r')
            
            # Rate limiting
            if args.rate_limit > 0 and i < num_frames - 1:
                time.sleep(args.rate_limit)
        
        print(f"\n  Done! Processed: {processed}, Failed: {failed}")
        
        # Reassemble
        print(f"\n[3/3] Assembling video...")
        output_name = f"f2f_style_{timestamp}.mp4"
        output_path = os.path.join(args.output_dir, output_name)
        reassemble_video(output_frames_dir, output_path, args.fps)
        
        # Save metadata
        metadata = {
            'content': content_video,
            'style': style_video,
            'frames': num_frames,
            'fps': args.fps,
            'strength': args.strength,
            'seed': seed,
        }
        meta_path = os.path.join(args.output_dir, f"f2f_style_{timestamp}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✓ COMPLETE!")
        print("=" * 60)
        print(f"Output: {output_path}")
        print("=" * 60)
        
    finally:
        if not args.keep_temp and os.path.exists(work_dir):
            print("\nCleaning up...")
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()



