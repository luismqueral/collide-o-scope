"""
style-transfer-video.py - Transfer style from one video to another

Takes two videos:
- CONTENT video: provides structure, motion, composition
- STYLE video: provides colors, textures, visual aesthetic

Uses IP-Adapter to blend the style of one video onto the content of another.

Usage:
  python style-transfer-video.py  # random content + random style
  python style-transfer-video.py --content video1.mp4 --style video2.mp4
  python style-transfer-video.py --style-strength 0.8  # stronger style transfer

EXAMPLES:
  # Subtle style influence
  python style-transfer-video.py --style-strength 0.4 --content-strength 0.3

  # Heavy stylization
  python style-transfer-video.py --style-strength 0.9 --content-strength 0.5

  # Quick test
  python style-transfer-video.py --duration 2 --fps 6
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
    print("Install with: pip install pillow numpy")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULTS = {
    'fps': 12,
    'duration': 10,
    'width': 768,
    'height': 768,
    
    # Style transfer settings
    'style_strength': 0.6,      # How much style to apply (0-1)
    'content_strength': 0.4,    # How much to transform content (0-1, lower = keep more structure)
    'prompt': '',               # Optional text prompt to guide generation
    
    # Model selection
    'model': 'photomaker-style',  # photomaker-style (two images), sdxl-img2img (prompt-based)
    
    'output_dir': 'output',
}

# Available models for style transfer
MODELS = {
    'photomaker-style': {
        # PhotoMaker style transfer - uses reference image for style (Replicate)
        'id': 'tencentarc/photomaker-style:467d062309da518648ba89d226490e02b8ed09b5abc15026e54e31c5a8cd0769',
        'backend': 'replicate',
        'supports_style_image': True,
    },
    'fal-ip-adapter': {
        # IP-Adapter on fal.ai - style transfer with reference image
        'id': 'fal-ai/ip-adapter-face-id',
        'backend': 'fal',
        'supports_style_image': True,
    },
    'fal-style-transfer': {
        # Fast style transfer on fal.ai
        'id': 'fal-ai/fast-sdxl/image-to-image',
        'backend': 'fal',
        'supports_style_image': False,
    },
    'sdxl-img2img': {
        'id': 'stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc',
        'backend': 'replicate',
        'supports_style_image': False,
    },
}


# =============================================================================
# REPLICATE API
# =============================================================================

def encode_image(image_path):
    """Encode image to base64 data URI."""
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/png;base64,{image_data}"


def process_with_photomaker_style(content_path, style_path, prompt, style_strength, content_strength, seed, width, height):
    """Process using PhotoMaker style transfer - transfers style from one image to another."""
    try:
        import replicate
    except ImportError:
        print("Error: replicate not installed")
        sys.exit(1)
    
    content_uri = encode_image(content_path)
    style_uri = encode_image(style_path)
    
    model_id = MODELS['photomaker-style']['id']
    
    # PhotoMaker style uses the style image as reference
    style_prompt = prompt if prompt else "in the style of img, artistic, stylized"
    
    input_params = {
        "input_image": content_uri,     # Content/structure image
        "input_image2": style_uri,      # Style reference image  
        "prompt": style_prompt,
        "negative_prompt": "blurry, low quality, watermark, text",
        "style_strength_ratio": int(style_strength * 50),  # 0-50 (model max)
        "num_steps": 30,
        "guidance_scale": 7.5,
        "width": width,
        "height": height,
    }
    
    if seed is not None:
        input_params["seed"] = seed
    
    output = replicate.run(model_id, input=input_params)
    
    if isinstance(output, list):
        return output[0] if output else None
    return output


def analyze_style_colors(style_path):
    """Extract dominant colors from style image to use in prompt."""
    img = Image.open(style_path).convert('RGB')
    img = img.resize((100, 100))  # Small for fast analysis
    pixels = np.array(img).reshape(-1, 3)
    
    # Get average color
    avg_color = pixels.mean(axis=0).astype(int)
    
    # Simple brightness check
    brightness = avg_color.mean()
    
    # Color descriptors based on RGB analysis
    descriptors = []
    
    if brightness > 170:
        descriptors.append("bright")
    elif brightness < 85:
        descriptors.append("dark")
    
    # Check for color dominance
    r, g, b = avg_color
    if r > g + 30 and r > b + 30:
        descriptors.append("warm tones, red and orange hues")
    elif b > r + 30 and b > g + 30:
        descriptors.append("cool tones, blue hues")
    elif g > r + 30 and g > b + 30:
        descriptors.append("natural greens")
    elif abs(r - g) < 20 and abs(g - b) < 20:
        descriptors.append("neutral tones")
    
    # Check saturation (rough estimate)
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    if max_c - min_c > 100:
        descriptors.append("vibrant saturated colors")
    elif max_c - min_c < 50:
        descriptors.append("muted desaturated palette")
    
    return ", ".join(descriptors) if descriptors else "artistic colors"


def process_with_img2img_style(content_path, style_path, prompt, style_strength, content_strength, seed, width, height):
    """
    Use img2img with style colors extracted from style image.
    Analyzes the style image and builds a color-aware prompt.
    """
    try:
        import replicate
    except ImportError:
        print("Error: replicate not installed")
        sys.exit(1)
    
    content_uri = encode_image(content_path)
    
    # Analyze style image for colors
    style_colors = analyze_style_colors(style_path)
    
    # Build prompt combining user prompt with extracted style
    if prompt:
        style_prompt = f"{prompt}, {style_colors}"
    else:
        style_prompt = f"artistic, {style_colors}, painterly texture, stylized"
    
    model_id = MODELS['img2img']['id']
    
    input_params = {
        "image": content_uri,
        "prompt": style_prompt,
        "negative_prompt": "blurry, low quality, watermark, text",
        "prompt_strength": content_strength,
        "guidance_scale": 7.5 + (style_strength * 5),  # Higher guidance for stronger style
        "num_inference_steps": 30,
        "width": width,
        "height": height,
    }
    
    if seed is not None:
        input_params["seed"] = seed
    
    output = replicate.run(model_id, input=input_params)
    
    if isinstance(output, list):
        return output[0] if output else None
    return output


def download_image(url, output_path):
    """Download image from URL and ensure it's saved as PNG."""
    import tempfile
    
    # Download to temp file first
    temp_path = output_path + '.tmp'
    urllib.request.urlretrieve(url, temp_path)
    
    # Convert to PNG if needed (handles JPEG from fal.ai)
    try:
        img = Image.open(temp_path)
        img.save(output_path, 'PNG')
        os.remove(temp_path)
    except Exception as e:
        # If conversion fails, just rename
        os.rename(temp_path, output_path)


def process_with_fal(content_path, style_path, prompt, style_strength, content_strength, seed, width, height):
    """Process using fal.ai IP-Adapter for style transfer."""
    try:
        import fal_client
    except ImportError:
        print("Error: fal-client not installed. Run: pip install fal-client")
        sys.exit(1)
    
    import os
    if not os.environ.get('FAL_KEY'):
        print("Error: FAL_KEY not set in environment")
        sys.exit(1)
    
    content_uri = encode_image(content_path)
    style_uri = encode_image(style_path)
    
    # Build prompt
    style_prompt = prompt if prompt else "artistic, stylized, high quality"
    
    # Use fal's IP-Adapter model
    result = fal_client.subscribe(
        "fal-ai/ip-adapter-face-id",
        arguments={
            "image_url": content_uri,
            "face_image_url": style_uri,  # Using style as reference
            "prompt": style_prompt,
            "negative_prompt": "blurry, low quality",
            "ip_adapter_scale": style_strength,
            "strength": content_strength,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "seed": seed if seed else None,
        }
    )
    
    if result and 'image' in result:
        return result['image']['url']
    elif result and 'images' in result and len(result['images']) > 0:
        return result['images'][0]['url']
    return None


def process_with_fal_img2img(content_path, style_path, prompt, style_strength, content_strength, seed, width, height):
    """Process using fal.ai fast SDXL img2img with style colors in prompt."""
    try:
        import fal_client
    except ImportError:
        print("Error: fal-client not installed")
        sys.exit(1)
    
    content_uri = encode_image(content_path)
    
    # Analyze style colors and build prompt
    style_colors = analyze_style_colors(style_path)
    style_prompt = f"{prompt}, {style_colors}" if prompt else f"artistic, {style_colors}"
    
    result = fal_client.subscribe(
        "fal-ai/fast-sdxl/image-to-image",
        arguments={
            "image_url": content_uri,
            "prompt": style_prompt,
            "negative_prompt": "blurry, low quality, watermark",
            "strength": content_strength,
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "seed": seed if seed else None,
        }
    )
    
    if result and 'images' in result and len(result['images']) > 0:
        return result['images'][0]['url']
    return None


# =============================================================================
# VIDEO PROCESSING
# =============================================================================

def get_random_video(folder='library/video', exclude=None):
    """Get a random video, optionally excluding one."""
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


def extract_style_frames(video_path, output_dir, num_frames, width, height):
    """
    Extract evenly-spaced frames from style video.
    These will be cycled through during processing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video duration
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ], capture_output=True, text=True)
    
    duration = float(result.stdout.strip())
    
    # Extract evenly spaced frames
    interval = duration / num_frames
    
    for i in range(num_frames):
        timestamp = i * interval
        output_path = os.path.join(output_dir, f'style_{i:03d}.png')
        
        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-ss', str(timestamp),
            '-i', video_path,
            '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
            '-frames:v', '1',
            output_path
        ]
        subprocess.run(cmd, check=True)
    
    return sorted([f for f in os.listdir(output_dir) if f.startswith('style_')])


def reassemble_video(frames_dir, output_path, fps, codec='libx264', crf=18):
    """Reassemble frames into video."""
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-framerate', str(fps),
        '-i', f'{frames_dir}/frame_%05d.png',
        '-c:v', codec,
        '-pix_fmt', 'yuv420p',
        '-crf', str(crf),
        output_path
    ]
    subprocess.run(cmd, check=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Transfer visual style from one video to another',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Random videos, default settings
  python style-transfer-video.py
  
  # Specific videos
  python style-transfer-video.py --content video1.mp4 --style video2.mp4
  
  # Heavy style transfer
  python style-transfer-video.py --style-strength 0.9
  
  # Preserve more content structure
  python style-transfer-video.py --content-strength 0.2
  
  # Quick test
  python style-transfer-video.py --duration 2 --fps 6
        """
    )
    
    # Input
    parser.add_argument('--content', type=str, help='Content video (structure/motion)')
    parser.add_argument('--style', type=str, help='Style video (colors/textures)')
    
    # Style transfer settings
    parser.add_argument('--style-strength', type=float, default=DEFAULTS['style_strength'],
                        help=f"How much style to apply 0-1 (default: {DEFAULTS['style_strength']})")
    parser.add_argument('--content-strength', type=float, default=DEFAULTS['content_strength'],
                        help=f"How much to transform 0-1, lower=keep structure (default: {DEFAULTS['content_strength']})")
    parser.add_argument('-p', '--prompt', type=str, default='',
                        help='Optional text prompt to guide generation')
    
    # Video settings
    parser.add_argument('--fps', type=int, default=DEFAULTS['fps'],
                        help=f"Frames per second (default: {DEFAULTS['fps']})")
    parser.add_argument('--duration', type=float, default=DEFAULTS['duration'],
                        help=f"Duration in seconds (default: {DEFAULTS['duration']})")
    parser.add_argument('--width', type=int, default=DEFAULTS['width'],
                        help=f"Output width (default: {DEFAULTS['width']})")
    parser.add_argument('--height', type=int, default=DEFAULTS['height'],
                        help=f"Output height (default: {DEFAULTS['height']})")
    
    # Processing
    parser.add_argument('--model', type=str, default='photomaker-style',
                        choices=['photomaker-style', 'fal-ip-adapter', 'fal-style-transfer', 'sdxl-img2img'],
                        help='Model to use (default: photomaker-style)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for consistency')
    parser.add_argument('--style-frames', type=int, default=5,
                        help='Number of style frames to extract and cycle through')
    parser.add_argument('--keep-temp', action='store_true',
                        help='Keep temporary files')
    parser.add_argument('--rate-limit', type=float, default=10,
                        help='Seconds between API calls (default: 10)')
    
    # Output
    parser.add_argument('-o', '--output-dir', type=str, default=DEFAULTS['output_dir'],
                        help=f"Output directory (default: {DEFAULTS['output_dir']})")
    
    args = parser.parse_args()
    
    # Get videos
    if args.content:
        content_video = args.content
    else:
        content_video = get_random_video()
        if not content_video:
            print("Error: No content video found")
            sys.exit(1)
    
    if args.style:
        style_video = args.style
    else:
        style_video = get_random_video(exclude=content_video)
        if not style_video:
            print("Error: No style video found")
            sys.exit(1)
    
    # Calculate totals
    total_frames = int(args.fps * args.duration)
    estimated_cost = total_frames * 0.005  # IP-Adapter is ~SDXL pricing
    estimated_time = total_frames * 8 / 60  # ~8 sec per frame
    
    print("=" * 60)
    print("STYLE TRANSFER VIDEO")
    print("=" * 60)
    print(f"Content: {os.path.basename(content_video)}")
    print(f"Style:   {os.path.basename(style_video)}")
    print("-" * 60)
    print(f"Style strength: {args.style_strength} (higher = more style)")
    print(f"Content strength: {args.content_strength} (higher = more change)")
    print(f"Model: {args.model}")
    print("-" * 60)
    print(f"Duration: {args.duration}s at {args.fps} FPS ({total_frames} frames)")
    print(f"Size: {args.width}x{args.height}")
    print(f"Estimated cost: ~${estimated_cost:.2f}")
    print(f"Estimated time: ~{estimated_time:.1f} minutes")
    print("=" * 60)
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = f"style_transfer_work_{timestamp}"
    content_frames_dir = os.path.join(work_dir, "content_frames")
    style_frames_dir = os.path.join(work_dir, "style_frames")
    output_frames_dir = os.path.join(work_dir, "output_frames")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)
    
    # Use consistent seed (capped for model compatibility)
    seed = args.seed if args.seed else random.randint(0, 2**31 - 1)
    print(f"\nUsing seed: {seed}")
    
    try:
        # Step 1: Extract content frames
        print(f"\n[1/4] Extracting content frames...")
        content_frames = extract_frames(
            content_video, content_frames_dir,
            args.fps, args.duration,
            args.width, args.height
        )
        print(f"    Extracted {len(content_frames)} content frames")
        
        # Step 2: Extract style frames
        print(f"\n[2/4] Extracting style reference frames...")
        style_frames = extract_style_frames(
            style_video, style_frames_dir,
            args.style_frames,
            args.width, args.height
        )
        print(f"    Extracted {len(style_frames)} style frames")
        
        # Step 3: Process each frame
        print(f"\n[3/4] Processing frames with style transfer...")
        print(f"    Style strength: {args.style_strength}")
        print(f"    Content strength: {args.content_strength}")
        
        start_time = time.time()
        processed = 0
        failed = 0
        
        for i, content_frame in enumerate(content_frames):
            content_path = os.path.join(content_frames_dir, content_frame)
            output_path = os.path.join(output_frames_dir, content_frame)
            
            # Cycle through style frames
            style_idx = i % len(style_frames)
            style_path = os.path.join(style_frames_dir, style_frames[style_idx])
            
            try:
                if args.model == 'photomaker-style':
                    result = process_with_photomaker_style(
                        content_path, style_path,
                        args.prompt,
                        args.style_strength,
                        args.content_strength,
                        seed,
                        args.width, args.height
                    )
                elif args.model == 'fal-ip-adapter':
                    result = process_with_fal(
                        content_path, style_path,
                        args.prompt,
                        args.style_strength,
                        args.content_strength,
                        seed,
                        args.width, args.height
                    )
                elif args.model == 'fal-style-transfer':
                    result = process_with_fal_img2img(
                        content_path, style_path,
                        args.prompt,
                        args.style_strength,
                        args.content_strength,
                        seed,
                        args.width, args.height
                    )
                else:
                    result = process_with_img2img_style(
                        content_path, style_path,
                        args.prompt,
                        args.style_strength,
                        args.content_strength,
                        seed,
                        args.width, args.height
                    )
                
                if result:
                    download_image(str(result), output_path)
                    processed += 1
                else:
                    shutil.copy(content_path, output_path)
                    failed += 1
                    
            except Exception as e:
                print(f"\n    Error on frame {i}: {e}")
                shutil.copy(content_path, output_path)
                failed += 1
            
            # Progress
            elapsed = time.time() - start_time
            per_frame = elapsed / (i + 1)
            eta = per_frame * (len(content_frames) - i - 1)
            progress = (i + 1) / len(content_frames) * 100
            print(f"    [{i+1}/{len(content_frames)}] {progress:.0f}% - {per_frame:.1f}s/frame - ETA: {eta:.0f}s", end='\r')
            
            # Rate limiting
            if args.rate_limit > 0 and i < len(content_frames) - 1:
                time.sleep(args.rate_limit)
        
        print(f"\n    Done! Processed: {processed}, Failed: {failed}")
        
        # Step 4: Reassemble video
        print(f"\n[4/4] Assembling video...")
        
        output_name = f"style_transfer_{timestamp}.mp4"
        output_path = os.path.join(args.output_dir, output_name)
        
        reassemble_video(output_frames_dir, output_path, args.fps)
        
        # Save metadata
        metadata = {
            'content_video': content_video,
            'style_video': style_video,
            'output': output_path,
            'timestamp': timestamp,
            'settings': {
                'style_strength': args.style_strength,
                'content_strength': args.content_strength,
                'prompt': args.prompt,
                'model': args.model,
                'seed': seed,
                'fps': args.fps,
                'duration': args.duration,
            }
        }
        
        metadata_path = os.path.join(args.output_dir, f"style_transfer_{timestamp}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✓ COMPLETE!")
        print("=" * 60)
        print(f"Output: {output_path}")
        print(f"Metadata: {metadata_path}")
        print("=" * 60)
        
    finally:
        if not args.keep_temp and os.path.exists(work_dir):
            print("\nCleaning up temporary files...")
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()

