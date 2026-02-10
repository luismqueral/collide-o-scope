"""
generative-frame-processor.py - AI-Powered Frame-by-Frame Video Processing

Takes a video, extracts frames, processes each through a generative AI model,
and reassembles into a new video. Uses cloud APIs so no local GPU needed.

SUPPORTED BACKENDS:
- replicate: Uses Replicate.com API (recommended, easy setup)
- fal: Uses fal.ai API (very fast)
- local: Uses local diffusers (requires GPU)

SETUP:
1. pip install replicate pillow numpy
2. export REPLICATE_API_TOKEN="your_token_here"
   (get token at https://replicate.com/account/api-tokens)

USAGE:
python scripts/ai/generative-frame-processor.py --input library/video/my_video.mp4 --prompt "cyberpunk neon city"
python scripts/ai/generative-frame-processor.py --help  # see all options

EXAMPLE WORKFLOWS:
# Light stylization (preserve structure)
python generative-frame-processor.py -i video.mp4 -p "oil painting" --strength 0.4

# Heavy transformation
python generative-frame-processor.py -i video.mp4 -p "anime style" --strength 0.7 --model sdxl

# Fast test run
python generative-frame-processor.py -i video.mp4 -p "watercolor" --duration 3 --fps 8
"""

import os
import sys
import argparse
import subprocess
import shutil
import time
import json
from datetime import datetime
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use environment variables directly

# Check for required packages
try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: PIL or numpy not installed")
    print("Install with: pip install pillow numpy")
    sys.exit(1)

# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================

DEFAULTS = {
    # Video extraction settings
    'fps': 12,                    # Frames per second to extract
    'duration': 10,               # Seconds of video to process
    'width': None,                # Output width (None = match source)
    'height': None,               # Output height (None = match source)
    'auto_size': True,            # Auto-detect size from source video
    
    # Generation settings
    'prompt': 'abstract art, colorful, psychedelic',
    'negative_prompt': 'blurry, low quality, watermark, text',
    'strength': 0.5,              # How much to change (0.0-1.0, higher = more change)
    'guidance_scale': 7.5,        # How closely to follow prompt (1-20)
    'seed': None,                 # Random seed (None = random, set for consistency)
    
    # Backend settings
    'backend': 'replicate',       # replicate, fal, or local
    'model': 'sd15',              # sd15, sdxl, or flux
    
    # Processing settings
    'batch_size': 1,              # Process N frames at once (for APIs that support it)
    'skip_existing': True,        # Skip frames that already exist
    'keep_temp': False,           # Keep temporary files for debugging
    'rate_limit_delay': 10,       # Seconds between API calls (avoid 429 errors)
    
    # Output settings
    'output_dir': 'output',
    'video_codec': 'libx264',
    'video_quality': 18,          # CRF value (lower = better quality, 18-23 recommended)
}

# Model configurations for different backends
MODELS = {
    'replicate': {
        'sd15': 'stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf',
        'sdxl': 'stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc',
        'flux': 'black-forest-labs/flux-1.1-pro',
        'flux-schnell': 'black-forest-labs/flux-schnell',
    },
    'fal': {
        'sd15': 'fal-ai/stable-diffusion-v15/image-to-image',
        'sdxl': 'fal-ai/fast-sdxl/image-to-image',
        'flux': 'fal-ai/flux-pro/v1.1',
    }
}


# =============================================================================
# BACKEND IMPLEMENTATIONS
# =============================================================================

class ReplicateBackend:
    """Process frames using Replicate.com API."""
    
    def __init__(self, model_key='sdxl'):
        try:
            import replicate
            self.replicate = replicate
        except ImportError:
            print("Error: replicate not installed")
            print("Install with: pip install replicate")
            print("Then set: export REPLICATE_API_TOKEN='your_token'")
            sys.exit(1)
        
        if not os.environ.get('REPLICATE_API_TOKEN'):
            print("Error: REPLICATE_API_TOKEN not set")
            print("Get your token at: https://replicate.com/account/api-tokens")
            print("Then run: export REPLICATE_API_TOKEN='your_token'")
            sys.exit(1)
        
        self.model_id = MODELS['replicate'].get(model_key, MODELS['replicate']['sdxl'])
        self.model_key = model_key
        print(f"Using Replicate model: {model_key} ({self.model_id})")
    
    def process_frame(self, image_path, prompt, negative_prompt, strength, guidance_scale, seed, width, height):
        """Process a single frame through Replicate API."""
        import base64
        
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        image_uri = f"data:image/png;base64,{image_data}"
        
        # Build input based on model type
        if 'flux' in self.model_key:
            # FLUX models have different parameters
            input_params = {
                "prompt": prompt,
                "image": image_uri,
                "strength": strength,
                "num_inference_steps": 28,
                "guidance": guidance_scale,
            }
            if seed is not None:
                input_params["seed"] = seed
        else:
            # SD/SDXL models
            input_params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image_uri,
                "prompt_strength": strength,  # Note: called prompt_strength in replicate
                "guidance_scale": guidance_scale,
                "num_inference_steps": 30,
                "width": width,
                "height": height,
            }
            if seed is not None:
                input_params["seed"] = seed
        
        # Run the model
        try:
            output = self.replicate.run(self.model_id, input=input_params)
            
            # Handle different output formats
            if isinstance(output, list):
                return output[0] if output else None
            return output
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None


class FalBackend:
    """Process frames using fal.ai API."""
    
    def __init__(self, model_key='sdxl'):
        try:
            import fal_client
            self.fal = fal_client
        except ImportError:
            print("Error: fal_client not installed")
            print("Install with: pip install fal-client")
            print("Then set: export FAL_KEY='your_key'")
            sys.exit(1)
        
        if not os.environ.get('FAL_KEY'):
            print("Error: FAL_KEY not set")
            print("Get your key at: https://fal.ai/dashboard/keys")
            sys.exit(1)
        
        self.model_id = MODELS['fal'].get(model_key, MODELS['fal']['sdxl'])
        print(f"Using fal.ai model: {model_key}")
    
    def process_frame(self, image_path, prompt, negative_prompt, strength, guidance_scale, seed, width, height):
        """Process a single frame through fal.ai API."""
        import base64
        
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        try:
            result = self.fal.subscribe(
                self.model_id,
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
            elif result and 'image' in result:
                return result['image']['url'] if isinstance(result['image'], dict) else result['image']
            else:
                print(f"\nUnexpected fal response: {result}")
                return None
        except Exception as e:
            print(f"\nFal API error: {e}")
            return None


class LocalBackend:
    """Process frames using local diffusers (requires GPU)."""
    
    def __init__(self, model_key='sdxl'):
        try:
            import torch
            from diffusers import AutoPipelineForImage2Image
            
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            
            if self.device == "cpu":
                print("Warning: No GPU detected, processing will be VERY slow")
            
            model_ids = {
                'sd15': 'runwayml/stable-diffusion-v1-5',
                'sdxl': 'stabilityai/stable-diffusion-xl-base-1.0',
            }
            
            model_id = model_ids.get(model_key, model_ids['sd15'])
            print(f"Loading {model_id} on {self.device}...")
            
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            ).to(self.device)
            
        except ImportError:
            print("Error: diffusers or torch not installed")
            print("Install with: pip install diffusers torch accelerate")
            sys.exit(1)
    
    def process_frame(self, image_path, prompt, negative_prompt, strength, guidance_scale, seed, width, height):
        """Process a single frame using local model."""
        image = Image.open(image_path).convert('RGB').resize((width, height))
        
        generator = None
        if seed is not None:
            generator = self.torch.Generator(device=self.device).manual_seed(seed)
        
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        return result


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def get_video_dimensions(video_path):
    """Get width and height of video."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split(',')
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    return 768, 768  # Default fallback


def extract_frames(video_path, output_dir, fps, duration, width, height):
    """Extract frames from video using ffmpeg."""
    print(f"\n[1/4] Extracting frames at {fps} FPS...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build scale filter - if width/height specified, scale to that; otherwise keep original
    if width and height:
        scale_filter = f'fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2'
    else:
        scale_filter = f'fps={fps}'
    
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-i', video_path,
        '-t', str(duration),
        '-vf', scale_filter,
        f'{output_dir}/frame_%05d.png'
    ]
    
    subprocess.run(cmd, check=True)
    
    frames = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"    Extracted {len(frames)} frames")
    return frames


def download_image(url, output_path):
    """Download image from URL and ensure PNG format."""
    import urllib.request
    temp_path = output_path + '.tmp'
    urllib.request.urlretrieve(url, temp_path)
    
    # Convert to PNG if needed (fal.ai returns JPEGs)
    try:
        img = Image.open(temp_path)
        img.save(output_path, 'PNG')
        os.remove(temp_path)
    except Exception:
        # If conversion fails, just rename
        os.rename(temp_path, output_path)


def process_frames(backend, frames_dir, output_dir, frame_files, args):
    """Process all frames through the generative model."""
    print(f"\n[2/4] Processing {len(frame_files)} frames with {args.backend}/{args.model}...")
    print(f"    Prompt: '{args.prompt}'")
    print(f"    Strength: {args.strength}, Guidance: {args.guidance_scale}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use consistent seed for temporal coherence if specified
    base_seed = args.seed if args.seed is not None else None
    if base_seed is None and args.consistent_seed:
        import random
        base_seed = random.randint(0, 2**32 - 1)
        print(f"    Using consistent seed: {base_seed}")
    
    processed_count = 0
    failed_count = 0
    start_time = time.time()
    
    for i, frame_file in enumerate(frame_files):
        input_path = os.path.join(frames_dir, frame_file)
        output_path = os.path.join(output_dir, frame_file)
        
        # Skip if already processed
        if args.skip_existing and os.path.exists(output_path):
            print(f"    Skipping {frame_file} (exists)")
            continue
        
        # Use same seed for all frames if consistent_seed is set
        frame_seed = base_seed
        
        try:
            result = backend.process_frame(
                input_path,
                args.prompt,
                args.negative_prompt,
                args.strength,
                args.guidance_scale,
                frame_seed,
                args.width,
                args.height
            )
            
            if result:
                # Handle different return types
                if isinstance(result, str) and result.startswith('http'):
                    download_image(result, output_path)
                elif isinstance(result, Image.Image):
                    result.save(output_path)
                elif hasattr(result, 'read'):
                    # File-like object
                    with open(output_path, 'wb') as f:
                        f.write(result.read())
                else:
                    # Try to download as URL
                    download_image(str(result), output_path)
                
                processed_count += 1
            else:
                # Copy original if processing failed
                shutil.copy(input_path, output_path)
                failed_count += 1
                
        except Exception as e:
            print(f"\n    Error on {frame_file}: {e}")
            shutil.copy(input_path, output_path)
            failed_count += 1
        
        # Progress
        elapsed = time.time() - start_time
        per_frame = elapsed / (i + 1)
        eta = per_frame * (len(frame_files) - i - 1)
        progress = (i + 1) / len(frame_files) * 100
        print(f"    [{i+1}/{len(frame_files)}] {progress:.0f}% - {per_frame:.1f}s/frame - ETA: {eta:.0f}s", end='\r')
        
        # Rate limiting - wait between API calls
        if hasattr(args, 'rate_limit') and args.rate_limit > 0 and i < len(frame_files) - 1:
            time.sleep(args.rate_limit)
    
    print(f"\n    Done! Processed: {processed_count}, Failed: {failed_count}")
    
    return processed_count


def create_comparison_frames(original_dir, processed_dir, comparison_dir, frame_files):
    """Create side-by-side comparison frames."""
    print(f"\n[3/4] Creating comparison frames...")
    
    os.makedirs(comparison_dir, exist_ok=True)
    
    for frame_file in frame_files:
        orig_path = os.path.join(original_dir, frame_file)
        proc_path = os.path.join(processed_dir, frame_file)
        comp_path = os.path.join(comparison_dir, frame_file)
        
        if os.path.exists(orig_path) and os.path.exists(proc_path):
            orig = Image.open(orig_path)
            proc = Image.open(proc_path).resize(orig.size)
            
            # Create side-by-side
            comparison = Image.new('RGB', (orig.width * 2, orig.height))
            comparison.paste(orig, (0, 0))
            comparison.paste(proc, (orig.width, 0))
            comparison.save(comp_path)
    
    print(f"    Created {len(frame_files)} comparison frames")


def reassemble_video(frames_dir, output_path, fps, audio_source=None, codec='libx264', crf=18):
    """Reassemble frames into video."""
    print(f"\n[4/4] Assembling video...")
    
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-framerate', str(fps),
        '-i', f'{frames_dir}/frame_%05d.png',
        '-c:v', codec,
        '-pix_fmt', 'yuv420p',
        '-crf', str(crf),
    ]
    
    if audio_source and os.path.exists(audio_source):
        cmd.extend(['-i', audio_source, '-c:a', 'aac', '-shortest'])
    
    cmd.append(output_path)
    
    subprocess.run(cmd, check=True)
    print(f"    Created: {output_path}")


def get_random_video(folder='library/video'):
    """Get a random video from input folder."""
    import random
    videos = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.mov', '.avi', '.webm'))]
    if not videos:
        return None
    return os.path.join(folder, random.choice(videos))


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='AI-powered frame-by-frame video processor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Basic usage with random input video
  python generative-frame-processor.py -p "oil painting style"
  
  # Specific video with custom settings  
  python generative-frame-processor.py -i video.mp4 -p "anime" --strength 0.6 --model sdxl
  
  # Quick test (3 seconds, low fps)
  python generative-frame-processor.py -p "watercolor" --duration 3 --fps 6
  
  # High quality output
  python generative-frame-processor.py -p "cinematic" --fps 24 --width 1024 --height 576

MODELS:
  sd15   - Stable Diffusion 1.5 (fast, good for stylization)
  sdxl   - Stable Diffusion XL (better quality, slower)
  flux   - FLUX Pro (best quality, most expensive)
  flux-schnell - FLUX Schnell (fast FLUX variant)
        """
    )
    
    # Input/Output
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('-i', '--input', type=str, default=None,
                          help='Input video path (default: random from library/video)')
    io_group.add_argument('-o', '--output-dir', type=str, default=DEFAULTS['output_dir'],
                          help=f"Output directory (default: {DEFAULTS['output_dir']})")
    io_group.add_argument('--keep-temp', action='store_true', default=DEFAULTS['keep_temp'],
                          help='Keep temporary frame files')
    
    # Video settings
    video_group = parser.add_argument_group('Video Settings')
    video_group.add_argument('--fps', type=int, default=DEFAULTS['fps'],
                             help=f"Frames per second (default: {DEFAULTS['fps']})")
    video_group.add_argument('--duration', type=float, default=DEFAULTS['duration'],
                             help=f"Duration in seconds (default: {DEFAULTS['duration']})")
    video_group.add_argument('--width', type=int, default=None,
                             help="Output width (default: match source video)")
    video_group.add_argument('--height', type=int, default=None,
                             help="Output height (default: match source video)")
    video_group.add_argument('--auto-size', action='store_true', default=True,
                             help="Auto-detect size from source video (default: True)")
    video_group.add_argument('--no-auto-size', action='store_false', dest='auto_size',
                             help="Don't auto-detect size, use --width/--height or 768x768")
    video_group.add_argument('--video-quality', type=int, default=DEFAULTS['video_quality'],
                             help=f"Video CRF quality, lower=better (default: {DEFAULTS['video_quality']})")
    video_group.add_argument('--video-codec', type=str, default=DEFAULTS['video_codec'],
                             help=f"Video codec (default: {DEFAULTS['video_codec']})")
    
    # Generation settings
    gen_group = parser.add_argument_group('Generation Settings')
    gen_group.add_argument('-p', '--prompt', type=str, default=DEFAULTS['prompt'],
                           help=f"Generation prompt (default: '{DEFAULTS['prompt']}')")
    gen_group.add_argument('-n', '--negative-prompt', type=str, default=DEFAULTS['negative_prompt'],
                           help=f"Negative prompt (default: '{DEFAULTS['negative_prompt']}')")
    gen_group.add_argument('-s', '--strength', type=float, default=DEFAULTS['strength'],
                           help=f"Transform strength 0.0-1.0 (default: {DEFAULTS['strength']})")
    gen_group.add_argument('-g', '--guidance-scale', type=float, default=DEFAULTS['guidance_scale'],
                           help=f"Guidance scale 1-20 (default: {DEFAULTS['guidance_scale']})")
    gen_group.add_argument('--seed', type=int, default=None,
                           help='Random seed (default: random)')
    gen_group.add_argument('--consistent-seed', action='store_true', default=True,
                           help='Use same seed for all frames (reduces flickering)')
    gen_group.add_argument('--no-consistent-seed', action='store_false', dest='consistent_seed',
                           help='Use different seed per frame')
    
    # Backend settings
    backend_group = parser.add_argument_group('Backend Settings')
    backend_group.add_argument('-b', '--backend', type=str, default=DEFAULTS['backend'],
                               choices=['replicate', 'fal', 'local'],
                               help=f"Processing backend (default: {DEFAULTS['backend']})")
    backend_group.add_argument('-m', '--model', type=str, default=DEFAULTS['model'],
                               choices=['sd15', 'sdxl', 'flux', 'flux-schnell'],
                               help=f"Model to use (default: {DEFAULTS['model']})")
    
    # Processing options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument('--skip-existing', action='store_true', default=DEFAULTS['skip_existing'],
                            help='Skip already processed frames')
    proc_group.add_argument('--no-skip-existing', action='store_false', dest='skip_existing',
                            help='Reprocess all frames')
    proc_group.add_argument('--rate-limit', type=float, default=DEFAULTS['rate_limit_delay'],
                            help=f"Seconds between API calls (default: {DEFAULTS['rate_limit_delay']})")
    proc_group.add_argument('--comparison', action='store_true', default=False,
                            help='Also create side-by-side comparison video')
    proc_group.add_argument('--audio', type=str, default=None,
                            help='Audio file to add to output')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Get input video
    if args.input:
        input_video = args.input
    else:
        input_video = get_random_video()
        if not input_video:
            print("Error: No input video specified and none found in library/video/")
            sys.exit(1)
    
    if not os.path.exists(input_video):
        print(f"Error: Input video not found: {input_video}")
        sys.exit(1)
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = f"genframe_work_{timestamp}"
    frames_dir = os.path.join(work_dir, "frames")
    processed_dir = os.path.join(work_dir, "processed")
    comparison_dir = os.path.join(work_dir, "comparison")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Auto-detect video size if enabled
    if args.auto_size and not (args.width and args.height):
        detected_width, detected_height = get_video_dimensions(input_video)
        # Round to nearest 64 for SD compatibility
        args.width = (detected_width // 64) * 64
        args.height = (detected_height // 64) * 64
        # Cap at reasonable size for API
        args.width = min(args.width, 1024)
        args.height = min(args.height, 1024)
    elif not args.width or not args.height:
        args.width = 768
        args.height = 768
    
    # Print config
    print("=" * 60)
    print("GENERATIVE FRAME PROCESSOR")
    print("=" * 60)
    print(f"Input: {input_video}")
    print(f"Backend: {args.backend} / Model: {args.model}")
    print(f"Duration: {args.duration}s at {args.fps} FPS ({int(args.duration * args.fps)} frames)")
    print(f"Size: {args.width}x{args.height} (auto-detected)" if args.auto_size else f"Size: {args.width}x{args.height}")
    print(f"Strength: {args.strength}")
    print(f"Rate limit: {args.rate_limit}s between calls")
    print("=" * 60)
    
    # Initialize backend
    if args.backend == 'replicate':
        backend = ReplicateBackend(args.model)
    elif args.backend == 'fal':
        backend = FalBackend(args.model)
    elif args.backend == 'local':
        backend = LocalBackend(args.model)
    else:
        print(f"Unknown backend: {args.backend}")
        sys.exit(1)
    
    try:
        # Step 1: Extract frames
        frame_files = extract_frames(
            input_video, frames_dir, 
            args.fps, args.duration, 
            args.width, args.height
        )
        
        # Step 2: Process frames
        process_frames(backend, frames_dir, processed_dir, frame_files, args)
        
        # Step 3: Create comparison if requested
        if args.comparison:
            create_comparison_frames(frames_dir, processed_dir, comparison_dir, frame_files)
        
        # Step 4: Reassemble video(s)
        output_name = f"genframe_{args.model}_{timestamp}.mp4"
        output_path = os.path.join(args.output_dir, output_name)
        
        reassemble_video(
            processed_dir, output_path, 
            args.fps, args.audio, 
            args.video_codec, args.video_quality
        )
        
        if args.comparison:
            comparison_output = os.path.join(args.output_dir, f"genframe_{args.model}_{timestamp}_comparison.mp4")
            reassemble_video(comparison_dir, comparison_output, args.fps)
        
        # Save metadata
        metadata = {
            'input': input_video,
            'output': output_path,
            'timestamp': timestamp,
            'settings': {
                'prompt': args.prompt,
                'negative_prompt': args.negative_prompt,
                'strength': args.strength,
                'guidance_scale': args.guidance_scale,
                'seed': args.seed,
                'backend': args.backend,
                'model': args.model,
                'fps': args.fps,
                'duration': args.duration,
                'width': args.width,
                'height': args.height,
            }
        }
        
        metadata_path = os.path.join(args.output_dir, f"genframe_{args.model}_{timestamp}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✓ COMPLETE!")
        print("=" * 60)
        print(f"Output: {output_path}")
        print(f"Metadata: {metadata_path}")
        if args.comparison:
            print(f"Comparison: {comparison_output}")
        print("=" * 60)
        
    finally:
        # Cleanup
        if not args.keep_temp and os.path.exists(work_dir):
            print("\nCleaning up temporary files...")
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()

