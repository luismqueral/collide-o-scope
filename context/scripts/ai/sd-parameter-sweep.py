"""
sd-parameter-sweep.py - Run parameter sweeps with HTML comparison output

Runs multiple SD frame-by-frame tests with different parameters,
extracts sample frames, and generates an HTML document for easy comparison.

Usage:
  python sd-parameter-sweep.py --input video.mp4 --prompt "your prompt" --sweep strength
  python sd-parameter-sweep.py --input video.mp4 --prompt "your prompt" --sweep guidance
  python sd-parameter-sweep.py --input video.mp4 --prompt "your prompt" --sweep custom --strengths 0.3 0.5 0.7
"""

import os
import sys
import subprocess
import shutil
import json
import base64
import time
import argparse
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
    'duration': 5,
    'width': 768,
    'height': 768,
    'base_strength': 0.5,
    'base_guidance': 7.5,
    'rate_limit': 0.5,
    'output_dir': 'output/sweeps',
    'sample_frames': [0, 0.25, 0.5, 0.75, 1.0],  # Positions to sample (0-1)
}

SWEEP_PRESETS = {
    'strength': [0.35, 0.45, 0.55, 0.65],
    'guidance': [5.0, 7.5, 10.0, 12.5],
    'strength_wide': [0.25, 0.4, 0.55, 0.7, 0.85],
}


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def encode_image_base64(image_path):
    """Encode image to base64 for HTML embedding."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def download_image(url, output_path):
    """Download image from URL and ensure PNG format."""
    import urllib.request
    temp_path = output_path + '.tmp'
    urllib.request.urlretrieve(url, temp_path)
    
    try:
        img = Image.open(temp_path)
        img.save(output_path, 'PNG')
        os.remove(temp_path)
    except Exception:
        os.rename(temp_path, output_path)


def process_frame_fal(image_path, prompt, negative_prompt, strength, guidance_scale, seed):
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


def run_single_test(input_video, prompt, strength, guidance, seed, fps, duration, width, height, rate_limit, work_dir, negative_prompt="blurry, low quality, watermark"):
    """Run a single parameter configuration and return results."""
    
    frames_dir = os.path.join(work_dir, "frames")
    processed_dir = os.path.join(work_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Extract frames (reuse if already extracted)
    if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
        print(f"  Extracting frames...")
        frame_files = extract_frames(input_video, frames_dir, fps, duration, width, height)
    else:
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    print(f"  Processing {len(frame_files)} frames (strength={strength}, guidance={guidance})...")
    
    processed = 0
    failed = 0
    start_time = time.time()
    
    for i, frame_file in enumerate(frame_files):
        input_path = os.path.join(frames_dir, frame_file)
        output_path = os.path.join(processed_dir, frame_file)
        
        result = process_frame_fal(
            input_path, prompt, negative_prompt,
            strength, guidance, seed
        )
        
        if result:
            download_image(result, output_path)
            processed += 1
        else:
            shutil.copy(input_path, output_path)
            failed += 1
        
        # Progress
        elapsed = time.time() - start_time
        per_frame = elapsed / (i + 1)
        eta = per_frame * (len(frame_files) - i - 1)
        print(f"    [{i+1}/{len(frame_files)}] {per_frame:.1f}s/frame, ETA: {eta:.0f}s", end='\r')
        
        if rate_limit > 0 and i < len(frame_files) - 1:
            time.sleep(rate_limit)
    
    print(f"\n  Done: {processed} processed, {failed} failed")
    
    return {
        'frames_dir': frames_dir,
        'processed_dir': processed_dir,
        'frame_files': frame_files,
        'processed': processed,
        'failed': failed,
    }


def extract_sample_frames(processed_dir, frame_files, sample_positions, output_dir):
    """Extract sample frames at specified positions for comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    samples = []
    total_frames = len(frame_files)
    
    for pos in sample_positions:
        idx = min(int(pos * (total_frames - 1)), total_frames - 1)
        frame_file = frame_files[idx]
        src_path = os.path.join(processed_dir, frame_file)
        
        if os.path.exists(src_path):
            samples.append({
                'position': pos,
                'index': idx,
                'path': src_path,
            })
    
    return samples


def generate_html_comparison(sweep_results, output_path, prompt, input_video, sweep_type):
    """Generate HTML document comparing sweep results."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SD Parameter Sweep - {sweep_type}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
            background: #0a0a0a;
            color: #e0e0e0;
            padding: 2rem;
            line-height: 1.6;
        }}
        
        h1 {{
            font-size: 1.5rem;
            font-weight: 400;
            margin-bottom: 0.5rem;
            color: #fff;
        }}
        
        .meta {{
            font-size: 0.8rem;
            color: #666;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #222;
        }}
        
        .meta span {{
            display: block;
            margin: 0.25rem 0;
        }}
        
        .prompt {{
            background: #111;
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 2rem;
            font-size: 0.9rem;
            color: #aaa;
        }}
        
        .prompt strong {{
            color: #fff;
        }}
        
        .sweep-grid {{
            display: grid;
            gap: 2rem;
        }}
        
        .test-row {{
            background: #111;
            border-radius: 8px;
            padding: 1.5rem;
        }}
        
        .test-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #222;
        }}
        
        .test-params {{
            font-size: 1.1rem;
            color: #4a9eff;
        }}
        
        .test-stats {{
            font-size: 0.75rem;
            color: #666;
        }}
        
        .frames-row {{
            display: flex;
            gap: 0.5rem;
            overflow-x: auto;
            padding: 0.5rem 0;
        }}
        
        .frame-cell {{
            flex-shrink: 0;
            text-align: center;
        }}
        
        .frame-cell img {{
            width: 200px;
            height: 200px;
            object-fit: cover;
            border-radius: 4px;
            border: 2px solid #222;
            transition: transform 0.2s, border-color 0.2s;
        }}
        
        .frame-cell img:hover {{
            transform: scale(2);
            border-color: #4a9eff;
            z-index: 100;
            position: relative;
        }}
        
        .frame-label {{
            font-size: 0.7rem;
            color: #666;
            margin-top: 0.25rem;
        }}
        
        .video-link {{
            display: inline-block;
            margin-top: 0.5rem;
            padding: 0.25rem 0.5rem;
            background: #222;
            color: #4a9eff;
            text-decoration: none;
            border-radius: 4px;
            font-size: 0.75rem;
        }}
        
        .video-link:hover {{
            background: #333;
        }}
        
        .comparison-section {{
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid #222;
        }}
        
        .comparison-section h2 {{
            font-size: 1.1rem;
            font-weight: 400;
            margin-bottom: 1rem;
            color: #888;
        }}
        
        .side-by-side {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }}
        
        .comparison-card {{
            background: #0d0d0d;
            border-radius: 4px;
            padding: 1rem;
            text-align: center;
        }}
        
        .comparison-card img {{
            width: 100%;
            max-width: 300px;
            border-radius: 4px;
        }}
        
        .comparison-card .label {{
            margin-top: 0.5rem;
            font-size: 0.8rem;
            color: #4a9eff;
        }}
    </style>
</head>
<body>
    <h1>SD Parameter Sweep: {sweep_type}</h1>
    <div class="meta">
        <span>Generated: {timestamp}</span>
        <span>Input: {os.path.basename(input_video)}</span>
    </div>
    
    <div class="prompt">
        <strong>Prompt:</strong> {prompt}
    </div>
    
    <div class="sweep-grid">
"""
    
    for result in sweep_results:
        params_str = f"strength={result['strength']}, guidance={result['guidance']}"
        stats_str = f"{result['processed']}/{result['total_frames']} frames processed"
        
        html += f"""
        <div class="test-row">
            <div class="test-header">
                <span class="test-params">{params_str}</span>
                <span class="test-stats">{stats_str}</span>
            </div>
            <div class="frames-row">
"""
        
        for sample in result['samples']:
            # Embed image as base64
            img_data = encode_image_base64(sample['path'])
            position_label = f"{sample['position']*100:.0f}%"
            
            html += f"""
                <div class="frame-cell">
                    <img src="data:image/png;base64,{img_data}" alt="Frame at {position_label}">
                    <div class="frame-label">{position_label}</div>
                </div>
"""
        
        if result.get('video_path'):
            video_name = os.path.basename(result['video_path'])
            html += f"""
            </div>
            <a href="{video_name}" class="video-link">▶ Watch video</a>
        </div>
"""
        else:
            html += """
            </div>
        </div>
"""
    
    # Add source frame comparison
    if sweep_results and sweep_results[0].get('source_samples'):
        html += """
    </div>
    
    <div class="comparison-section">
        <h2>Original Source Frames</h2>
        <div class="frames-row">
"""
        for sample in sweep_results[0]['source_samples']:
            img_data = encode_image_base64(sample['path'])
            position_label = f"{sample['position']*100:.0f}%"
            html += f"""
            <div class="frame-cell">
                <img src="data:image/png;base64,{img_data}" alt="Source at {position_label}">
                <div class="frame-label">{position_label}</div>
            </div>
"""
        html += """
        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"HTML comparison saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SD Parameter Sweep with HTML output')
    
    parser.add_argument('-i', '--input', required=True, help='Input video path')
    parser.add_argument('-p', '--prompt', required=True, help='SD prompt')
    parser.add_argument('-n', '--negative-prompt', default='blurry, low quality, watermark, text',
                        help='Negative prompt')
    
    # Sweep configuration
    parser.add_argument('--sweep', type=str, default='strength',
                        choices=['strength', 'guidance', 'strength_wide', 'custom'],
                        help='Sweep type')
    parser.add_argument('--strengths', type=float, nargs='+',
                        help='Custom strength values (for --sweep custom)')
    parser.add_argument('--guidances', type=float, nargs='+',
                        help='Custom guidance values (for --sweep custom)')
    
    # Fixed parameters
    parser.add_argument('--strength', type=float, default=DEFAULTS['base_strength'],
                        help='Base strength (when not sweeping strength)')
    parser.add_argument('--guidance', type=float, default=DEFAULTS['base_guidance'],
                        help='Base guidance (when not sweeping guidance)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    # Video settings
    parser.add_argument('--fps', type=int, default=DEFAULTS['fps'])
    parser.add_argument('--duration', type=float, default=DEFAULTS['duration'])
    parser.add_argument('--width', type=int, default=DEFAULTS['width'])
    parser.add_argument('--height', type=int, default=DEFAULTS['height'])
    
    # Processing
    parser.add_argument('--rate-limit', type=float, default=DEFAULTS['rate_limit'])
    parser.add_argument('-o', '--output-dir', default=DEFAULTS['output_dir'])
    parser.add_argument('--keep-temp', action='store_true')
    parser.add_argument('--no-video', action='store_true', help='Skip video assembly')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        sys.exit(1)
    
    # Determine sweep values
    if args.sweep == 'custom':
        if args.strengths:
            sweep_values = [{'strength': s, 'guidance': args.guidance} for s in args.strengths]
            sweep_type = 'strength'
        elif args.guidances:
            sweep_values = [{'strength': args.strength, 'guidance': g} for g in args.guidances]
            sweep_type = 'guidance'
        else:
            print("Error: --sweep custom requires --strengths or --guidances")
            sys.exit(1)
    elif args.sweep == 'guidance':
        sweep_values = [{'strength': args.strength, 'guidance': g} for g in SWEEP_PRESETS['guidance']]
        sweep_type = 'guidance'
    else:
        sweep_values = [{'strength': s, 'guidance': args.guidance} for s in SWEEP_PRESETS.get(args.sweep, SWEEP_PRESETS['strength'])]
        sweep_type = 'strength'
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(args.output_dir, f"sweep_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)
    
    seed = args.seed if args.seed else int(time.time()) % (2**31)
    
    print("=" * 60)
    print("SD PARAMETER SWEEP")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Prompt: {args.prompt}")
    print(f"Sweep: {sweep_type} ({len(sweep_values)} configurations)")
    print(f"Seed: {seed}")
    print(f"Duration: {args.duration}s @ {args.fps}fps")
    print("=" * 60)
    
    sweep_results = []
    
    # Shared frames directory (extract once, reuse)
    shared_frames_dir = os.path.join(sweep_dir, "source_frames")
    
    for i, config in enumerate(sweep_values):
        strength = config['strength']
        guidance = config['guidance']
        
        print(f"\n[{i+1}/{len(sweep_values)}] Testing strength={strength}, guidance={guidance}")
        
        test_dir = os.path.join(sweep_dir, f"test_{i+1}_s{strength}_g{guidance}")
        os.makedirs(test_dir, exist_ok=True)
        
        # Link to shared frames
        frames_link = os.path.join(test_dir, "frames")
        if not os.path.exists(shared_frames_dir):
            os.makedirs(shared_frames_dir, exist_ok=True)
            extract_frames(args.input, shared_frames_dir, args.fps, args.duration, args.width, args.height)
        
        if not os.path.exists(frames_link):
            os.symlink(os.path.abspath(shared_frames_dir), frames_link)
        
        # Run test
        result = run_single_test(
            args.input, args.prompt, strength, guidance, seed,
            args.fps, args.duration, args.width, args.height,
            args.rate_limit, test_dir, args.negative_prompt
        )
        
        # Extract samples
        samples = extract_sample_frames(
            result['processed_dir'],
            result['frame_files'],
            DEFAULTS['sample_frames'],
            os.path.join(test_dir, "samples")
        )
        
        # Source samples (only for first test)
        source_samples = None
        if i == 0:
            source_samples = extract_sample_frames(
                shared_frames_dir,
                result['frame_files'],
                DEFAULTS['sample_frames'],
                os.path.join(sweep_dir, "source_samples")
            )
        
        # Assemble video if requested
        video_path = None
        if not args.no_video and result['processed'] > 0:
            video_path = os.path.join(sweep_dir, f"sweep_s{strength}_g{guidance}.mp4")
            reassemble_video(result['processed_dir'], video_path, args.fps)
        
        sweep_results.append({
            'strength': strength,
            'guidance': guidance,
            'processed': result['processed'],
            'failed': result['failed'],
            'total_frames': len(result['frame_files']),
            'samples': samples,
            'source_samples': source_samples,
            'video_path': video_path,
        })
    
    # Generate HTML comparison
    html_path = os.path.join(sweep_dir, "comparison.html")
    generate_html_comparison(sweep_results, html_path, args.prompt, args.input, sweep_type)
    
    print("\n" + "=" * 60)
    print("✓ SWEEP COMPLETE!")
    print("=" * 60)
    print(f"Results: {sweep_dir}")
    print(f"Comparison: {html_path}")
    print("=" * 60)
    
    # Cleanup if requested
    if not args.keep_temp:
        # Keep only samples and videos, remove processed frames
        for i, config in enumerate(sweep_values):
            test_dir = os.path.join(sweep_dir, f"test_{i+1}_s{config['strength']}_g{config['guidance']}")
            processed_dir = os.path.join(test_dir, "processed")
            if os.path.exists(processed_dir):
                shutil.rmtree(processed_dir)


if __name__ == "__main__":
    main()

