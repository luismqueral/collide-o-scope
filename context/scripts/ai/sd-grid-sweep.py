"""
sd-grid-sweep.py - 2D Parameter Grid Sweep (Single Frame)

Runs a grid of strength × guidance combinations on a single frame,
generating an HTML comparison grid for rapid parameter exploration.

Usage:
  python sd-grid-sweep.py --input video.mp4 --prompt "your prompt" \
    --strength-range 0.3 0.7 --strength-step 0.1 \
    --guidance-range 5.0 10.0 --guidance-step 0.5
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


def extract_single_frame(video_path, output_path, timestamp=None, width=768, height=768):
    """Extract a single frame from video."""
    if timestamp is None:
        # Get video duration and pick middle
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ], capture_output=True, text=True)
        duration = float(result.stdout.strip())
        timestamp = duration / 2
    
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-ss', str(timestamp),
        '-i', video_path,
        '-frames:v', '1',
        '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
        output_path
    ]
    subprocess.run(cmd, check=True)
    return timestamp


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
        print(f" Error: {e}")
        return None


def generate_grid_html(results, output_path, prompt, input_video, source_image_path, 
                       strength_values, guidance_values):
    """Generate HTML grid comparison."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_b64 = encode_image_base64(source_image_path)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SD Grid Sweep - Strength × Guidance</title>
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
            margin-bottom: 1rem;
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
        
        .source-section {{
            margin-bottom: 2rem;
            padding: 1rem;
            background: #0d0d0d;
            border-radius: 8px;
            display: inline-block;
        }}
        
        .source-section h2 {{
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 0.5rem;
            font-weight: normal;
        }}
        
        .source-section img {{
            width: 200px;
            height: 200px;
            border-radius: 4px;
            border: 2px solid #333;
        }}
        
        .grid-container {{
            overflow-x: auto;
        }}
        
        .grid-table {{
            border-collapse: collapse;
            background: #111;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .grid-table th {{
            background: #1a1a1a;
            padding: 0.75rem 1rem;
            font-weight: normal;
            font-size: 0.8rem;
            color: #888;
            border: 1px solid #222;
        }}
        
        .grid-table th.strength-header {{
            color: #4a9eff;
            writing-mode: vertical-rl;
            text-orientation: mixed;
            transform: rotate(180deg);
            padding: 1rem 0.5rem;
        }}
        
        .grid-table th.guidance-header {{
            color: #ff6b6b;
        }}
        
        .grid-table th.corner {{
            background: #0d0d0d;
            font-size: 0.7rem;
        }}
        
        .grid-table td {{
            padding: 4px;
            border: 1px solid #222;
            text-align: center;
            vertical-align: middle;
        }}
        
        .grid-cell {{
            position: relative;
        }}
        
        .grid-cell img {{
            width: 120px;
            height: 120px;
            object-fit: cover;
            border-radius: 4px;
            transition: transform 0.2s, z-index 0s;
            cursor: pointer;
        }}
        
        .grid-cell img:hover {{
            transform: scale(2.5);
            z-index: 1000;
            position: relative;
            box-shadow: 0 0 20px rgba(0,0,0,0.8);
        }}
        
        .grid-cell.failed img {{
            opacity: 0.3;
        }}
        
        .grid-cell .label {{
            position: absolute;
            bottom: 8px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.8);
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.6rem;
            color: #888;
            pointer-events: none;
        }}
        
        .legend {{
            margin-top: 2rem;
            padding: 1rem;
            background: #111;
            border-radius: 8px;
            font-size: 0.8rem;
            color: #666;
        }}
        
        .legend h3 {{
            font-size: 0.9rem;
            font-weight: normal;
            color: #888;
            margin-bottom: 0.5rem;
        }}
        
        .axis-label {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0.25rem 0;
        }}
        
        .axis-label .color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }}
        
        .stats {{
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #222;
        }}
    </style>
</head>
<body>
    <h1>SD Grid Sweep: Strength × Guidance</h1>
    <div class="meta">
        Generated: {timestamp} | Input: {os.path.basename(input_video)}
    </div>
    
    <div class="prompt">
        <strong>Prompt:</strong> {prompt}
    </div>
    
    <div class="source-section">
        <h2>Source Frame</h2>
        <img src="data:image/png;base64,{source_b64}" alt="Source">
    </div>
    
    <div class="grid-container">
        <table class="grid-table">
            <thead>
                <tr>
                    <th class="corner">strength ↓<br>guidance →</th>
"""
    
    # Guidance headers
    for g in guidance_values:
        html += f'                    <th class="guidance-header">{g:.1f}</th>\n'
    
    html += """                </tr>
            </thead>
            <tbody>
"""
    
    # Grid rows
    for s in strength_values:
        html += f'                <tr>\n'
        html += f'                    <th class="strength-header">{s:.2f}</th>\n'
        
        for g in guidance_values:
            key = f"{s:.2f}_{g:.1f}"
            result = results.get(key, {})
            
            if result.get('success') and result.get('image_path'):
                img_b64 = encode_image_base64(result['image_path'])
                html += f'''                    <td>
                        <div class="grid-cell">
                            <img src="data:image/png;base64,{img_b64}" alt="s={s:.2f} g={g:.1f}">
                        </div>
                    </td>
'''
            else:
                # Failed cell - show source with opacity
                html += f'''                    <td>
                        <div class="grid-cell failed">
                            <img src="data:image/png;base64,{source_b64}" alt="Failed">
                            <span class="label">failed</span>
                        </div>
                    </td>
'''
        
        html += '                </tr>\n'
    
    # Stats
    total = len(strength_values) * len(guidance_values)
    successful = sum(1 for r in results.values() if r.get('success'))
    
    html += f"""            </tbody>
        </table>
    </div>
    
    <div class="legend">
        <h3>Parameters</h3>
        <div class="axis-label">
            <span class="color" style="background: #4a9eff;"></span>
            <span><strong>Strength</strong> (vertical): {min(strength_values):.2f} → {max(strength_values):.2f} (step {strength_values[1]-strength_values[0]:.2f})</span>
        </div>
        <div class="axis-label">
            <span class="color" style="background: #ff6b6b;"></span>
            <span><strong>Guidance</strong> (horizontal): {min(guidance_values):.1f} → {max(guidance_values):.1f} (step {guidance_values[1]-guidance_values[0]:.1f})</span>
        </div>
        <div class="stats">
            <strong>Results:</strong> {successful}/{total} successful ({successful/total*100:.0f}%)
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"\nHTML grid saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='2D Grid Sweep (single frame)')
    
    parser.add_argument('-i', '--input', required=True, help='Input video path')
    parser.add_argument('-p', '--prompt', required=True, help='SD prompt')
    parser.add_argument('-n', '--negative-prompt', default='blurry, low quality, watermark, text')
    
    # Strength range
    parser.add_argument('--strength-range', type=float, nargs=2, default=[0.3, 0.7],
                        metavar=('MIN', 'MAX'), help='Strength range (default: 0.3 0.7)')
    parser.add_argument('--strength-step', type=float, default=0.1,
                        help='Strength increment (default: 0.1)')
    
    # Guidance range  
    parser.add_argument('--guidance-range', type=float, nargs=2, default=[5.0, 10.0],
                        metavar=('MIN', 'MAX'), help='Guidance range (default: 5.0 10.0)')
    parser.add_argument('--guidance-step', type=float, default=0.5,
                        help='Guidance increment (default: 0.5)')
    
    # Other
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--timestamp', type=float, default=None,
                        help='Video timestamp to extract frame from (default: middle)')
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--rate-limit', type=float, default=0.5)
    parser.add_argument('-o', '--output-dir', default='output/sweeps')
    parser.add_argument('--dry-run', action='store_true', help='Show grid without processing')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input not found: {args.input}")
        sys.exit(1)
    
    # Generate value ranges
    strength_values = []
    s = args.strength_range[0]
    while s <= args.strength_range[1] + 0.001:  # Small epsilon for float comparison
        strength_values.append(round(s, 2))
        s += args.strength_step
    
    guidance_values = []
    g = args.guidance_range[0]
    while g <= args.guidance_range[1] + 0.001:
        guidance_values.append(round(g, 1))
        g += args.guidance_step
    
    total_cells = len(strength_values) * len(guidance_values)
    
    print("=" * 60)
    print("SD GRID SWEEP (Single Frame)")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Prompt: {args.prompt}")
    print()
    print(f"Strength: {strength_values[0]:.2f} → {strength_values[-1]:.2f} (step {args.strength_step})")
    print(f"  Values: {', '.join(f'{s:.2f}' for s in strength_values)}")
    print()
    print(f"Guidance: {guidance_values[0]:.1f} → {guidance_values[-1]:.1f} (step {args.guidance_step})")
    print(f"  Values: {', '.join(f'{g:.1f}' for g in guidance_values)}")
    print()
    print(f"Grid: {len(strength_values)} × {len(guidance_values)} = {total_cells} combinations")
    print("=" * 60)
    
    if args.dry_run:
        print("\n[DRY RUN] Would generate grid with these combinations:")
        print()
        print("        |", end="")
        for g in guidance_values:
            print(f" g={g:<4.1f}", end="")
        print()
        print("-" * (9 + 8 * len(guidance_values)))
        for s in strength_values:
            print(f"s={s:.2f} |", end="")
            for g in guidance_values:
                print(f"   ✓   ", end="")
            print()
        print()
        print(f"Total API calls: {total_cells}")
        print(f"Estimated time: ~{total_cells * 3}s ({total_cells * 3 / 60:.1f} min)")
        return
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(args.output_dir, f"grid_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)
    
    seed = args.seed if args.seed else int(time.time()) % (2**31)
    print(f"Seed: {seed}")
    
    # Extract source frame
    source_path = os.path.join(sweep_dir, "source.png")
    print(f"\nExtracting source frame...")
    frame_time = extract_single_frame(args.input, source_path, args.timestamp, args.width, args.height)
    print(f"  Frame at {frame_time:.2f}s")
    
    # Process grid
    print(f"\nProcessing {total_cells} combinations...")
    results = {}
    processed = 0
    
    for i, s in enumerate(strength_values):
        for j, g in enumerate(guidance_values):
            key = f"{s:.2f}_{g:.1f}"
            cell_num = i * len(guidance_values) + j + 1
            
            print(f"  [{cell_num}/{total_cells}] s={s:.2f} g={g:.1f}", end="", flush=True)
            
            output_path = os.path.join(sweep_dir, f"s{s:.2f}_g{g:.1f}.png")
            
            url = process_frame_fal(
                source_path, args.prompt, args.negative_prompt,
                s, g, seed
            )
            
            if url:
                download_image(url, output_path)
                results[key] = {'success': True, 'image_path': output_path}
                processed += 1
                print(" ✓")
            else:
                results[key] = {'success': False}
                print(" ✗")
            
            if args.rate_limit > 0 and cell_num < total_cells:
                time.sleep(args.rate_limit)
    
    print(f"\nProcessed: {processed}/{total_cells}")
    
    # Generate HTML
    html_path = os.path.join(sweep_dir, "grid.html")
    generate_grid_html(
        results, html_path, args.prompt, args.input, source_path,
        strength_values, guidance_values
    )
    
    print("\n" + "=" * 60)
    print("✓ GRID SWEEP COMPLETE!")
    print("=" * 60)
    print(f"Results: {sweep_dir}")
    print(f"Grid HTML: {html_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

