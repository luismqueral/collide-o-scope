"""
test-variations.py - Test multiple parameter combinations on a single frame

Creates a grid of outputs varying strength, guidance, and prompts
so you can visually compare and find the sweet spot.

Usage:
  python test-variations.py
  python test-variations.py --input my_video.mp4
  python test-variations.py --quick  # fewer variations, faster
"""

import os
import sys
import subprocess
import shutil
import random
import base64
import urllib.request
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =============================================================================
# CONFIGURATION - Edit these to customize your test
# =============================================================================

# Strength values to test (how much to change the image)
STRENGTHS = [0.2, 0.35, 0.5, 0.65, 0.8]

# Guidance scale values to test (how strictly to follow prompt)
GUIDANCES = [4, 7.5, 12]

# Prompts to test
PROMPTS = [
    "high quality, detailed, enhanced",
    "oil painting, impressionist, vibrant colors",
    "cinematic, film grain, moody lighting",
]

# Quick mode (fewer variations for faster testing)
QUICK_STRENGTHS = [0.25, 0.5, 0.75]
QUICK_GUIDANCES = [7.5]
QUICK_PROMPTS = [
    "high quality, detailed",
    "oil painting, artistic",
]

# Output settings
OUTPUT_SIZE = (512, 512)  # Smaller for faster processing
GRID_PADDING = 10
LABEL_HEIGHT = 40

# =============================================================================
# REPLICATE API
# =============================================================================

def process_with_replicate(image_path, prompt, strength, guidance, seed):
    """Process a single image through Replicate."""
    try:
        import replicate
    except ImportError:
        print("Error: replicate not installed. Run: pip install replicate")
        sys.exit(1)
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    image_uri = f"data:image/png;base64,{image_data}"
    
    model_id = "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf"
    
    input_params = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, watermark, text",
        "image": image_uri,
        "prompt_strength": strength,
        "guidance_scale": guidance,
        "num_inference_steps": 30,
        "width": OUTPUT_SIZE[0],
        "height": OUTPUT_SIZE[1],
        "seed": seed,
    }
    
    output = replicate.run(model_id, input=input_params)
    
    if isinstance(output, list):
        return output[0] if output else None
    return output


def download_image(url, output_path):
    """Download image from URL."""
    urllib.request.urlretrieve(url, output_path)


# =============================================================================
# FRAME EXTRACTION
# =============================================================================

def extract_single_frame(video_path, output_path, width, height):
    """Extract a single frame from video."""
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-i', video_path,
        '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
        '-frames:v', '1',
        output_path
    ]
    subprocess.run(cmd, check=True)


def get_random_video(folder='library/video'):
    """Get a random video from input folder."""
    videos = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.mov', '.avi', '.webm'))]
    if not videos:
        return None
    return os.path.join(folder, random.choice(videos))


# =============================================================================
# GRID CREATION
# =============================================================================

def create_comparison_grid(images_dict, output_path, original_image=None):
    """
    Create a labeled grid of images.
    
    images_dict: {(strength, guidance, prompt_idx): image_path, ...}
    """
    # Get unique values
    strengths = sorted(set(k[0] for k in images_dict.keys()))
    guidances = sorted(set(k[1] for k in images_dict.keys()))
    prompts = sorted(set(k[2] for k in images_dict.keys()))
    
    # Calculate grid dimensions
    cols = len(strengths) + 1  # +1 for labels
    rows = len(guidances) * len(prompts) + 1  # +1 for header
    
    cell_width = OUTPUT_SIZE[0] + GRID_PADDING
    cell_height = OUTPUT_SIZE[1] + LABEL_HEIGHT + GRID_PADDING
    
    grid_width = cols * cell_width + 150  # Extra space for row labels
    grid_height = rows * cell_height + 100  # Extra for original
    
    # Create grid image
    grid = Image.new('RGB', (grid_width, grid_height), (30, 30, 35))
    draw = ImageDraw.Draw(grid)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Add original image at top
    if original_image and os.path.exists(original_image):
        orig = Image.open(original_image).resize(OUTPUT_SIZE)
        grid.paste(orig, (20, 20))
        draw.text((20, 20 + OUTPUT_SIZE[1] + 5), "ORIGINAL", fill=(255, 255, 255), font=font)
    
    # Header row - strength values
    for i, strength in enumerate(strengths):
        x = 150 + i * cell_width
        draw.text((x + 20, OUTPUT_SIZE[1] + 60), f"strength={strength}", fill=(255, 200, 100), font=font)
    
    # Draw images
    y_offset = OUTPUT_SIZE[1] + 100
    
    for p_idx, prompt_idx in enumerate(prompts):
        for g_idx, guidance in enumerate(guidances):
            row = p_idx * len(guidances) + g_idx
            y = y_offset + row * cell_height
            
            # Row label
            label = f"g={guidance}"
            if g_idx == 0:
                # Also show prompt (truncated)
                prompt_text = PROMPTS[prompt_idx][:30] + "..." if len(PROMPTS[prompt_idx]) > 30 else PROMPTS[prompt_idx]
                draw.text((10, y + 10), f"P{prompt_idx+1}: {prompt_text}", fill=(100, 200, 255), font=small_font)
            draw.text((10, y + 30), label, fill=(200, 200, 200), font=small_font)
            
            for s_idx, strength in enumerate(strengths):
                x = 150 + s_idx * cell_width
                
                key = (strength, guidance, prompt_idx)
                if key in images_dict and os.path.exists(images_dict[key]):
                    img = Image.open(images_dict[key]).resize(OUTPUT_SIZE)
                    grid.paste(img, (x, y))
    
    grid.save(output_path)
    print(f"Grid saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test parameter variations')
    parser.add_argument('-i', '--input', type=str, help='Input video path')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer variations)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for consistency')
    args = parser.parse_args()
    
    # Select parameters based on mode
    if args.quick:
        strengths = QUICK_STRENGTHS
        guidances = QUICK_GUIDANCES
        prompts = QUICK_PROMPTS
    else:
        strengths = STRENGTHS
        guidances = GUIDANCES
        prompts = PROMPTS
    
    total_variations = len(strengths) * len(guidances) * len(prompts)
    estimated_cost = total_variations * 0.002
    estimated_time = total_variations * 6 / 60  # ~6 sec per image
    
    print("=" * 60)
    print("PARAMETER VARIATION TEST")
    print("=" * 60)
    print(f"Strengths: {strengths}")
    print(f"Guidances: {guidances}")
    print(f"Prompts: {len(prompts)}")
    print(f"Total variations: {total_variations}")
    print(f"Estimated cost: ~${estimated_cost:.2f}")
    print(f"Estimated time: ~{estimated_time:.1f} minutes")
    print("=" * 60)
    
    # Get input video
    if args.input:
        input_video = args.input
    else:
        input_video = get_random_video()
        if not input_video:
            print("Error: No input video found")
            sys.exit(1)
    
    print(f"Input: {input_video}")
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = f"variations_test_{timestamp}"
    os.makedirs(work_dir, exist_ok=True)
    
    # Extract frame
    frame_path = os.path.join(work_dir, "original_frame.png")
    print("\nExtracting frame...")
    extract_single_frame(input_video, frame_path, OUTPUT_SIZE[0], OUTPUT_SIZE[1])
    
    # Use consistent seed
    seed = args.seed if args.seed else random.randint(0, 2**32 - 1)
    print(f"Using seed: {seed}")
    
    # Process variations
    results = {}
    count = 0
    
    print(f"\nProcessing {total_variations} variations...")
    
    for p_idx, prompt in enumerate(prompts):
        for guidance in guidances:
            for strength in strengths:
                count += 1
                output_path = os.path.join(work_dir, f"s{strength}_g{guidance}_p{p_idx}.png")
                
                print(f"[{count}/{total_variations}] strength={strength}, guidance={guidance}, prompt={p_idx+1}", end="")
                
                try:
                    result_url = process_with_replicate(
                        frame_path, prompt, strength, guidance, seed
                    )
                    
                    if result_url:
                        download_image(str(result_url), output_path)
                        results[(strength, guidance, p_idx)] = output_path
                        print(" ✓")
                    else:
                        print(" ✗ (no result)")
                        
                except Exception as e:
                    print(f" ✗ ({e})")
    
    # Create comparison grid
    print("\nCreating comparison grid...")
    grid_path = f"output/variations_grid_{timestamp}.png"
    os.makedirs("output", exist_ok=True)
    create_comparison_grid(results, grid_path, frame_path)
    
    # Also save individual images to output
    individual_dir = f"output/variations_{timestamp}"
    os.makedirs(individual_dir, exist_ok=True)
    
    # Copy original
    shutil.copy(frame_path, os.path.join(individual_dir, "00_original.png"))
    
    # Copy and rename results
    for (strength, guidance, p_idx), path in results.items():
        if os.path.exists(path):
            new_name = f"s{strength}_g{guidance}_p{p_idx}_{prompts[p_idx][:20].replace(' ', '_')}.png"
            shutil.copy(path, os.path.join(individual_dir, new_name))
    
    # Save prompts reference
    with open(os.path.join(individual_dir, "prompts.txt"), 'w') as f:
        for i, prompt in enumerate(prompts):
            f.write(f"P{i+1}: {prompt}\n")
    
    print(f"\nIndividual images: {individual_dir}/")
    
    # Cleanup
    shutil.rmtree(work_dir)
    
    print("\n" + "=" * 60)
    print("✓ COMPLETE!")
    print("=" * 60)
    print(f"Grid: {grid_path}")
    print(f"Individual images: {individual_dir}/")
    print("=" * 60)
    
    # Open the grid
    print("\nOpening grid...")
    subprocess.run(['open', grid_path])


if __name__ == "__main__":
    main()



