"""
rembg-per-frame-test.py - Test per-frame ML background removal

This is SLOW but shows what true per-frame rembg looks like.
Creates a short test video with ML background removal on every frame.

Usage: python rembg-per-frame-test.py [input_video] [duration_seconds]
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime

# Check for rembg
try:
    from rembg import remove as rembg_remove
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: rembg, PIL, or numpy not installed")
    print("Install with: pip install rembg pillow numpy")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Frame rate for output (lower = faster processing, stop-motion look)
# 12 fps = classic animation, 15 fps = smooth stop-motion, 24 fps = film
OUTPUT_FPS = 12

# Output size
OUTPUT_SIZE = (500, 500)

# How many seconds to process (keep short for testing!)
DEFAULT_DURATION = 5

# =============================================================================
# MAIN
# =============================================================================

def get_random_video(folder='library/video'):
    """Get a random video from the input folder."""
    import random
    videos = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.mov', '.avi'))]
    if not videos:
        print(f"No videos found in {folder}")
        sys.exit(1)
    return os.path.join(folder, random.choice(videos))


def process_video_per_frame(input_video, duration, output_fps):
    """Process a video with rembg on every frame."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = f"perframe_work_{timestamp}"
    frames_dir = os.path.join(work_dir, "frames")
    processed_dir = os.path.join(work_dir, "processed")
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"PER-FRAME REMBG TEST")
    print(f"{'='*60}")
    print(f"Input: {os.path.basename(input_video)}")
    print(f"Duration: {duration} seconds")
    print(f"FPS: {output_fps}")
    print(f"Total frames to process: {duration * output_fps}")
    print(f"{'='*60}\n")
    
    # Step 1: Extract frames
    print(f"[1/4] Extracting frames at {output_fps} FPS...")
    subprocess.run([
        'ffmpeg', '-y', '-v', 'error',
        '-i', input_video,
        '-t', str(duration),
        '-vf', f'fps={output_fps},scale={OUTPUT_SIZE[0]}:{OUTPUT_SIZE[1]}',
        f'{frames_dir}/frame_%05d.png'
    ], check=True)
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    total_frames = len(frame_files)
    print(f"    Extracted {total_frames} frames")
    
    # Step 2: Process each frame with rembg
    print(f"\n[2/4] Processing frames with rembg (this will take a while)...")
    
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)
        output_path = os.path.join(processed_dir, frame_file)
        
        # Load frame
        img = Image.open(frame_path).convert('RGBA')
        
        # Run rembg
        result = rembg_remove(img)
        
        # Save with transparency
        result.save(output_path)
        
        # Progress
        progress = (i + 1) / total_frames * 100
        eta = (total_frames - i - 1) * 1.5  # Rough estimate: 1.5 sec per frame
        print(f"    Frame {i+1}/{total_frames} ({progress:.1f}%) - ETA: {eta:.0f}s", end='\r')
    
    print(f"\n    Done processing all frames!")
    
    # Step 3: Create a background video (solid color or another video)
    print(f"\n[3/4] Creating composite with transparent foreground...")
    
    # For now, composite over a solid color background
    bg_color = (30, 30, 40)  # Dark blue-gray
    
    composite_dir = os.path.join(work_dir, "composite")
    os.makedirs(composite_dir, exist_ok=True)
    
    for frame_file in frame_files:
        processed_path = os.path.join(processed_dir, frame_file)
        composite_path = os.path.join(composite_dir, frame_file)
        
        # Load processed frame with alpha
        fg = Image.open(processed_path).convert('RGBA')
        
        # Create background
        bg = Image.new('RGBA', fg.size, bg_color + (255,))
        
        # Composite
        composite = Image.alpha_composite(bg, fg)
        composite.convert('RGB').save(composite_path)
    
    # Step 4: Reassemble video
    print(f"\n[4/4] Reassembling video...")
    
    output_path = f"output/perframe_rembg_{timestamp}.mp4"
    os.makedirs('output', exist_ok=True)
    
    subprocess.run([
        'ffmpeg', '-y', '-v', 'error',
        '-framerate', str(output_fps),
        '-i', f'{composite_dir}/frame_%05d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_path
    ], check=True)
    
    print(f"\n{'='*60}")
    print(f"✓ Created: {output_path}")
    print(f"{'='*60}")
    
    # Also save the transparent frames as a separate video (for compositing later)
    transparent_output = f"output/perframe_rembg_{timestamp}_transparent.mov"
    subprocess.run([
        'ffmpeg', '-y', '-v', 'error',
        '-framerate', str(output_fps),
        '-i', f'{processed_dir}/frame_%05d.png',
        '-c:v', 'png',  # Lossless with alpha
        transparent_output
    ], check=True)
    print(f"✓ Transparent version: {transparent_output}")
    
    # Cleanup
    print(f"\nCleaning up work directory...")
    shutil.rmtree(work_dir)
    
    return output_path


if __name__ == "__main__":
    # Get input video
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
    else:
        input_video = get_random_video()
    
    # Get duration
    if len(sys.argv) > 2:
        duration = int(sys.argv[2])
    else:
        duration = DEFAULT_DURATION
    
    print(f"Using video: {input_video}")
    print(f"Duration: {duration} seconds at {OUTPUT_FPS} fps")
    
    process_video_per_frame(input_video, duration, OUTPUT_FPS)



