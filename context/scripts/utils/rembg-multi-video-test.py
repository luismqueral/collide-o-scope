"""
rembg-multi-video-test.py - Multi-video per-frame ML background removal

Composites multiple videos together with TRUE per-frame rembg on each layer.
This is SLOW but shows what proper ML compositing looks like.

Usage: python rembg-multi-video-test.py [num_videos] [duration_seconds]
"""

import os
import sys
import subprocess
import shutil
import random
from datetime import datetime

# Check for dependencies
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

# Frame rate for output (lower = faster processing)
OUTPUT_FPS = 12  # 12 = classic animation, 15 = smooth stop-motion

# Output size
OUTPUT_SIZE = (500, 500)

# Default settings
DEFAULT_NUM_VIDEOS = 3
DEFAULT_DURATION = 5  # seconds

# Blend mode for compositing
# 'alpha' = simple alpha compositing (foregrounds over each other)
# 'additive' = add pixel values (brighter, glowy)
BLEND_MODE = 'alpha'

# Background options
# True = use a random frame from another video as background
# False = use solid dark background
USE_VIDEO_BACKGROUND = True

# Solid background color (only used if USE_VIDEO_BACKGROUND = False)
SOLID_BG_COLOR = (20, 20, 25, 255)  # Dark blue-gray

# =============================================================================
# MAIN
# =============================================================================

def get_random_videos(folder='library/video', count=3):
    """Get random videos from the input folder."""
    videos = [os.path.join(folder, f) for f in os.listdir(folder) 
              if f.endswith(('.mp4', '.mov', '.avi'))]
    if len(videos) < count:
        print(f"Not enough videos. Found {len(videos)}, need {count}")
        sys.exit(1)
    return random.sample(videos, count)


def extract_background_frame(folder='library/video', exclude_videos=None):
    """Extract a single random frame from a random video to use as background."""
    exclude_videos = exclude_videos or []
    exclude_names = [os.path.basename(v) for v in exclude_videos]
    
    videos = [os.path.join(folder, f) for f in os.listdir(folder) 
              if f.endswith(('.mp4', '.mov', '.avi')) and f not in exclude_names]
    
    if not videos:
        print("No videos available for background, using solid color")
        return None
    
    bg_video = random.choice(videos)
    print(f"    Background video: {os.path.basename(bg_video)}")
    
    # Get video duration
    duration_cmd = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=nw=1:nk=1', bg_video],
        capture_output=True, text=True
    )
    vid_duration = float(duration_cmd.stdout.strip() or 60)
    random_time = random.uniform(0, max(0, vid_duration - 1))
    
    # Extract frame
    temp_path = f"/tmp/bg_frame_{random.randint(1000,9999)}.png"
    subprocess.run([
        'ffmpeg', '-y', '-v', 'error',
        '-ss', str(random_time),
        '-i', bg_video,
        '-frames:v', '1',
        '-vf', f'scale={OUTPUT_SIZE[0]}:{OUTPUT_SIZE[1]}',
        temp_path
    ], check=True)
    
    bg_image = Image.open(temp_path).convert('RGBA')
    os.remove(temp_path)
    
    return bg_image


def process_multi_video(num_videos, duration, output_fps):
    """Process multiple videos with per-frame rembg and composite them."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = f"perframe_multi_{timestamp}"
    
    os.makedirs(work_dir, exist_ok=True)
    
    # Get random videos
    input_videos = get_random_videos(count=num_videos)
    
    total_frames = duration * output_fps
    
    print(f"\n{'='*60}")
    print(f"MULTI-VIDEO PER-FRAME REMBG")
    print(f"{'='*60}")
    print(f"Videos: {num_videos}")
    print(f"Duration: {duration}s @ {output_fps}fps = {total_frames} frames")
    print(f"Blend mode: {BLEND_MODE}")
    print(f"Background: {'Video frame' if USE_VIDEO_BACKGROUND else 'Solid color'}")
    print(f"{'='*60}")
    for i, v in enumerate(input_videos):
        print(f"  V{i}: {os.path.basename(v)}")
    print(f"{'='*60}\n")
    
    # Get background frame if enabled
    bg_frame = None
    if USE_VIDEO_BACKGROUND:
        print("[0/4] Extracting background frame...")
        bg_frame = extract_background_frame(exclude_videos=input_videos)
        if bg_frame:
            print(f"    ✓ Background frame extracted")
        else:
            print(f"    Using solid color fallback")
    
    # Step 1: Extract frames from all videos
    print(f"[1/4] Extracting frames from {num_videos} videos...")
    
    video_frames = {}
    for i, video_path in enumerate(input_videos):
        frames_dir = os.path.join(work_dir, f"video{i}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Get random start time
        duration_cmd = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=nw=1:nk=1', video_path],
            capture_output=True, text=True
        )
        vid_duration = float(duration_cmd.stdout.strip() or 60)
        start_time = random.uniform(0, max(0, vid_duration - duration - 1))
        
        subprocess.run([
            'ffmpeg', '-y', '-v', 'error',
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(duration),
            '-vf', f'fps={output_fps},scale={OUTPUT_SIZE[0]}:{OUTPUT_SIZE[1]}',
            f'{frames_dir}/frame_%05d.png'
        ], check=True)
        
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        video_frames[i] = (frames_dir, frame_files)
        print(f"    V{i}: {len(frame_files)} frames from {os.path.basename(video_path)}")
    
    # Step 2: Process each frame with rembg for all videos
    print(f"\n[2/4] Running rembg on all frames...")
    
    processed_dirs = {}
    for vid_idx in range(num_videos):
        frames_dir, frame_files = video_frames[vid_idx]
        processed_dir = os.path.join(work_dir, f"video{vid_idx}_processed")
        os.makedirs(processed_dir, exist_ok=True)
        processed_dirs[vid_idx] = processed_dir
        
        print(f"    Processing V{vid_idx}...")
        for frame_idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, frame_file)
            output_path = os.path.join(processed_dir, frame_file)
            
            img = Image.open(frame_path).convert('RGBA')
            result = rembg_remove(img)
            result.save(output_path)
            
            progress = (frame_idx + 1) / len(frame_files) * 100
            print(f"        Frame {frame_idx+1}/{len(frame_files)} ({progress:.0f}%)", end='\r')
        print(f"        V{vid_idx} complete!                    ")
    
    # Step 3: Composite all frames together
    print(f"\n[3/4] Compositing {num_videos} video layers...")
    
    composite_dir = os.path.join(work_dir, "composite")
    os.makedirs(composite_dir, exist_ok=True)
    
    # Get the minimum frame count across all videos
    min_frames = min(len(video_frames[i][1]) for i in range(num_videos))
    
    for frame_idx in range(min_frames):
        frame_name = f"frame_{frame_idx+1:05d}.png"
        
        # Start with video 0 as base (with its background removed)
        base_path = os.path.join(processed_dirs[0], video_frames[0][1][frame_idx])
        composite = Image.open(base_path).convert('RGBA')
        
        # Create background (either video frame or solid color)
        if bg_frame is not None:
            bg = bg_frame.copy()
        else:
            bg = Image.new('RGBA', OUTPUT_SIZE, SOLID_BG_COLOR)
        composite = Image.alpha_composite(bg, composite)
        
        # Layer each subsequent video on top
        for vid_idx in range(1, num_videos):
            layer_path = os.path.join(processed_dirs[vid_idx], 
                                      video_frames[vid_idx][1][frame_idx])
            layer = Image.open(layer_path).convert('RGBA')
            
            if BLEND_MODE == 'alpha':
                composite = Image.alpha_composite(composite, layer)
            elif BLEND_MODE == 'additive':
                # Additive blend
                comp_arr = np.array(composite).astype(float)
                layer_arr = np.array(layer).astype(float)
                # Add RGB where layer has alpha
                alpha = layer_arr[:, :, 3:4] / 255.0
                comp_arr[:, :, :3] = np.clip(
                    comp_arr[:, :, :3] + layer_arr[:, :, :3] * alpha,
                    0, 255
                )
                composite = Image.fromarray(comp_arr.astype(np.uint8))
        
        composite.convert('RGB').save(os.path.join(composite_dir, frame_name))
        
        progress = (frame_idx + 1) / min_frames * 100
        print(f"    Frame {frame_idx+1}/{min_frames} ({progress:.0f}%)", end='\r')
    
    print(f"    Compositing complete!                    ")
    
    # Step 4: Reassemble video
    print(f"\n[4/4] Encoding final video...")
    
    output_path = f"output/perframe_multi_{num_videos}vid_{timestamp}.mp4"
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
    
    # Cleanup
    print(f"\nCleaning up work directory...")
    shutil.rmtree(work_dir)
    
    return output_path


if __name__ == "__main__":
    num_videos = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NUM_VIDEOS
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DURATION
    
    print(f"Processing {num_videos} videos for {duration} seconds...")
    
    process_multi_video(num_videos, duration, OUTPUT_FPS)

