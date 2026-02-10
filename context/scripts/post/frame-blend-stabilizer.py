"""
frame-blend-stabilizer.py - Temporal Smoothing via Frame Blending

Reduces flickering in frame-by-frame SD animations by blending each frame
with its temporal neighbors. This creates smoother transitions while
preserving the overall motion and style.

HOW IT WORKS:
-------------
For each frame, we compute a weighted average of nearby frames:

  Frame N-2   Frame N-1   Frame N   Frame N+1   Frame N+2
    0.05   +   0.20    +  0.50   +   0.20    +   0.05     = Blended Frame N

The center frame gets the most weight (50%), so it dominates the result.
Neighboring frames contribute less, smoothing out frame-to-frame variations.

BLEND MODES:
------------
- 'light':  3-frame window (prev, current, next) - subtle smoothing
- 'medium': 5-frame window - balanced smoothing  
- 'heavy':  7-frame window - strong smoothing, may cause ghosting

USAGE:
------
  # From a video file
  python frame-blend-stabilizer.py -i video.mp4 -o smoothed.mp4 --mode medium

  # From a folder of frames (e.g., from SD processing)
  python frame-blend-stabilizer.py -i frames_folder/ -o smoothed.mp4 --mode medium
  
  # Preview: just process and save blended frames without video assembly
  python frame-blend-stabilizer.py -i frames_folder/ --frames-only -o blended_frames/
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: PIL and numpy required")
    print("Install with: pip install pillow numpy")
    sys.exit(1)


# =============================================================================
# BLEND WEIGHT CONFIGURATIONS
# =============================================================================
# 
# These define how much each neighboring frame contributes to the blend.
# Weights are symmetric around the center frame and sum to 1.0.
#
# Higher center weight = less smoothing, more detail preserved
# Lower center weight = more smoothing, potential ghosting on fast motion

BLEND_MODES = {
    # Light smoothing: 3-frame window
    # Good for: subtle flicker reduction, fast motion content
    # Weights: [prev, current, next]
    'light': {
        'weights': [0.20, 0.60, 0.20],
        'description': '3-frame blend (subtle smoothing)',
    },
    
    # Medium smoothing: 5-frame window  
    # Good for: most content, balanced smoothing
    # Weights: [N-2, N-1, N, N+1, N+2]
    'medium': {
        'weights': [0.05, 0.20, 0.50, 0.20, 0.05],
        'description': '5-frame blend (balanced smoothing)',
    },
    
    # Heavy smoothing: 7-frame window
    # Good for: very flickery content, slow/static scenes
    # Warning: may cause visible ghosting on fast motion
    # Weights: [N-3, N-2, N-1, N, N+1, N+2, N+3]
    'heavy': {
        'weights': [0.02, 0.08, 0.20, 0.40, 0.20, 0.08, 0.02],
        'description': '7-frame blend (strong smoothing)',
    },
}


# =============================================================================
# CORE BLENDING FUNCTIONS
# =============================================================================

def load_frame(path):
    """
    Load a frame as a numpy array for blending.
    
    We convert to float32 for accurate arithmetic operations.
    Values are normalized to 0-1 range for proper blending.
    
    Args:
        path: Path to PNG/JPG image file
        
    Returns:
        numpy array of shape (height, width, channels) with float32 values 0-1
    """
    img = Image.open(path).convert('RGB')
    # Convert to float32 and normalize to 0-1 range
    # This prevents overflow/clipping during weighted averaging
    return np.array(img, dtype=np.float32) / 255.0


def save_frame(array, path):
    """
    Save a blended frame array back to an image file.
    
    Converts from float32 (0-1) back to uint8 (0-255) for saving.
    Clips values to valid range in case of any arithmetic edge cases.
    
    Args:
        array: numpy array of shape (H, W, C) with float32 values 0-1
        path: Output path for PNG image
    """
    # Clip to valid range (in case of floating point edge cases)
    array = np.clip(array, 0.0, 1.0)
    # Convert back to uint8 for saving
    img = Image.fromarray((array * 255).astype(np.uint8))
    img.save(path, 'PNG')


def blend_frames(frames, weights):
    """
    Blend multiple frames using weighted averaging.
    
    This is the core blending operation. Each frame contributes to the
    output proportionally to its weight. Weights should sum to 1.0.
    
    Mathematical operation:
        output[x,y,c] = Σ (frames[i][x,y,c] * weights[i])
    
    Args:
        frames: List of numpy arrays (all same shape)
        weights: List of floats that sum to 1.0
        
    Returns:
        Blended frame as numpy array
    
    Example:
        # 3-frame blend with center emphasis
        result = blend_frames(
            [prev_frame, current_frame, next_frame],
            [0.2, 0.6, 0.2]
        )
    """
    # Verify weights sum to ~1.0 (allow small floating point error)
    assert abs(sum(weights) - 1.0) < 0.01, f"Weights must sum to 1.0, got {sum(weights)}"
    assert len(frames) == len(weights), "Must have same number of frames and weights"
    
    # Start with zeros in the same shape as input frames
    blended = np.zeros_like(frames[0])
    
    # Accumulate weighted contribution from each frame
    for frame, weight in zip(frames, weights):
        blended += frame * weight
    
    return blended


def get_frame_window(frame_arrays, center_idx, window_size):
    """
    Get a window of frames centered on a specific index.
    
    Handles edge cases at the start and end of the sequence by
    repeating the first/last frames (edge padding).
    
    Args:
        frame_arrays: List of all frame arrays
        center_idx: Index of the center frame
        window_size: Total window size (must be odd)
        
    Returns:
        List of frame arrays in the window
        
    Example:
        # For window_size=5, center_idx=2:
        # Returns frames at indices [0, 1, 2, 3, 4]
        
        # For window_size=5, center_idx=0 (edge case):
        # Returns frames at indices [0, 0, 0, 1, 2] (padded)
    """
    half_window = window_size // 2
    total_frames = len(frame_arrays)
    
    window = []
    for offset in range(-half_window, half_window + 1):
        idx = center_idx + offset
        
        # Clamp index to valid range (edge padding)
        # This repeats the first/last frame at sequence boundaries
        idx = max(0, min(idx, total_frames - 1))
        
        window.append(frame_arrays[idx])
    
    return window


# =============================================================================
# FRAME EXTRACTION & VIDEO ASSEMBLY
# =============================================================================

def extract_frames_from_video(video_path, output_dir, fps=None):
    """
    Extract all frames from a video file using ffmpeg.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        fps: Optional frame rate (None = use video's native fps)
        
    Returns:
        List of frame file paths in order
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = ['ffmpeg', '-y', '-v', 'error', '-i', video_path]
    
    if fps:
        cmd.extend(['-vf', f'fps={fps}'])
    
    cmd.append(f'{output_dir}/frame_%05d.png')
    
    subprocess.run(cmd, check=True)
    
    # Return sorted list of extracted frames
    frames = sorted([
        os.path.join(output_dir, f) 
        for f in os.listdir(output_dir) 
        if f.endswith('.png')
    ])
    
    return frames


def get_video_fps(video_path):
    """Get the frame rate of a video file."""
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ], capture_output=True, text=True)
    
    # Parse fraction like "30/1" or "24000/1001"
    fps_str = result.stdout.strip()
    if '/' in fps_str:
        num, den = fps_str.split('/')
        return float(num) / float(den)
    return float(fps_str)


def assemble_video(frames_dir, output_path, fps):
    """
    Reassemble frames into a video using ffmpeg.
    
    Args:
        frames_dir: Directory containing frame_00001.png, etc.
        output_path: Output video path
        fps: Frame rate for output video
    """
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


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_frames(input_frames, output_dir, mode='medium', progress_callback=None):
    """
    Apply temporal blending to a sequence of frames.
    
    This is the main processing function. It:
    1. Loads all frames into memory
    2. For each frame, gets a temporal window
    3. Blends the window using weighted averaging
    4. Saves the blended result
    
    Args:
        input_frames: List of input frame paths
        output_dir: Directory to save blended frames
        mode: Blend mode ('light', 'medium', 'heavy')
        progress_callback: Optional function(current, total) for progress updates
        
    Returns:
        List of output frame paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get blend configuration
    config = BLEND_MODES[mode]
    weights = config['weights']
    window_size = len(weights)
    
    print(f"Blend mode: {mode} ({config['description']})")
    print(f"Window size: {window_size} frames")
    print(f"Weights: {weights}")
    print()
    
    # -------------------------------------------------------------------------
    # STEP 1: Load all frames into memory
    # -------------------------------------------------------------------------
    # We load everything upfront for faster processing.
    # For very long videos, you could implement streaming/chunked processing.
    
    print(f"Loading {len(input_frames)} frames...")
    frame_arrays = []
    for i, path in enumerate(input_frames):
        frame_arrays.append(load_frame(path))
        if (i + 1) % 20 == 0:
            print(f"  Loaded {i + 1}/{len(input_frames)}")
    
    print(f"  All frames loaded ({frame_arrays[0].shape})")
    
    # -------------------------------------------------------------------------
    # STEP 2: Process each frame with temporal blending
    # -------------------------------------------------------------------------
    
    print(f"\nBlending frames...")
    output_frames = []
    
    for i in range(len(frame_arrays)):
        # Get the temporal window centered on this frame
        # For edge frames, this will repeat the first/last frame
        window = get_frame_window(frame_arrays, i, window_size)
        
        # Blend the frames using weighted averaging
        blended = blend_frames(window, weights)
        
        # Save the blended frame
        output_path = os.path.join(output_dir, f'frame_{i+1:05d}.png')
        save_frame(blended, output_path)
        output_frames.append(output_path)
        
        # Progress update
        if progress_callback:
            progress_callback(i + 1, len(frame_arrays))
        elif (i + 1) % 20 == 0 or i == len(frame_arrays) - 1:
            print(f"  Processed {i + 1}/{len(frame_arrays)}")
    
    return output_frames


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Temporal smoothing via frame blending',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
BLEND MODES:
  light   - 3-frame window, subtle smoothing (fast motion safe)
  medium  - 5-frame window, balanced smoothing (recommended)
  heavy   - 7-frame window, strong smoothing (may ghost on motion)

EXAMPLES:
  # Smooth a video file
  python frame-blend-stabilizer.py -i animation.mp4 -o smoothed.mp4

  # Process a folder of frames
  python frame-blend-stabilizer.py -i frames/ -o smoothed.mp4 --fps 14

  # Heavy smoothing for very flickery content
  python frame-blend-stabilizer.py -i animation.mp4 -o smoothed.mp4 --mode heavy
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input video file or folder of frames')
    parser.add_argument('-o', '--output', required=True,
                        help='Output video file or folder (with --frames-only)')
    parser.add_argument('--mode', choices=['light', 'medium', 'heavy'],
                        default='medium', help='Blend strength (default: medium)')
    parser.add_argument('--fps', type=float, default=None,
                        help='Output FPS (default: detect from input)')
    parser.add_argument('--frames-only', action='store_true',
                        help='Output blended frames only, no video assembly')
    parser.add_argument('--keep-temp', action='store_true',
                        help='Keep temporary frame files')
    
    args = parser.parse_args()
    
    # Determine if input is video or folder
    input_is_video = os.path.isfile(args.input)
    
    print("=" * 60)
    print("FRAME BLEND STABILIZER")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Mode: {args.mode}")
    print("=" * 60)
    
    # Setup working directory
    timestamp = str(int(__import__('time').time()))
    work_dir = f"/tmp/blend_stabilizer_{timestamp}"
    os.makedirs(work_dir, exist_ok=True)
    
    try:
        # ---------------------------------------------------------------------
        # Get input frames
        # ---------------------------------------------------------------------
        if input_is_video:
            print(f"\nExtracting frames from video...")
            fps = args.fps or get_video_fps(args.input)
            input_frames_dir = os.path.join(work_dir, "input_frames")
            input_frames = extract_frames_from_video(args.input, input_frames_dir)
            print(f"  Extracted {len(input_frames)} frames at {fps:.2f} fps")
        else:
            # Input is a folder of frames
            input_frames = sorted([
                os.path.join(args.input, f)
                for f in os.listdir(args.input)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
            fps = args.fps or 14  # Default FPS for frame folders
            print(f"\nFound {len(input_frames)} frames in folder")
        
        if len(input_frames) < 3:
            print("Error: Need at least 3 frames for blending")
            sys.exit(1)
        
        # ---------------------------------------------------------------------
        # Process frames
        # ---------------------------------------------------------------------
        blended_dir = os.path.join(work_dir, "blended_frames")
        output_frames = process_frames(input_frames, blended_dir, args.mode)
        
        # ---------------------------------------------------------------------
        # Output results
        # ---------------------------------------------------------------------
        if args.frames_only:
            # Copy blended frames to output location
            print(f"\nCopying blended frames to {args.output}...")
            if os.path.exists(args.output):
                shutil.rmtree(args.output)
            shutil.copytree(blended_dir, args.output)
        else:
            # Assemble into video
            print(f"\nAssembling video at {fps:.2f} fps...")
            assemble_video(blended_dir, args.output, fps)
        
        print("\n" + "=" * 60)
        print("✓ COMPLETE!")
        print("=" * 60)
        print(f"Output: {args.output}")
        print("=" * 60)
        
    finally:
        # Cleanup
        if not args.keep_temp and os.path.exists(work_dir):
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()

