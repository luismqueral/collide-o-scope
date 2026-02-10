"""
optical-flow-stabilizer.py - Temporal Coherence via Optical Flow Warping

Uses optical flow to warp each processed frame based on the motion in the
source video, then blends with the next processed frame. This creates
much smoother transitions than simple frame blending.

HOW IT WORKS:
-------------
For each pair of consecutive frames:

1. Calculate optical flow between SOURCE frames N and N+1
   - This tells us how each pixel moved in the original video
   
2. Warp PROCESSED frame N using this flow
   - Moves pixels in the processed frame to match the motion
   
3. Blend the warped frame with processed frame N+1
   - Combines motion-compensated previous frame with current frame
   - Reduces flickering while preserving motion

OPTICAL FLOW ALGORITHMS:
------------------------
- 'farneback': Dense flow, good quality, moderate speed (default)
- 'dis': DIS optical flow, faster, still good quality
- 'lucas-kanade': Sparse flow, very fast but less accurate

USAGE:
------
  python optical-flow-stabilizer.py \
    --source original_video.mp4 \
    --processed sd_output.mp4 \
    --output stabilized.mp4

  # Adjust blend strength (0=all warped, 1=all current)
  python optical-flow-stabilizer.py \
    --source original.mp4 --processed styled.mp4 \
    --output stabilized.mp4 --blend 0.6
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install opencv-python numpy pillow")
    sys.exit(1)


# =============================================================================
# OPTICAL FLOW FUNCTIONS
# =============================================================================

def calculate_optical_flow_farneback(frame1, frame2):
    """
    Calculate dense optical flow using Farneback algorithm.
    
    This computes the motion vector for every pixel between two frames.
    The result is a 2-channel image where:
      - Channel 0: horizontal displacement (x direction)
      - Channel 1: vertical displacement (y direction)
    
    Args:
        frame1: First frame (grayscale numpy array)
        frame2: Second frame (grayscale numpy array)
        
    Returns:
        Flow field of shape (H, W, 2) containing (dx, dy) per pixel
    """
    # Farneback parameters:
    # - pyr_scale: pyramid scale (0.5 = classical pyramid)
    # - levels: number of pyramid levels
    # - winsize: averaging window size (larger = smoother)
    # - iterations: number of iterations at each level
    # - poly_n: size of pixel neighborhood for polynomial expansion
    # - poly_sigma: std of Gaussian for polynomial expansion
    # - flags: operation flags
    
    flow = cv2.calcOpticalFlowFarneback(
        frame1, frame2,
        None,                    # Output flow (None = allocate new)
        pyr_scale=0.5,          # Pyramid scale
        levels=3,               # Pyramid levels
        winsize=15,             # Window size
        iterations=3,           # Iterations per level
        poly_n=5,               # Polynomial expansion neighborhood
        poly_sigma=1.2,         # Polynomial expansion sigma
        flags=0
    )
    
    return flow


def calculate_optical_flow_dis(frame1, frame2):
    """
    Calculate optical flow using DIS (Dense Inverse Search) algorithm.
    
    DIS is faster than Farneback while maintaining good quality.
    Good choice for longer videos where speed matters.
    """
    # Create DIS optical flow object
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    
    flow = dis.calc(frame1, frame2, None)
    
    return flow


def warp_frame_with_flow(frame, flow):
    """
    Warp a frame using an optical flow field.
    
    This moves each pixel according to the flow vectors, effectively
    "predicting" what the frame would look like if it moved like the
    source video.
    
    The warping uses bilinear interpolation for smooth results.
    
    Args:
        frame: RGB frame to warp (H, W, 3)
        flow: Optical flow field (H, W, 2) with (dx, dy) per pixel
        
    Returns:
        Warped frame (H, W, 3)
    """
    h, w = flow.shape[:2]
    
    # Create coordinate grid
    # flow_map[y, x] = (x + dx, y + dy) = where to sample from
    flow_map = np.zeros((h, w, 2), dtype=np.float32)
    flow_map[:, :, 0] = np.arange(w)  # x coordinates
    flow_map[:, :, 1] = np.arange(h)[:, np.newaxis]  # y coordinates
    
    # Add flow to get sampling positions
    # We negate because remap samples FROM the position, not TO it
    flow_map[:, :, 0] += flow[:, :, 0]
    flow_map[:, :, 1] += flow[:, :, 1]
    
    # Remap using bilinear interpolation
    warped = cv2.remap(
        frame, 
        flow_map[:, :, 0], 
        flow_map[:, :, 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return warped


def blend_frames(frame1, frame2, alpha):
    """
    Blend two frames with linear interpolation.
    
    Args:
        frame1: First frame (the warped previous frame)
        frame2: Second frame (the current processed frame)
        alpha: Blend factor (0 = all frame1, 1 = all frame2)
        
    Returns:
        Blended frame
    """
    return cv2.addWeighted(frame1, 1.0 - alpha, frame2, alpha, 0)


# =============================================================================
# FRAME I/O
# =============================================================================

def extract_frames(video_path, output_dir, duration=None, fps=None, width=None, height=None):
    """Extract frames from a video with optional constraints."""
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = ['ffmpeg', '-y', '-v', 'error', '-i', video_path]
    
    # Duration limit
    if duration:
        cmd.extend(['-t', str(duration)])
    
    # Build filter string
    filters = []
    if fps:
        filters.append(f'fps={fps}')
    if width and height:
        filters.append(f'scale={width}:{height}:force_original_aspect_ratio=decrease')
        filters.append(f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2')
    
    if filters:
        cmd.extend(['-vf', ','.join(filters)])
    
    cmd.append(f'{output_dir}/frame_%05d.png')
    subprocess.run(cmd, check=True)
    
    frames = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith('.png')
    ])
    return frames


def get_video_fps(video_path):
    """Get FPS of a video file."""
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ], capture_output=True, text=True)
    
    fps_str = result.stdout.strip()
    if '/' in fps_str:
        num, den = fps_str.split('/')
        return float(num) / float(den)
    return float(fps_str)


def assemble_video(frames_dir, output_path, fps):
    """Assemble frames into video."""
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
# MAIN PROCESSING
# =============================================================================

def process_with_optical_flow(source_frames, processed_frames, output_dir, 
                               blend_alpha=0.6, flow_method='farneback'):
    """
    Apply optical flow stabilization to processed frames.
    
    For each frame N:
    1. Calculate flow from source N-1 to source N
    2. Warp processed frame N-1 using this flow
    3. Blend warped N-1 with processed N
    
    Args:
        source_frames: List of paths to original video frames
        processed_frames: List of paths to SD-processed frames
        output_dir: Output directory for stabilized frames
        blend_alpha: How much of current frame to keep (0.5-0.8 recommended)
        flow_method: 'farneback' or 'dis'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_frames = min(len(source_frames), len(processed_frames))
    
    print(f"Processing {n_frames} frames with optical flow ({flow_method})...")
    print(f"Blend alpha: {blend_alpha} (higher = more current frame)")
    
    # Choose flow calculation method
    if flow_method == 'dis':
        calc_flow = calculate_optical_flow_dis
    else:
        calc_flow = calculate_optical_flow_farneback
    
    # Load first frames
    prev_source_gray = cv2.cvtColor(
        cv2.imread(source_frames[0]), 
        cv2.COLOR_BGR2GRAY
    )
    prev_processed = cv2.imread(processed_frames[0])
    
    # First frame passes through unchanged
    cv2.imwrite(os.path.join(output_dir, 'frame_00001.png'), prev_processed)
    
    for i in range(1, n_frames):
        # Load current frames
        curr_source = cv2.imread(source_frames[i])
        curr_source_gray = cv2.cvtColor(curr_source, cv2.COLOR_BGR2GRAY)
        curr_processed = cv2.imread(processed_frames[i])
        
        # =====================================================================
        # STEP 1: Calculate optical flow from previous to current SOURCE frame
        # =====================================================================
        # This tells us how the scene moved in the original video
        flow = calc_flow(prev_source_gray, curr_source_gray)
        
        # =====================================================================
        # STEP 2: Warp the PREVIOUS PROCESSED frame using this flow
        # =====================================================================
        # This "predicts" what the previous processed frame would look like
        # if it moved the same way as the source
        warped_prev = warp_frame_with_flow(prev_processed, flow)
        
        # =====================================================================
        # STEP 3: Blend warped previous with current processed
        # =====================================================================
        # The warped frame carries style from the previous frame
        # The current frame has the "correct" content for this moment
        # Blending combines temporal consistency with frame accuracy
        stabilized = blend_frames(warped_prev, curr_processed, blend_alpha)
        
        # Save stabilized frame
        output_path = os.path.join(output_dir, f'frame_{i+1:05d}.png')
        cv2.imwrite(output_path, stabilized)
        
        # Progress
        if (i + 1) % 20 == 0 or i == n_frames - 1:
            print(f"  Processed {i + 1}/{n_frames}")
        
        # Update previous frames for next iteration
        prev_source_gray = curr_source_gray
        prev_processed = stabilized  # Use stabilized as prev for temporal accumulation
    
    return n_frames


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Optical flow stabilization for SD animations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE:
  python optical-flow-stabilizer.py \\
    --source output/output_20251201_235110.mp4 \\
    --processed output/genframe_sd15_20251205_150446.mp4 \\
    --output output/genframe_stabilized_flow.mp4

BLEND VALUES:
  0.5 = Equal blend (very smooth, may lose detail)
  0.6 = Balanced (recommended)
  0.7 = More current frame (less smoothing)
  0.8 = Subtle smoothing only
        """
    )
    
    parser.add_argument('--source', '-s', required=True,
                        help='Original source video (before SD processing)')
    parser.add_argument('--processed', '-p', required=True,
                        help='SD-processed video')
    parser.add_argument('--output', '-o', required=True,
                        help='Output stabilized video')
    parser.add_argument('--blend', '-b', type=float, default=0.6,
                        help='Blend alpha: 0=all warped, 1=all current (default: 0.6)')
    parser.add_argument('--method', '-m', choices=['farneback', 'dis'],
                        default='farneback', help='Flow algorithm (default: farneback)')
    parser.add_argument('--keep-temp', action='store_true',
                        help='Keep temporary frame files')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OPTICAL FLOW STABILIZER")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Processed: {args.processed}")
    print(f"Method: {args.method}")
    print(f"Blend: {args.blend}")
    print("=" * 60)
    
    # Setup
    import time
    work_dir = f"/tmp/flow_stabilizer_{int(time.time())}"
    os.makedirs(work_dir, exist_ok=True)
    
    try:
        # First extract processed frames to get dimensions
        print("\nExtracting processed frames...")
        processed_dir = os.path.join(work_dir, "processed")
        processed_frames = extract_frames(args.processed, processed_dir)
        print(f"  {len(processed_frames)} frames")
        
        fps = get_video_fps(args.processed)
        
        # Get dimensions from first processed frame
        first_processed = cv2.imread(processed_frames[0])
        height, width = first_processed.shape[:2]
        duration = len(processed_frames) / fps
        print(f"  Size: {width}x{height}, Duration: {duration:.1f}s, FPS: {fps}")
        
        # Extract source frames with MATCHING parameters
        print(f"\nExtracting source frames (matching {width}x{height}, {duration:.1f}s)...")
        source_dir = os.path.join(work_dir, "source")
        source_frames = extract_frames(
            args.source, source_dir,
            duration=duration, fps=fps, width=width, height=height
        )
        print(f"  {len(source_frames)} frames")
        print(f"  FPS: {fps}")
        
        # Process
        output_frames_dir = os.path.join(work_dir, "output")
        n_frames = process_with_optical_flow(
            source_frames, processed_frames, output_frames_dir,
            blend_alpha=args.blend, flow_method=args.method
        )
        
        # Assemble
        print("\nAssembling video...")
        assemble_video(output_frames_dir, args.output, fps)
        
        print("\n" + "=" * 60)
        print("✓ COMPLETE!")
        print("=" * 60)
        print(f"Output: {args.output}")
        print("=" * 60)
        
    finally:
        if not args.keep_temp and os.path.exists(work_dir):
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()

