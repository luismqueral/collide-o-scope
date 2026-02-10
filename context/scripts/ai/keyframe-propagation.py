"""
keyframe-propagation.py - Style Propagation from Keyframes

Instead of processing every frame through SD (expensive and flickery),
this approach:
1. Processes only keyframes (every Nth frame) through SD
2. Propagates the style to in-between frames using optical flow

This gives near-perfect temporal coherence because in-between frames
are derived from keyframes rather than independently generated.

HOW IT WORKS:
-------------
Given keyframe interval of 5:

  Frame:    1    2    3    4    5    6    7    8    9   10
            K    .    .    .    K    .    .    .    K    .
            ↓                   ↓                   ↓
           SD                  SD                  SD
            ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
            K → warp → → →  K → warp → → →  K → warp → →

Between keyframes, we:
1. Calculate cumulative optical flow from the keyframe
2. Warp the keyframe style to each in-between frame
3. Optionally blend between prev and next keyframe warps

ADVANTAGES:
-----------
- Much fewer SD calls (1/5 with interval=5)
- Near-perfect temporal coherence
- Maintains motion from original video
- Style is consistent across frames

LIMITATIONS:
------------
- Large motions between keyframes can cause warping artifacts
- Very fast motion needs smaller keyframe intervals
- Occlusions (new objects appearing) can look odd

USAGE:
------
  # Process source video with keyframe propagation
  python keyframe-propagation.py \\
    --input source_video.mp4 \\
    --prompt "3d man, dreamcast graphics" \\
    --keyframe-interval 5 \\
    --output styled_video.mp4
"""

import os
import sys
import argparse
import subprocess
import shutil
import base64
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install opencv-python numpy pillow")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# OPTICAL FLOW FUNCTIONS
# =============================================================================

def calculate_flow(frame1_gray, frame2_gray):
    """
    Calculate dense optical flow between two grayscale frames.
    
    Uses Farneback algorithm for dense (per-pixel) flow estimation.
    """
    flow = cv2.calcOpticalFlowFarneback(
        frame1_gray, frame2_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow


def accumulate_flow(flows):
    """
    Accumulate multiple flow fields into a single cumulative flow.
    
    If we have flow from A→B and B→C, the cumulative flow A→C is
    approximately the sum (with proper warping of the second flow).
    
    For simplicity, we use direct addition which works well for
    small inter-frame motions.
    """
    if len(flows) == 0:
        return None
    
    cumulative = flows[0].copy()
    for flow in flows[1:]:
        cumulative += flow
    
    return cumulative


def warp_frame(frame, flow):
    """
    Warp a frame using an optical flow field.
    
    Creates a sampling map from the flow and uses cv2.remap for
    efficient bilinear interpolation.
    """
    h, w = flow.shape[:2]
    
    # Create coordinate grid
    flow_map = np.zeros((h, w, 2), dtype=np.float32)
    flow_map[:, :, 0] = np.arange(w)
    flow_map[:, :, 1] = np.arange(h)[:, np.newaxis]
    
    # Add flow
    flow_map[:, :, 0] += flow[:, :, 0]
    flow_map[:, :, 1] += flow[:, :, 1]
    
    # Remap
    warped = cv2.remap(
        frame,
        flow_map[:, :, 0],
        flow_map[:, :, 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return warped


# =============================================================================
# SD PROCESSING
# =============================================================================

def process_frame_fal(image_path, prompt, negative_prompt, strength, guidance, seed):
    """Process a single keyframe through fal.ai SDXL."""
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
                "guidance_scale": guidance,
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


def download_image(url, output_path):
    """Download image and save as PNG."""
    import urllib.request
    temp_path = output_path + '.tmp'
    urllib.request.urlretrieve(url, temp_path)
    
    img = Image.open(temp_path)
    img.save(output_path, 'PNG')
    os.remove(temp_path)


# =============================================================================
# FRAME I/O
# =============================================================================

def extract_frames(video_path, output_dir, fps=None):
    """Extract frames from video."""
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = ['ffmpeg', '-y', '-v', 'error', '-i', video_path]
    if fps:
        cmd.extend(['-vf', f'fps={fps}'])
    cmd.append(f'{output_dir}/frame_%05d.png')
    
    subprocess.run(cmd, check=True)
    
    return sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith('.png')
    ])


def get_video_fps(video_path):
    """Get video FPS."""
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

def process_with_keyframes(source_frames, output_dir, prompt, negative_prompt,
                           strength, guidance, seed, keyframe_interval=5,
                           rate_limit=0.5):
    """
    Process video using keyframe propagation.
    
    Algorithm:
    1. Identify keyframes (every Nth frame)
    2. Process keyframes through SD
    3. For in-between frames:
       a. Calculate cumulative flow from nearest keyframe
       b. Warp the keyframe's styled result
       c. Optionally blend between prev/next keyframe warps
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_frames = len(source_frames)
    keyframe_indices = list(range(0, n_frames, keyframe_interval))
    
    # Also add last frame as keyframe if not already
    if (n_frames - 1) not in keyframe_indices:
        keyframe_indices.append(n_frames - 1)
    
    print(f"Total frames: {n_frames}")
    print(f"Keyframe interval: {keyframe_interval}")
    print(f"Keyframes to process: {len(keyframe_indices)}")
    print(f"SD calls saved: {n_frames - len(keyframe_indices)} ({100*(n_frames-len(keyframe_indices))/n_frames:.0f}%)")
    print()
    
    # =========================================================================
    # STEP 1: Process keyframes through SD
    # =========================================================================
    print("STEP 1: Processing keyframes through SD...")
    
    keyframe_results = {}  # idx -> processed frame path
    
    for i, kf_idx in enumerate(keyframe_indices):
        print(f"  Keyframe {i+1}/{len(keyframe_indices)} (frame {kf_idx+1})", end="", flush=True)
        
        # Process through SD
        url = process_frame_fal(
            source_frames[kf_idx],
            prompt, negative_prompt,
            strength, guidance, seed
        )
        
        if url:
            output_path = os.path.join(output_dir, f'keyframe_{kf_idx:05d}.png')
            download_image(url, output_path)
            keyframe_results[kf_idx] = output_path
            print(" ✓")
        else:
            print(" ✗ (will use source)")
            keyframe_results[kf_idx] = source_frames[kf_idx]
        
        if rate_limit > 0 and i < len(keyframe_indices) - 1:
            time.sleep(rate_limit)
    
    # =========================================================================
    # STEP 2: Pre-compute optical flow between all consecutive frames
    # =========================================================================
    print("\nSTEP 2: Computing optical flow...")
    
    flows = []  # flows[i] = flow from frame i to frame i+1
    
    prev_gray = cv2.cvtColor(cv2.imread(source_frames[0]), cv2.COLOR_BGR2GRAY)
    
    for i in range(1, n_frames):
        curr_gray = cv2.cvtColor(cv2.imread(source_frames[i]), cv2.COLOR_BGR2GRAY)
        flow = calculate_flow(prev_gray, curr_gray)
        flows.append(flow)
        prev_gray = curr_gray
        
        if (i + 1) % 20 == 0:
            print(f"  Computed flow for {i + 1}/{n_frames} frames")
    
    print(f"  Flow computed for all {n_frames} frames")
    
    # =========================================================================
    # STEP 3: Propagate style from keyframes to all frames
    # =========================================================================
    print("\nSTEP 3: Propagating style to all frames...")
    
    # Sort keyframe indices for finding prev/next
    sorted_kf = sorted(keyframe_indices)
    
    for frame_idx in range(n_frames):
        output_path = os.path.join(output_dir, f'frame_{frame_idx+1:05d}.png')
        
        # If this is a keyframe, just copy it
        if frame_idx in keyframe_results:
            styled = cv2.imread(keyframe_results[frame_idx])
            cv2.imwrite(output_path, styled)
            continue
        
        # Find nearest keyframes before and after this frame
        prev_kf = max([k for k in sorted_kf if k < frame_idx])
        next_kf_candidates = [k for k in sorted_kf if k > frame_idx]
        next_kf = min(next_kf_candidates) if next_kf_candidates else None
        
        # Load previous keyframe styled result
        prev_styled = cv2.imread(keyframe_results[prev_kf])
        
        # Accumulate flow from prev_kf to current frame
        # flows[i] goes from frame i to frame i+1
        cumulative_flows = flows[prev_kf:frame_idx]
        
        if cumulative_flows:
            cumulative_flow = accumulate_flow(cumulative_flows)
            warped = warp_frame(prev_styled, cumulative_flow)
        else:
            warped = prev_styled
        
        # Optionally blend with next keyframe if available
        if next_kf is not None and next_kf in keyframe_results:
            next_styled = cv2.imread(keyframe_results[next_kf])
            
            # Accumulate backward flow (negative) from next_kf to current
            backward_flows = flows[frame_idx:next_kf]
            if backward_flows:
                # Reverse the flows for backward warping
                backward_flow = accumulate_flow(backward_flows)
                backward_flow = -backward_flow  # Invert for backward warp
                warped_next = warp_frame(next_styled, backward_flow)
                
                # Blend based on position between keyframes
                t = (frame_idx - prev_kf) / (next_kf - prev_kf)
                warped = cv2.addWeighted(warped, 1.0 - t, warped_next, t, 0)
        
        cv2.imwrite(output_path, warped)
        
        if (frame_idx + 1) % 20 == 0:
            print(f"  Propagated {frame_idx + 1}/{n_frames} frames")
    
    print(f"  All {n_frames} frames complete")
    
    return n_frames


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Keyframe propagation for stable SD animations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE:
  python keyframe-propagation.py \\
    --input output/output_20251201_235110.mp4 \\
    --prompt "3d man, dreamcast graphics 1997 voodooFX nvidia" \\
    --keyframe-interval 5 \\
    --duration 10 --fps 14 \\
    --output output/keyframe_styled.mp4

KEYFRAME INTERVALS:
  3  - Very stable, more SD calls (33% of frames)
  5  - Balanced (20% of frames) [recommended]
  8  - Fewer calls, may have warp artifacts
  10 - Minimal calls, only for slow/static content
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input video')
    parser.add_argument('-o', '--output', required=True, help='Output video')
    parser.add_argument('-p', '--prompt', required=True, help='SD prompt')
    parser.add_argument('-n', '--negative-prompt', 
                        default='blurry, low quality, watermark')
    parser.add_argument('--keyframe-interval', '-k', type=int, default=5,
                        help='Process every Nth frame (default: 5)')
    parser.add_argument('--duration', type=float, default=10,
                        help='Duration in seconds (default: 10)')
    parser.add_argument('--fps', type=int, default=14,
                        help='Frame rate (default: 14)')
    parser.add_argument('--strength', '-s', type=float, default=0.5)
    parser.add_argument('--guidance', '-g', type=float, default=6.0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--rate-limit', type=float, default=0.5)
    parser.add_argument('--keep-temp', action='store_true')
    
    args = parser.parse_args()
    
    seed = args.seed or int(time.time()) % (2**31)
    
    print("=" * 60)
    print("KEYFRAME PROPAGATION")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Prompt: {args.prompt}")
    print(f"Keyframe interval: {args.keyframe_interval}")
    print(f"Strength: {args.strength}, Guidance: {args.guidance}")
    print(f"Seed: {seed}")
    print("=" * 60)
    
    work_dir = f"/tmp/keyframe_prop_{int(time.time())}"
    os.makedirs(work_dir, exist_ok=True)
    
    try:
        # Extract frames
        print("\nExtracting source frames...")
        source_dir = os.path.join(work_dir, "source")
        source_frames = extract_frames(
            args.input, source_dir
        )
        
        # Limit to duration
        total_frames = int(args.fps * args.duration)
        source_frames = source_frames[:total_frames]
        print(f"  Using {len(source_frames)} frames ({args.duration}s @ {args.fps}fps)")
        
        # Process
        output_frames_dir = os.path.join(work_dir, "output")
        process_with_keyframes(
            source_frames, output_frames_dir,
            args.prompt, args.negative_prompt,
            args.strength, args.guidance, seed,
            keyframe_interval=args.keyframe_interval,
            rate_limit=args.rate_limit
        )
        
        # Assemble
        print("\nAssembling video...")
        assemble_video(output_frames_dir, args.output, args.fps)
        
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

