"""
analog-processor.py - Add Analog Character to Digital Video

Adds organic imperfections that give digital video an analog feel:

1. FILM GRAIN
   - Random luminance noise that varies per-frame
   - Adjustable intensity and size
   - Optional color grain for RGB channels

2. MICRO-DISTORTION (Frame Breathing)
   - Subtle random scale variations per frame
   - Creates that organic "breathing" quality
   - Like each frame was captured on slightly different equipment

3. Optional extras:
   - Slight random rotation
   - Subtle position drift
   - Vignette
   - Color drift

The goal is subtle imperfection - enough to feel analog,
not so much it's distracting.

USAGE:
------
  # Basic grain + breathing
  python analog-processor.py -i video.mp4 -o analog.mp4

  # Heavy grain, more breathing
  python analog-processor.py -i video.mp4 -o analog.mp4 \\
    --grain 0.15 --breathe 0.03

  # Full analog treatment
  python analog-processor.py -i video.mp4 -o analog.mp4 \\
    --grain 0.12 --breathe 0.02 --vignette --color-drift
"""

import os
import sys
import argparse
import subprocess
import shutil
import random
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageFilter, ImageDraw
except ImportError:
    print("Error: PIL and numpy required")
    print("Install with: pip install pillow numpy")
    sys.exit(1)


# =============================================================================
# GRAIN GENERATION - Multiple Algorithms
# =============================================================================

def generate_perlin_noise(width, height, scale=8):
    """
    Generate Perlin-like noise using multiple octaves of sine waves.
    
    Creates smooth, organic-looking noise that clumps together
    like real film grain emulsion.
    """
    x = np.linspace(0, scale, width)
    y = np.linspace(0, scale, height)
    X, Y = np.meshgrid(x, y)
    
    # Multiple octaves for more natural look
    noise = np.zeros((height, width))
    for octave in range(4):
        freq = 2 ** octave
        amp = 0.5 ** octave
        # Random phase for this octave
        phase_x = np.random.uniform(0, 2 * np.pi)
        phase_y = np.random.uniform(0, 2 * np.pi)
        noise += amp * np.sin(freq * X + phase_x) * np.sin(freq * Y + phase_y)
    
    # Normalize to -1 to 1
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 2 - 1
    return noise


def generate_poisson_noise(frame, intensity=0.1):
    """
    Generate Poisson (shot) noise - intensity dependent.
    
    Brighter areas get more noise, darker areas get less.
    This mimics real photographic/sensor noise.
    """
    # Convert to float
    frame_float = frame.astype(np.float32) / 255.0
    
    # Poisson noise is proportional to signal level
    # Scale factor controls intensity
    scale = 1.0 / (intensity * 10 + 0.01)
    
    # Generate Poisson samples
    noisy = np.random.poisson(frame_float * scale) / scale
    
    # Return the noise component (difference from original)
    noise = noisy - frame_float
    return noise


def generate_salt_pepper_noise(width, height, density=0.02):
    """
    Generate salt and pepper noise - random black/white speckles.
    
    Creates sharp digital-looking artifacts, like dust or 
    transmission errors.
    """
    noise = np.zeros((height, width))
    
    # Salt (white speckles)
    salt_mask = np.random.random((height, width)) < density / 2
    noise[salt_mask] = 1.0
    
    # Pepper (black speckles)
    pepper_mask = np.random.random((height, width)) < density / 2
    noise[pepper_mask] = -1.0
    
    return noise


def generate_blue_noise(width, height, intensity=0.1):
    """
    Generate blue noise - evenly distributed, less clumpy.
    
    High-frequency noise that looks more refined than Gaussian.
    Used in high-quality dithering.
    """
    # Start with white noise
    noise = np.random.normal(0, intensity, (height, width))
    
    # High-pass filter to make it "blue" (high frequency)
    from PIL import ImageFilter
    noise_img = Image.fromarray(((noise + 0.5) * 255).clip(0, 255).astype(np.uint8))
    
    # Apply edge enhancement (high-pass)
    noise_img = noise_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    noise = (np.array(noise_img).astype(np.float32) / 255.0) - 0.5
    noise = noise * intensity / (np.std(noise) + 0.001)  # Normalize intensity
    
    return noise


def generate_grain(width, height, intensity=0.1, grain_size=1, color_grain=False, algorithm='gaussian'):
    """
    Generate film grain using various algorithms.
    
    Args:
        width: Frame width
        height: Frame height
        intensity: Grain strength (0.0-0.3 typical, 0.1 recommended)
        grain_size: Size of grain particles (1=fine, 2-3=coarse)
        color_grain: If True, different noise per RGB channel
        algorithm: 'gaussian', 'perlin', 'salt_pepper', 'blue'
        
    Returns:
        Grain array of shape (height, width, 3) with values centered at 0
    """
    # Calculate working dimensions (for coarse grain)
    work_h = height // grain_size if grain_size > 1 else height
    work_w = width // grain_size if grain_size > 1 else width
    
    # Generate noise based on algorithm
    if algorithm == 'perlin':
        # Perlin: smooth, organic clumps
        if color_grain:
            grain = np.stack([
                generate_perlin_noise(work_w, work_h, scale=8) * intensity,
                generate_perlin_noise(work_w, work_h, scale=8) * intensity,
                generate_perlin_noise(work_w, work_h, scale=8) * intensity,
            ], axis=2)
        else:
            mono = generate_perlin_noise(work_w, work_h, scale=8) * intensity
            grain = np.stack([mono, mono, mono], axis=2)
            
    elif algorithm == 'salt_pepper':
        # Salt & pepper: sharp speckles
        if color_grain:
            grain = np.stack([
                generate_salt_pepper_noise(work_w, work_h, intensity) * 0.5,
                generate_salt_pepper_noise(work_w, work_h, intensity) * 0.5,
                generate_salt_pepper_noise(work_w, work_h, intensity) * 0.5,
            ], axis=2)
        else:
            mono = generate_salt_pepper_noise(work_w, work_h, intensity) * 0.5
            grain = np.stack([mono, mono, mono], axis=2)
            
    elif algorithm == 'blue':
        # Blue noise: evenly distributed
        if color_grain:
            grain = np.stack([
                generate_blue_noise(work_w, work_h, intensity),
                generate_blue_noise(work_w, work_h, intensity),
                generate_blue_noise(work_w, work_h, intensity),
            ], axis=2)
        else:
            mono = generate_blue_noise(work_w, work_h, intensity)
            grain = np.stack([mono, mono, mono], axis=2)
            
    else:  # 'gaussian' (default)
        # Gaussian: classic random noise
        if color_grain:
            grain = np.random.normal(0, intensity, (work_h, work_w, 3))
        else:
            mono = np.random.normal(0, intensity, (work_h, work_w, 1))
            grain = np.repeat(mono, 3, axis=2)
    
    # Upscale if using coarse grain
    if grain_size > 1:
        grain_img = Image.fromarray(((grain + 0.5) * 255).clip(0, 255).astype(np.uint8))
        grain_img = grain_img.resize((width, height), Image.NEAREST)
        grain = (np.array(grain_img).astype(np.float32) / 255.0) - 0.5
    
    return grain.astype(np.float32)


def apply_grain(frame, grain):
    """
    Apply grain to a frame using additive blending.
    
    Grain is added to the image, then clipped to valid range.
    This simulates how film grain appears brighter in shadows
    and darker in highlights.
    
    Args:
        frame: RGB frame as numpy array (0-255)
        grain: Grain array centered at 0
        
    Returns:
        Grained frame as numpy array (0-255)
    """
    # Convert to float for arithmetic
    frame_float = frame.astype(np.float32) / 255.0
    
    # Add grain
    grained = frame_float + grain
    
    # Clip and convert back
    grained = np.clip(grained, 0, 1)
    return (grained * 255).astype(np.uint8)


# =============================================================================
# MICRO-DISTORTION (Frame Breathing)
# =============================================================================

def apply_breathing(frame, scale_variation=0.02, rotation_variation=0.0, 
                    position_variation=0.0, seed=None):
    """
    Apply subtle per-frame distortion for organic "breathing" effect.
    
    This simulates imperfections in analog capture/playback:
    - Slight zoom variations (lens breathing, tape stretch)
    - Tiny rotations (mechanical wobble)
    - Position drift (gate weave in film projectors)
    
    Args:
        frame: PIL Image
        scale_variation: Max scale change (0.02 = ±2% zoom)
        rotation_variation: Max rotation in degrees (0.5 typical)
        position_variation: Max position shift as fraction of size (0.01 = ±1%)
        seed: Random seed for reproducibility
        
    Returns:
        Distorted PIL Image
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    width, height = frame.size
    
    # Calculate random distortions
    # Using Gaussian distribution so most frames are near-normal
    # with occasional larger variations
    
    # Scale: slight random zoom (breathing)
    # Positive = zoom in, negative = zoom out
    scale_offset = np.random.normal(0, scale_variation / 2)
    scale = 1.0 + np.clip(scale_offset, -scale_variation, scale_variation)
    
    # Rotation: tiny random tilt
    if rotation_variation > 0:
        rotation = np.random.normal(0, rotation_variation / 2)
        rotation = np.clip(rotation, -rotation_variation, rotation_variation)
    else:
        rotation = 0
    
    # Position: slight drift from center
    if position_variation > 0:
        dx = np.random.normal(0, position_variation * width / 2)
        dy = np.random.normal(0, position_variation * height / 2)
        dx = np.clip(dx, -position_variation * width, position_variation * width)
        dy = np.clip(dy, -position_variation * height, position_variation * height)
    else:
        dx, dy = 0, 0
    
    # Apply transformations
    # We scale around center, then shift
    
    # Calculate new size for scaling
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize
    transformed = frame.resize((new_width, new_height), Image.LANCZOS)
    
    # Rotate if specified
    if rotation != 0:
        transformed = transformed.rotate(rotation, Image.BILINEAR, expand=False)
    
    # Crop/pad back to original size from center
    result = Image.new('RGB', (width, height), (0, 0, 0))
    
    # Calculate paste position (center the transformed image, then apply drift)
    paste_x = (width - new_width) // 2 + int(dx)
    paste_y = (height - new_height) // 2 + int(dy)
    
    # Handle cropping if scaled up
    if scale > 1:
        # Crop from center of larger image
        crop_x = (new_width - width) // 2 - int(dx)
        crop_y = (new_height - height) // 2 - int(dy)
        crop_x = max(0, min(crop_x, new_width - width))
        crop_y = max(0, min(crop_y, new_height - height))
        result = transformed.crop((crop_x, crop_y, crop_x + width, crop_y + height))
    else:
        # Paste smaller image onto canvas
        result.paste(transformed, (paste_x, paste_y))
    
    return result


# =============================================================================
# ADDITIONAL ANALOG EFFECTS
# =============================================================================

def apply_vignette(frame, strength=0.3):
    """
    Apply vignette (darkened corners) effect.
    
    Vignette occurs in real lenses due to light falloff at edges.
    Creates a subtle focus-drawing effect toward the center.
    
    Args:
        frame: PIL Image
        strength: Vignette intensity (0.0-0.5 typical)
        
    Returns:
        Vignetted PIL Image
    """
    width, height = frame.size
    
    # Create radial gradient mask
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Distance from center (normalized)
    distance = np.sqrt(X**2 + Y**2)
    
    # Vignette falloff (smooth curve from center to edges)
    # Using cosine for smooth falloff
    vignette = 1.0 - (distance * strength)
    vignette = np.clip(vignette, 0, 1)
    
    # Apply to frame
    frame_array = np.array(frame).astype(np.float32)
    vignette_3d = np.stack([vignette] * 3, axis=2)
    vignetted = frame_array * vignette_3d
    
    return Image.fromarray(vignetted.astype(np.uint8))


def apply_color_drift(frame, drift_amount=0.02):
    """
    Apply subtle color channel drift.
    
    Simulates chromatic aberration and color timing drift
    in analog video. Slightly shifts each color channel.
    
    Args:
        frame: PIL Image
        drift_amount: Max channel shift as fraction of size
        
    Returns:
        Color-drifted PIL Image
    """
    width, height = frame.size
    r, g, b = frame.split()
    
    # Random shifts for each channel (R and B drift, G stays centered)
    r_shift = int(np.random.uniform(-drift_amount, drift_amount) * width)
    b_shift = int(np.random.uniform(-drift_amount, drift_amount) * width)
    
    # Shift channels horizontally
    def shift_channel(channel, shift):
        arr = np.array(channel)
        if shift > 0:
            shifted = np.pad(arr, ((0, 0), (shift, 0)), mode='edge')[:, :-shift]
        elif shift < 0:
            shifted = np.pad(arr, ((0, 0), (0, -shift)), mode='edge')[:, -shift:]
        else:
            shifted = arr
        return Image.fromarray(shifted)
    
    r_shifted = shift_channel(r, r_shift)
    b_shifted = shift_channel(b, b_shift)
    
    return Image.merge('RGB', (r_shifted, g, b_shifted))


# =============================================================================
# FRAME I/O
# =============================================================================

def extract_frames(video_path, output_dir):
    """Extract all frames from video."""
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-i', video_path,
        f'{output_dir}/frame_%05d.png'
    ]
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

def process_analog(input_frames, output_dir, grain_intensity=0.1, grain_size=1,
                   color_grain=False, grain_algorithm='gaussian', 
                   breathe_scale=0.02, breathe_rotation=0.0,
                   breathe_position=0.0, vignette_strength=0.0, color_drift=0.0,
                   base_seed=None):
    """
    Apply analog processing to a sequence of frames.
    
    Each frame gets:
    1. Unique grain pattern (different random seed)
    2. Unique breathing distortion (from base_seed + frame number)
    3. Optional vignette and color drift
    """
    os.makedirs(output_dir, exist_ok=True)
    n_frames = len(input_frames)
    
    print(f"Processing {n_frames} frames with analog effects...")
    print(f"  Grain: {grain_intensity} (size {grain_size}, algo: {grain_algorithm})")
    print(f"  Breathing: scale ±{breathe_scale*100:.1f}%")
    if vignette_strength > 0:
        print(f"  Vignette: {vignette_strength}")
    if color_drift > 0:
        print(f"  Color drift: {color_drift}")
    print()
    
    for i, frame_path in enumerate(input_frames):
        # Load frame
        frame = Image.open(frame_path).convert('RGB')
        width, height = frame.size
        
        # Generate unique seed for this frame's randomness
        frame_seed = (base_seed or 0) + i
        
        # ---------------------------------------------------------------------
        # 1. Apply breathing (scale/rotation/position variation)
        # ---------------------------------------------------------------------
        if breathe_scale > 0 or breathe_rotation > 0 or breathe_position > 0:
            frame = apply_breathing(
                frame,
                scale_variation=breathe_scale,
                rotation_variation=breathe_rotation,
                position_variation=breathe_position,
                seed=frame_seed
            )
        
        # ---------------------------------------------------------------------
        # 2. Apply vignette
        # ---------------------------------------------------------------------
        if vignette_strength > 0:
            frame = apply_vignette(frame, vignette_strength)
        
        # ---------------------------------------------------------------------
        # 3. Apply color drift
        # ---------------------------------------------------------------------
        if color_drift > 0:
            random.seed(frame_seed + 1000)
            np.random.seed(frame_seed + 1000)
            frame = apply_color_drift(frame, color_drift)
        
        # ---------------------------------------------------------------------
        # 4. Apply grain (last, so it's on top of everything)
        # ---------------------------------------------------------------------
        if grain_intensity > 0:
            random.seed(frame_seed + 2000)
            np.random.seed(frame_seed + 2000)
            
            frame_array = np.array(frame)
            grain = generate_grain(width, height, grain_intensity, grain_size, color_grain, grain_algorithm)
            frame_array = apply_grain(frame_array, grain)
            frame = Image.fromarray(frame_array)
        
        # Save processed frame
        output_path = os.path.join(output_dir, f'frame_{i+1:05d}.png')
        frame.save(output_path, 'PNG')
        
        # Progress
        if (i + 1) % 20 == 0 or i == n_frames - 1:
            print(f"  Processed {i + 1}/{n_frames}")
    
    return n_frames


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Add analog character to digital video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Subtle analog treatment
  python analog-processor.py -i video.mp4 -o analog.mp4

  # Heavy grain, strong breathing
  python analog-processor.py -i video.mp4 -o analog.mp4 \\
    --grain 0.15 --breathe 0.03

  # Full analog treatment (grain + breathing + vignette + color drift)
  python analog-processor.py -i video.mp4 -o analog.mp4 \\
    --grain 0.12 --breathe 0.02 --vignette 0.3 --color-drift 0.01

GRAIN VALUES:
  0.05 = Very subtle, barely visible
  0.10 = Light film grain (recommended)
  0.15 = Medium, clearly visible
  0.20 = Heavy, stylized look

BREATHE VALUES (scale variation):
  0.01 = Very subtle (±1% zoom)
  0.02 = Noticeable but natural (±2%)
  0.03 = Obvious, dreamlike (±3%)
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input video')
    parser.add_argument('-o', '--output', required=True, help='Output video')
    
    # Grain options
    parser.add_argument('--grain', type=float, default=0.1,
                        help='Grain intensity (default: 0.1)')
    parser.add_argument('--grain-size', type=int, default=1,
                        help='Grain particle size, 1=fine, 2-3=coarse (default: 1)')
    parser.add_argument('--grain-algo', type=str, default='gaussian',
                        choices=['gaussian', 'perlin', 'salt_pepper', 'blue'],
                        help='Grain algorithm (default: gaussian)')
    parser.add_argument('--color-grain', action='store_true',
                        help='Use chromatic (colored) grain')
    
    # Breathing options
    parser.add_argument('--breathe', type=float, default=0.02,
                        help='Scale breathing amount (default: 0.02 = ±2%%)')
    parser.add_argument('--breathe-rotation', type=float, default=0.0,
                        help='Rotation variation in degrees (default: 0)')
    parser.add_argument('--breathe-position', type=float, default=0.0,
                        help='Position drift amount (default: 0)')
    
    # Additional effects
    parser.add_argument('--vignette', type=float, nargs='?', const=0.3, default=0.0,
                        help='Add vignette (default strength: 0.3)')
    parser.add_argument('--color-drift', type=float, nargs='?', const=0.005, default=0.0,
                        help='Add color channel drift (default: 0.005)')
    
    # Other options
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--keep-temp', action='store_true')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ANALOG PROCESSOR")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Grain: {args.grain} ({args.grain_algo})")
    print(f"Breathing: ±{args.breathe*100:.1f}% scale variation")
    if args.vignette:
        print(f"Vignette: {args.vignette}")
    if args.color_drift:
        print(f"Color drift: {args.color_drift}")
    print("=" * 60)
    
    import time
    work_dir = f"/tmp/analog_proc_{int(time.time())}"
    os.makedirs(work_dir, exist_ok=True)
    
    try:
        # Extract frames
        print("\nExtracting frames...")
        input_dir = os.path.join(work_dir, "input")
        input_frames = extract_frames(args.input, input_dir)
        fps = get_video_fps(args.input)
        print(f"  {len(input_frames)} frames at {fps} fps")
        
        # Process
        output_frames_dir = os.path.join(work_dir, "output")
        process_analog(
            input_frames, output_frames_dir,
            grain_intensity=args.grain,
            grain_size=args.grain_size,
            color_grain=args.color_grain,
            grain_algorithm=args.grain_algo,
            breathe_scale=args.breathe,
            breathe_rotation=args.breathe_rotation,
            breathe_position=args.breathe_position,
            vignette_strength=args.vignette,
            color_drift=args.color_drift,
            base_seed=args.seed
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

