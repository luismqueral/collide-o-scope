"""
convert-to-vertical.py - Convert landscape videos to vertical (9:16) for Instagram Reels

Takes 16:9 horizontal videos and converts them to 9:16 vertical format
with creative background treatments to fill the extra space.

FILL MODES:
  blur      - Zoomed blurry version of the video behind (TikTok style)
  color     - Solid color (black, white, or extracted from video)
  gradient  - Vertical gradient using colors from the video
  mirror    - Mirrored/reflected video above and below
  tile      - Tiled pattern of the video
  vhs       - Retro VHS tracking lines effect
  noise     - Animated static/grain

USAGE:
    python convert-to-vertical.py input.mp4 output.mp4 --mode blur
    python convert-to-vertical.py input.mp4 output.mp4 --mode gradient
    python convert-to-vertical.py --batch input_dir/ output_dir/ --mode blur

OUTPUT:
    1080x1920 (standard Instagram Reels resolution)
"""

import subprocess
import argparse
import os
import sys
import random
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Output dimensions (9:16 vertical)
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920

# Video quality
CRF = 23  # Lower = better quality, bigger file
PRESET = 'medium'

# =============================================================================
# FILL MODE FILTERS
# =============================================================================

def get_blur_filter():
    """
    Blur mode: Zoomed, blurred version of the video as background.
    This is the classic TikTok/Reels treatment.
    """
    return f"""
    [0:v]split=2[bg][fg];
    [bg]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,
        crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},
        boxblur=20:5,
        eq=brightness=-0.1:saturation=0.8[bg_blur];
    [fg]scale=-1:{int(OUTPUT_HEIGHT * 0.5)}:force_original_aspect_ratio=decrease[fg_scaled];
    [bg_blur][fg_scaled]overlay=(W-w)/2:(H-h)/2
    """

def get_color_filter(color='black'):
    """
    Color mode: Solid color background.
    Options: black, white, or hex color like '0x1a1a2e'
    """
    if color == 'black':
        color_hex = '0x000000'
    elif color == 'white':
        color_hex = '0xffffff'
    else:
        color_hex = color
    
    return f"""
    color=c={color_hex}:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:d=1[bg];
    [0:v]scale=-1:{int(OUTPUT_HEIGHT * 0.5)}:force_original_aspect_ratio=decrease[fg];
    [bg][fg]overlay=(W-w)/2:(H-h)/2:shortest=1
    """

def get_gradient_filter():
    """
    Gradient mode: Vertical gradient background using video's colors.
    Creates a moody, stylized look.
    """
    # Generate random dark gradient colors for that synthwave vibe
    colors = [
        ('0x0f0c29', '0x302b63'),  # Deep purple
        ('0x000000', '0x434343'),  # Dark gray
        ('0x200122', '0x6f0000'),  # Deep red
        ('0x0f2027', '0x203a43'),  # Teal dark
        ('0x1a1a2e', '0x16213e'),  # Navy
        ('0x0d0d0d', '0x1a1a2e'),  # Almost black to navy
    ]
    c1, c2 = random.choice(colors)
    
    return f"""
    gradients=s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:
        c0={c1}:c1={c1}:c2={c2}:c3={c2}:
        x0=0:y0=0:x1=0:y1={OUTPUT_HEIGHT}:
        nb_colors=4:type=linear:d=1[bg];
    [0:v]scale=-1:{int(OUTPUT_HEIGHT * 0.5)}:force_original_aspect_ratio=decrease[fg];
    [bg][fg]overlay=(W-w)/2:(H-h)/2:shortest=1
    """

def get_gradient_filter_simple():
    """
    Simpler gradient using geq (works on more ffmpeg versions).
    """
    return f"""
    [0:v]scale=-1:{int(OUTPUT_HEIGHT * 0.5)}:force_original_aspect_ratio=decrease[fg];
    color=black:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:d=1,
        geq=r='40+Y/10':g='20+Y/15':b='60+Y/8'[bg];
    [bg][fg]overlay=(W-w)/2:(H-h)/2:shortest=1
    """

def get_mirror_filter():
    """
    Mirror mode: Reflected video above and below the main video.
    Creates a kaleidoscopic, psychedelic effect.
    """
    return f"""
    [0:v]split=3[top][mid][bot];
    [top]scale=-1:{int(OUTPUT_HEIGHT * 0.25)}:force_original_aspect_ratio=decrease,
        vflip,crop=iw:{int(OUTPUT_HEIGHT * 0.25)},
        setsar=1,eq=brightness=-0.2:saturation=0.7[top_mir];
    [mid]scale=-1:{int(OUTPUT_HEIGHT * 0.5)}:force_original_aspect_ratio=decrease[mid_scaled];
    [bot]scale=-1:{int(OUTPUT_HEIGHT * 0.25)}:force_original_aspect_ratio=decrease,
        vflip,crop=iw:{int(OUTPUT_HEIGHT * 0.25)},
        setsar=1,eq=brightness=-0.2:saturation=0.7[bot_mir];
    color=black:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:d=1[canvas];
    [canvas][top_mir]overlay=(W-w)/2:0[c1];
    [c1][mid_scaled]overlay=(W-w)/2:(H-h)/2[c2];
    [c2][bot_mir]overlay=(W-w)/2:H-h
    """

def get_tile_filter():
    """
    Tile mode: Small tiled versions of the video as background.
    Creates a busy, energetic pattern.
    """
    return f"""
    [0:v]split=2[bg][fg];
    [bg]scale={int(OUTPUT_WIDTH/3)}:-1,
        tile=3x6,
        crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},
        eq=brightness=-0.3:saturation=0.5[bg_tiled];
    [fg]scale=-1:{int(OUTPUT_HEIGHT * 0.5)}:force_original_aspect_ratio=decrease[fg_scaled];
    [bg_tiled][fg_scaled]overlay=(W-w)/2:(H-h)/2
    """

def get_vhs_filter():
    """
    VHS mode: Retro scan lines and tracking artifacts.
    Perfect for the Richard Cigarette aesthetic!
    """
    return f"""
    [0:v]split=2[bg][fg];
    [bg]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,
        crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},
        boxblur=8:3,
        noise=alls=20:allf=t+u,
        eq=brightness=-0.15:saturation=0.7:contrast=1.1,
        drawbox=x=0:y=mod(n*3\\,{OUTPUT_HEIGHT}):w={OUTPUT_WIDTH}:h=3:c=white@0.1:t=fill,
        drawbox=x=0:y=mod(n*7+100\\,{OUTPUT_HEIGHT}):w={OUTPUT_WIDTH}:h=2:c=white@0.15:t=fill[bg_vhs];
    [fg]scale=-1:{int(OUTPUT_HEIGHT * 0.5)}:force_original_aspect_ratio=decrease[fg_scaled];
    [bg_vhs][fg_scaled]overlay=(W-w)/2:(H-h)/2
    """

def get_noise_filter():
    """
    Noise mode: Animated static/grain background.
    Dark and moody.
    """
    return f"""
    color=c=0x0a0a0a:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:d=1,
        noise=alls=40:allf=t+u[bg];
    [0:v]scale=-1:{int(OUTPUT_HEIGHT * 0.5)}:force_original_aspect_ratio=decrease[fg];
    [bg][fg]overlay=(W-w)/2:(H-h)/2:shortest=1
    """

def get_dominant_color_filter():
    """
    Extract dominant color from video and use as background.
    Samples first frame, gets average color.
    """
    # This is handled specially - we extract color first, then apply
    return "EXTRACT_COLOR"

# =============================================================================
# COLOR EXTRACTION
# =============================================================================

def extract_dominant_color(video_path):
    """Extract the dominant/average color from the first frame of a video."""
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vframes', '1',
        '-vf', 'scale=1:1',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        'pipe:1'
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0 and len(result.stdout) >= 3:
        r, g, b = result.stdout[0], result.stdout[1], result.stdout[2]
        return f'0x{r:02x}{g:02x}{b:02x}'
    return '0x1a1a2e'  # Default dark blue

# =============================================================================
# CONVERSION
# =============================================================================

def convert_video(input_path, output_path, mode='blur'):
    """Convert a single video to vertical format."""
    
    print(f"Converting: {input_path}")
    print(f"Mode: {mode}")
    
    # Get the appropriate filter
    if mode == 'blur':
        vf = get_blur_filter()
    elif mode == 'color':
        vf = get_color_filter('black')
    elif mode == 'white':
        vf = get_color_filter('white')
    elif mode == 'gradient':
        vf = get_gradient_filter_simple()  # Use simpler version for compatibility
    elif mode == 'mirror':
        vf = get_mirror_filter()
    elif mode == 'tile':
        vf = get_tile_filter()
    elif mode == 'vhs':
        vf = get_vhs_filter()
    elif mode == 'noise':
        vf = get_noise_filter()
    elif mode == 'dominant':
        color = extract_dominant_color(input_path)
        print(f"Extracted color: {color}")
        vf = get_color_filter(color)
    elif mode == 'random':
        mode = random.choice(['blur', 'gradient', 'vhs', 'noise', 'mirror'])
        print(f"Random mode selected: {mode}")
        return convert_video(input_path, output_path, mode)
    else:
        print(f"Unknown mode: {mode}, using blur")
        vf = get_blur_filter()
    
    # Clean up filter (remove newlines, extra spaces)
    vf = ' '.join(vf.split())
    
    # Build ffmpeg command
    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-vf', vf,
        '-c:v', 'libx264',
        '-preset', PRESET,
        '-crf', str(CRF),
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        output_path
    ]
    
    print("Running ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    print(f"Success: {output_path}")
    return True

def batch_convert(input_dir, output_dir, mode='blur', limit=None):
    """Convert all MP4s in a directory."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    videos = sorted(input_path.glob('*.mp4'), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if limit:
        videos = videos[:limit]
    
    print(f"Found {len(videos)} videos to convert")
    
    success = 0
    for i, video in enumerate(videos):
        out_file = output_path / f"vertical_{video.name}"
        print(f"\n[{i+1}/{len(videos)}]")
        
        # Use random mode for variety if specified
        current_mode = mode
        if mode == 'random':
            current_mode = random.choice(['blur', 'gradient', 'vhs', 'noise', 'mirror'])
            print(f"Random mode: {current_mode}")
        
        if convert_video(str(video), str(out_file), current_mode):
            success += 1
    
    print(f"\n✅ Converted {success}/{len(videos)} videos")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert landscape videos to vertical (9:16) for Instagram Reels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
  blur      Zoomed blurry background (TikTok style)
  color     Solid black background  
  white     Solid white background
  gradient  Dark gradient background
  mirror    Mirrored reflections above/below
  tile      Tiled pattern background
  vhs       Retro VHS scan lines
  noise     Animated static grain
  dominant  Use video's dominant color
  random    Pick random mode per video

EXAMPLES:
  python convert-to-vertical.py video.mp4 vertical.mp4 --mode blur
  python convert-to-vertical.py --batch output/ vertical/ --mode random --limit 50
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input video file or directory (with --batch)')
    parser.add_argument('output', nargs='?', help='Output video file or directory (with --batch)')
    parser.add_argument('--mode', '-m', default='blur', 
                        choices=['blur', 'color', 'white', 'gradient', 'mirror', 'tile', 'vhs', 'noise', 'dominant', 'random'],
                        help='Fill mode (default: blur)')
    parser.add_argument('--batch', '-b', action='store_true', help='Batch convert directory')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of videos (batch mode)')
    
    args = parser.parse_args()
    
    if not args.input or not args.output:
        parser.print_help()
        sys.exit(1)
    
    if args.batch:
        batch_convert(args.input, args.output, args.mode, args.limit)
    else:
        convert_video(args.input, args.output, args.mode)

if __name__ == '__main__':
    main()




