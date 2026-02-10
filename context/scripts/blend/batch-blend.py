"""
batch-blend.py - Generate multiple blended videos in a single run

Uses blend-video-alt.py for advanced color keying modes (luminance, rembg, kmeans).

Usage:
    python scripts/blend/batch-blend.py --count 10 --output projects/my-project/output
    python scripts/blend/batch-blend.py -n 10 -o projects/my-project/output
    
    # Use simple mode (blend-video-multi-vid.py) instead:
    python scripts/blend/batch-blend.py -n 10 -o output --simple
"""

import argparse
import os
import sys
import datetime

# Add the parent directory to path so we can import the blend module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path setup
import importlib.util

def run_batch(count: int, output_directory: str, use_simple: bool = False):
    """
    Run the blend script multiple times.
    
    Args:
        count: Number of videos to generate
        output_directory: Where to save outputs
        use_simple: If True, use blend-video-multi-vid.py (simple black/white keying)
                   If False, use blend-video-alt.py (advanced ML color keying)
    """
    # Dynamically load the blend module
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if use_simple:
        blend_path = os.path.join(script_dir, 'blend-video-multi-vid.py')
        module_name = "blend_multi"
        mode_desc = "SIMPLE (black/white colorkey)"
    else:
        blend_path = os.path.join(script_dir, 'blend-video-alt.py')
        module_name = "blend_alt"
        mode_desc = "ADVANCED (luminance/rembg/kmeans colorkey)"
    
    spec = importlib.util.spec_from_file_location(module_name, blend_path)
    blend = importlib.util.module_from_spec(spec)
    
    # Override the output directory before loading
    spec.loader.exec_module(blend)
    blend.OUTPUT_DIRECTORY = output_directory
    
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"BATCH BLEND - Generating {count} videos")
    print(f"Mode: {mode_desc}")
    print(f"Output directory: {output_directory}")
    
    # Show color mode info for advanced mode
    if not use_simple and hasattr(blend, 'COLOR_MODE'):
        print(f"Color Mode: {blend.COLOR_MODE}")
        if blend.COLOR_MODE == 'random' and hasattr(blend, 'RANDOM_MODE_CHOICES'):
            print(f"Random choices: {blend.RANDOM_MODE_CHOICES}")
    
    # Show HD filter info
    if hasattr(blend, 'HD_ONLY'):
        print(f"HD Filter: {'ON' if blend.HD_ONLY else 'OFF'} (min {getattr(blend, 'MIN_RESOLUTION', 720)}p)")
    
    print(f"{'='*60}\n")
    
    successful = 0
    failed = 0
    
    for i in range(count):
        print(f"\n{'='*60}")
        print(f"VIDEO {i+1} of {count}")
        print(f"{'='*60}\n")
        
        try:
            blend.main()
            successful += 1
            print(f"\n✓ Video {i+1} completed successfully")
        except Exception as e:
            failed += 1
            print(f"\n✗ Video {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"Successful: {successful}/{count}")
    print(f"Failed: {failed}/{count}")
    print(f"Output: {output_directory}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate multiple blended videos in batch'
    )
    parser.add_argument(
        '-n', '--count',
        type=int,
        default=10,
        help='Number of videos to generate (default: 10)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output directory for generated videos'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Use simple mode (blend-video-multi-vid.py) instead of advanced mode'
    )
    
    args = parser.parse_args()
    run_batch(args.count, args.output, args.simple)


if __name__ == "__main__":
    main()
