"""
batch-render.py - render multiple videos in sequence

runs multi-layer.py N times with unique seeds each run.
each video gets its own random duration, layer selection, and
embedded metadata (title + description).

usage:
    python tools/batch-render.py --count 10
    python tools/batch-render.py --count 50 --preset classic-white
    python tools/batch-render.py --count 20 --preset kmeans-default --fps 24
    python tools/batch-render.py --count 10 --output-dir projects/my-project/output
"""

import os
import sys
import subprocess
import argparse
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render_one(index, total, preset=None, fps=None, num_videos=None,
               duration=None, mode=None, output_dir=None, project=None,
               seed=None):
    """
    run a single render by calling multi-layer.py as a subprocess.
    returns the output path on success, None on failure.
    """
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, 'scripts', 'blend', 'multi-layer.py'),
    ]

    if preset:
        cmd.extend(['--preset', preset])
    if fps:
        cmd.extend(['--fps', str(fps)])
    if num_videos:
        cmd.extend(['--num-videos', str(num_videos)])
    if duration:
        cmd.extend(['--duration', str(duration)])
    if mode:
        cmd.extend(['--mode', mode])
    if output_dir:
        cmd.extend(['--output-dir', output_dir])
    if project:
        cmd.extend(['--project', project])
    if seed is not None:
        cmd.extend(['--seed', str(seed)])

    print(f"\n{'='*60}")
    print(f"  video {index + 1} of {total}")
    print(f"{'='*60}")

    start = time.time()

    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        print(f"\n  completed in {mins}m{secs:02d}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='render multiple videos in sequence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python tools/batch-render.py --count 10
  python tools/batch-render.py --count 50 --preset classic-white
  python tools/batch-render.py --count 20 --fps 24 --num-videos 4
  python tools/batch-render.py --count 10 --output-dir projects/my-series/output
        """
    )

    parser.add_argument('--count', '-n', type=int, required=True,
                        help='number of videos to render')
    parser.add_argument('--preset', default=None,
                        help='preset to use for all renders')
    parser.add_argument('--fps', type=int, default=None,
                        help='frame rate for all renders')
    parser.add_argument('--num-videos', type=int, default=None,
                        help='number of layers per render')
    parser.add_argument('--duration', type=int, default=None,
                        help='exact duration in seconds (omit for random 1-3 min)')
    parser.add_argument('--mode', default=None,
                        choices=['fixed', 'kmeans', 'luminance', 'rembg', 'random'],
                        help='color keying mode')
    parser.add_argument('--output-dir', default=None,
                        help='output directory for all renders')
    parser.add_argument('--project', default=None,
                        help='project name')

    args = parser.parse_args()

    # default output to a timestamped batch folder if nothing specified
    if not args.output_dir and not args.project:
        from datetime import datetime
        batch_name = f"batch-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        args.output_dir = os.path.join(PROJECT_ROOT, 'projects', batch_name, 'output')

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nbatch render: {args.count} videos")
    if args.preset:
        print(f"preset: {args.preset}")
    if args.output_dir:
        print(f"output: {args.output_dir}")

    batch_start = time.time()
    successes = 0
    failures = 0

    for i in range(args.count):
        ok = render_one(
            i, args.count,
            preset=args.preset,
            fps=args.fps,
            num_videos=args.num_videos,
            duration=args.duration,
            mode=args.mode,
            output_dir=args.output_dir,
            project=args.project,
        )
        if ok:
            successes += 1
        else:
            failures += 1

    batch_elapsed = time.time() - batch_start
    batch_mins = int(batch_elapsed) // 60
    batch_secs = int(batch_elapsed) % 60

    print(f"\n{'='*60}")
    print(f"  batch complete")
    print(f"  {successes} succeeded, {failures} failed")
    print(f"  total time: {batch_mins}m{batch_secs:02d}s")
    if successes > 0:
        avg = batch_elapsed / successes
        avg_mins = int(avg) // 60
        avg_secs = int(avg) % 60
        print(f"  average: {avg_mins}m{avg_secs:02d}s per video")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
