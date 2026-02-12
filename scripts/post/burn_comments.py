"""
burn_comments.py - overlay comment text into videos

pulls comment strings and renders them as faded text overlays.
uses Pillow to render text onto transparent PNG frames, then
composites them via ffmpeg's overlay filter. works with any
ffmpeg build — no drawtext/freetype dependency.

the style is lo-fi: low opacity, monospace, small, randomized
position. not subtitles — more like graffiti on the signal.
something you'd notice on a second watch.

comments from the previous burst's videos get burned into the
next batch, so there's always a one-cycle delay between posting
a comment and seeing it appear.

usage:
    python scripts/post/burn_comments.py --input video.mp4 --comments "first comment" "second comment"
    python scripts/post/burn_comments.py --input video.mp4 --comments "hello" --output out.mp4
    python scripts/post/burn_comments.py --input video.mp4 --comments-file comments.json
    python scripts/post/burn_comments.py --input video.mp4 --comments "test" --dry-run

requires:
    pip install Pillow
    ffmpeg (any build — uses overlay filter, not drawtext)
"""

import os
import sys
import json
import subprocess
import argparse
import random
import tempfile
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def get_video_info(video_path):
    """get duration and resolution via ffprobe."""
    info = {'duration': 0, 'width': 1920, 'height': 1080, 'fps': 30}
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-show_entries', 'stream=width,height,r_frame_rate',
                '-select_streams', 'v:0',
                '-of', 'json',
                video_path
            ],
            capture_output=True, text=True
        )
        import json as jsonmod
        data = jsonmod.loads(result.stdout)
        info['duration'] = float(data.get('format', {}).get('duration', 0))

        streams = data.get('streams', [])
        if streams:
            info['width'] = int(streams[0].get('width', 1920))
            info['height'] = int(streams[0].get('height', 1080))
            fps_str = streams[0].get('r_frame_rate', '30/1')
            if '/' in fps_str:
                num, den = fps_str.split('/')
                info['fps'] = float(num) / float(den)
            else:
                info['fps'] = float(fps_str)
    except Exception:
        pass
    return info


def _plan_comments(comments, duration, rng):
    """
    plan the timing, position, and style for each comment overlay.
    returns a list of dicts describing each comment's appearance.
    """
    plans = []

    color_choices = [
        (204, 204, 204),  # neutral grey
        (212, 200, 184),  # warm parchment
        (184, 200, 212),  # cool blue-grey
        (200, 212, 184),  # faded green
        (212, 184, 200),  # dusty pink
        (255, 255, 255),  # white
        (160, 160, 160),  # darker grey
    ]

    for comment_text in comments:
        text = comment_text.strip()
        if len(text) > 80:
            text = text[:77] + '...'

        visible_duration = rng.uniform(3.0, 8.0)
        margin = 2.0
        max_start = max(margin, duration - visible_duration - margin)
        start_time = rng.uniform(margin, max_start)
        end_time = start_time + visible_duration

        fade_in = rng.uniform(0.5, 1.5)
        fade_out = rng.uniform(0.5, 1.5)
        max_alpha = rng.uniform(0.12, 0.28)

        x_frac = rng.uniform(0.05, 0.7)
        y_frac = rng.uniform(0.1, 0.85)
        fontsize_frac = rng.uniform(0.035, 0.055)
        color = rng.choice(color_choices)

        plans.append({
            'text': text,
            'start': start_time,
            'end': end_time,
            'fade_in': fade_in,
            'fade_out': fade_out,
            'max_alpha': max_alpha,
            'x_frac': x_frac,
            'y_frac': y_frac,
            'fontsize_frac': fontsize_frac,
            'color': color,
        })

    return plans


def _render_overlay_video(plans, width, height, fps, duration, tmp_dir):
    """
    render a transparent overlay video with all comment text.

    uses Pillow to draw text onto RGBA frames, writes them as PNGs,
    then assembles into a video with ffmpeg. the result is a
    transparent video that gets composited over the source.

    this is the heavy part — but it only renders frames where
    comments are actually visible (sparse), and the overlay video
    is short-lived in /tmp.
    """
    from PIL import Image, ImageDraw, ImageFont

    # find a monospace font — try common system locations
    mono_font_paths = [
        '/System/Library/Fonts/Menlo.ttc',
        '/System/Library/Fonts/SFMono-Regular.otf',
        '/System/Library/Fonts/Courier.dfont',
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
        '/usr/share/fonts/TTF/DejaVuSansMono.ttf',
    ]

    def get_font(size):
        for path in mono_font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        # fallback to default (not monospace but works)
        return ImageFont.load_default()

    # figure out which frames actually need rendering.
    # most frames are fully transparent — only render frames
    # where at least one comment is visible.
    total_frames = int(duration * fps)
    frame_dir = os.path.join(tmp_dir, 'frames')
    os.makedirs(frame_dir, exist_ok=True)

    # pre-compute which frames each comment is active on
    comment_ranges = []
    for p in plans:
        start_frame = int(p['start'] * fps)
        end_frame = int(p['end'] * fps)
        comment_ranges.append((start_frame, end_frame))

    # find all frames that need rendering
    active_frames = set()
    for start_f, end_f in comment_ranges:
        for f in range(max(0, start_f), min(total_frames, end_f + 1)):
            active_frames.add(f)

    print(f"    rendering {len(active_frames)} overlay frames "
          f"({len(active_frames) / total_frames * 100:.0f}% of {total_frames} total)...")

    # render each frame
    for frame_num in range(total_frames):
        frame_path = os.path.join(frame_dir, f'{frame_num:06d}.png')

        if frame_num not in active_frames:
            # fully transparent — write a tiny 1x1 transparent PNG
            # and we'll scale it up in ffmpeg. actually, ffmpeg's
            # image2 demuxer needs consistent sizes, so write full size
            # but only for active frames. for inactive frames, symlink
            # to a shared blank.
            continue

        # create RGBA frame
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        t = frame_num / fps  # current time in seconds

        for plan, (start_f, end_f) in zip(plans, comment_ranges):
            if frame_num < start_f or frame_num > end_f:
                continue

            # compute alpha for this frame (fade in/hold/fade out)
            if t < plan['start']:
                alpha = 0
            elif t < plan['start'] + plan['fade_in']:
                alpha = plan['max_alpha'] * (t - plan['start']) / plan['fade_in']
            elif t < plan['end'] - plan['fade_out']:
                alpha = plan['max_alpha']
            elif t < plan['end']:
                alpha = plan['max_alpha'] * (1 - (t - (plan['end'] - plan['fade_out'])) / plan['fade_out'])
            else:
                alpha = 0

            if alpha <= 0.005:
                continue

            fontsize = max(12, int(plan['fontsize_frac'] * height))
            font = get_font(fontsize)

            # measure the text so we can clamp it inside the frame
            bbox = draw.textbbox((0, 0), plan['text'], font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            pad = int(width * 0.03)
            x = int(plan['x_frac'] * width)
            y = int(plan['y_frac'] * height)

            # keep the text fully inside the frame
            x = max(pad, min(x, width - text_w - pad))
            y = max(pad, min(y, height - text_h - pad))

            r, g, b = plan['color']
            a = int(alpha * 255)

            draw.text((x, y), plan['text'], fill=(r, g, b, a), font=font)

        img.save(frame_path, 'PNG')

    # write a blank frame for gaps
    blank_path = os.path.join(frame_dir, 'blank.png')
    blank = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    blank.save(blank_path, 'PNG')

    # fill in missing frames with symlinks to blank
    for frame_num in range(total_frames):
        frame_path = os.path.join(frame_dir, f'{frame_num:06d}.png')
        if not os.path.exists(frame_path):
            os.symlink(blank_path, frame_path)

    # assemble into a video with alpha channel (yuva420p or rgba)
    overlay_path = os.path.join(tmp_dir, 'overlay.mov')

    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frame_dir, '%06d.png'),
        '-c:v', 'png',  # lossless with alpha
        '-pix_fmt', 'rgba',
        overlay_path,
    ]

    print(f"    encoding overlay video...")
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    return overlay_path


def build_drawtext_filters(comments, duration, rng=None):
    """
    build comment placement plans (kept for API compatibility
    with the simulation/preview code).

    returns a list of plan dicts describing each comment's appearance.
    """
    if rng is None:
        rng = random.Random()
    return _plan_comments(comments, duration, rng)


def burn_comments(input_path, comments, output_path=None, rng=None, dry_run=False):
    """
    overlay comment text into a video file.

    renders text as transparent PNG frames via Pillow, assembles
    into an overlay video, then composites over the source with
    ffmpeg's overlay filter. works with any ffmpeg build.

    if output_path is None, overwrites the input file (renders to a
    temp file first, then replaces). this is the mode the autopilot uses.

    Args:
        input_path: path to the source MP4
        comments: list of comment text strings (1-5 recommended)
        output_path: path for the output MP4 (or None to overwrite input)
        rng: random.Random instance
        dry_run: if True, print the command but don't run it

    Returns:
        output path on success, None on failure
    """
    if not comments:
        return input_path

    if rng is None:
        rng = random.Random()

    info = get_video_info(input_path)
    duration = info['duration']
    if duration <= 0:
        print(f"  error: could not read duration of {input_path}")
        return None

    plans = _plan_comments(comments, duration, rng)
    if not plans:
        return input_path

    if dry_run:
        print(f"  [dry run] would burn {len(comments)} comment(s) into {os.path.basename(input_path)}")
        for p in plans:
            print(f"    \"{p['text'][:50]}\" at ({p['x_frac']:.0%}, {p['y_frac']:.0%}), "
                  f"{p['start']:.0f}-{p['end']:.0f}s, {p['max_alpha']:.0%} opacity")
        return input_path

    print(f"  burning {len(comments)} comment(s) into {os.path.basename(input_path)}...")

    # if no output path, we'll render to a temp file and replace the input
    overwrite = output_path is None
    if overwrite:
        output_path = input_path + '.tmp.mp4'

    with tempfile.TemporaryDirectory(prefix='burn_comments_') as tmp_dir:
        try:
            # render the transparent overlay video
            overlay_path = _render_overlay_video(
                plans,
                info['width'], info['height'],
                info['fps'], duration,
                tmp_dir,
            )

            # composite overlay onto the source video
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-i', overlay_path,
                '-filter_complex', '[0:v][1:v]overlay=0:0:shortest=1',
                '-c:v', 'libx264',
                '-crf', '20',
                '-preset', 'fast',
                '-c:a', 'copy',
                '-map', '0:a?',
                output_path,
            ]

            print(f"    compositing...")
            subprocess.run(cmd, check=True, capture_output=True, text=True)

        except subprocess.CalledProcessError as e:
            print(f"  burn failed: {e}")
            if e.stderr:
                lines = e.stderr.strip().split('\n')
                for line in lines[-5:]:
                    print(f"    {line}")
            if overwrite and os.path.exists(output_path):
                os.remove(output_path)
            return None
        except Exception as e:
            print(f"  burn failed: {e}")
            if overwrite and os.path.exists(output_path):
                os.remove(output_path)
            return None

    # if overwriting, swap the temp file in
    if overwrite:
        os.replace(output_path, input_path)
        output_path = input_path

    print(f"  done: {os.path.basename(output_path)}")
    return output_path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='burn comment text into videos as faded overlays',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/post/burn_comments.py --input video.mp4 --comments "first comment" "second comment"
  python scripts/post/burn_comments.py --input video.mp4 --comments-file comments.json
  python scripts/post/burn_comments.py --input video.mp4 --comments "test" --output out.mp4 --seed 42
        """
    )

    parser.add_argument('--input', required=True,
                        help='input MP4 file')
    parser.add_argument('--output', default=None,
                        help='output MP4 file (default: overwrite input)')

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--comments', nargs='+',
                        help='comment text strings to overlay')
    source.add_argument('--comments-file',
                        help='JSON file with a list of comment strings')

    parser.add_argument('--max-comments', type=int, default=5,
                        help='max comments to overlay (default: 5)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for reproducible placement')
    parser.add_argument('--dry-run', action='store_true',
                        help='show what would happen without rendering')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"error: file not found: {args.input}")
        sys.exit(1)

    # load comments
    if args.comments_file:
        with open(args.comments_file) as f:
            comments = json.load(f)
        if not isinstance(comments, list):
            print("error: comments file must be a JSON array of strings")
            sys.exit(1)
    else:
        comments = args.comments

    # limit comment count
    rng = random.Random(args.seed)
    if len(comments) > args.max_comments:
        comments = rng.sample(comments, args.max_comments)

    result = burn_comments(
        args.input,
        comments,
        output_path=args.output,
        rng=rng,
        dry_run=args.dry_run,
    )

    if result:
        print(f"\ndone: {result}")
    else:
        print("\nfailed")
        sys.exit(1)


if __name__ == '__main__':
    main()
