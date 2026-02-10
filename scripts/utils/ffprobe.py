"""
ffprobe.py - Video metadata extraction via ffprobe

Thin wrappers around ffprobe for reading video file properties.
Every script that needs to know about a video's duration, resolution,
frame rate, or audio streams should import from here.

Requires ffprobe to be installed and on PATH (comes with ffmpeg).
"""

import subprocess


def get_video_duration(video_path):
    """
    Get the duration of a video file in seconds.

    Args:
        video_path: Path to the video file

    Returns:
        Duration in seconds as a float, or 0 if there's an error
    """
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        return float(result.stdout)
    except Exception as e:
        print(f"Error getting video duration for {video_path}: {e}")
        return 0


def get_video_resolution(video_path):
    """
    Get the resolution (width, height) of a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Tuple of (width, height) in pixels, or (0, 0) if there's an error
    """
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0:s=x',
                video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        dims = result.stdout.strip().split('x')
        if len(dims) == 2:
            return (int(dims[0]), int(dims[1]))
        return (0, 0)
    except Exception as e:
        print(f"Error getting video resolution for {video_path}: {e}")
        return (0, 0)


def get_video_fps(video_path):
    """
    Get the frame rate of a video file.

    ffprobe returns fps as a fraction (e.g., "30000/1001" for 29.97fps),
    so we parse and divide.

    Args:
        video_path: Path to the video file

    Returns:
        Frame rate as a float, or 0 if there's an error
    """
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ],
            capture_output=True,
            text=True
        )
        fps_str = result.stdout.strip()
        if '/' in fps_str:
            num, den = fps_str.split('/')
            return float(num) / float(den)
        return float(fps_str)
    except Exception as e:
        print(f"Error getting video fps for {video_path}: {e}")
        return 0


def video_has_audio(video_path):
    """
    Check if a video file has an audio stream.

    Args:
        video_path: Path to the video file

    Returns:
        True if the video has audio, False otherwise
    """
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a',
                '-show_entries', 'stream=codec_type',
                '-of', 'csv=p=0',
                video_path
            ],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def get_audio_volume(video_path):
    """
    Analyze the audio volume of a video file using ffmpeg's volumedetect filter.

    Returns the mean volume in dB. Lower values = quieter.
    Useful for detecting silent outputs that need fallback audio.

    Args:
        video_path: Path to the video file

    Returns:
        Mean volume in dB as a float, or None if no audio/error
    """
    try:
        result = subprocess.run(
            [
                'ffmpeg', '-i', video_path,
                '-af', 'volumedetect',
                '-vn', '-sn', '-dn',
                '-f', 'null', '/dev/null'
            ],
            capture_output=True,
            text=True
        )

        # volumedetect outputs to stderr
        for line in result.stderr.split('\n'):
            if 'mean_volume:' in line:
                parts = line.split('mean_volume:')
                if len(parts) > 1:
                    volume_str = parts[1].strip().replace('dB', '').strip()
                    return float(volume_str)

        return None
    except Exception as e:
        print(f"Error analyzing audio volume for {video_path}: {e}")
        return None


def get_metadata(video_path):
    """
    Read embedded metadata (title, comment/description, artist, etc.) from a video.

    Uses ffprobe to extract the format-level tags from the container.
    This is what the bulk uploader reads to build the YouTube upload queue.

    Args:
        video_path: Path to the video file

    Returns:
        dict with metadata keys (title, comment, artist, date, etc.)
        Keys are lowercase. Returns empty dict on error.
    """
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format_tags',
                '-of', 'json',
                video_path
            ],
            capture_output=True,
            text=True
        )
        import json
        data = json.loads(result.stdout)
        tags = data.get('format', {}).get('tags', {})
        # normalize keys to lowercase
        return {k.lower(): v for k, v in tags.items()}
    except Exception as e:
        print(f"Error reading metadata for {video_path}: {e}")
        return {}
