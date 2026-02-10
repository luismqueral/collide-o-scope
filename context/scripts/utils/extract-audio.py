#!/usr/bin/env python3
"""
Audio Extraction Script
Extracts audio tracks from video files in the output folder.
Run periodically to extract audio from new videos that don't have audio extracted yet.
"""

import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Configuration
OUTPUT_DIR = Path(__file__).parent / "output"
AUDIO_DIR = OUTPUT_DIR / "audio"

def ensure_audio_dir():
    """Create audio directory if it doesn't exist."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Audio output directory: {AUDIO_DIR}")

def get_video_files():
    """Get all video files from output directory."""
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    videos = []
    
    for f in OUTPUT_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in video_extensions:
            videos.append(f)
    
    return sorted(videos)

def get_existing_audio_files():
    """Get set of video names that already have audio extracted."""
    if not AUDIO_DIR.exists():
        return set()
    
    existing = set()
    for f in AUDIO_DIR.iterdir():
        if f.is_file():
            # Remove the audio extension to get the original video name stem
            existing.add(f.stem)
    
    return existing

def check_video_has_audio(video_path):
    """Check if video file has an audio stream."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a', 
             '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', str(video_path)],
            capture_output=True, text=True
        )
        return 'audio' in result.stdout
    except Exception as e:
        print(f"  Warning: Could not check audio stream: {e}")
        return True  # Assume it has audio and let ffmpeg handle it

def extract_audio(video_path, force=False):
    """Extract audio from a video file to the audio directory."""
    audio_output = AUDIO_DIR / f"{video_path.stem}.mp3"
    
    # Skip if already extracted (unless force)
    if audio_output.exists() and not force:
        return False, "already exists"
    
    # Check if video has audio stream
    if not check_video_has_audio(video_path):
        return False, "no audio stream"
    
    try:
        # Use ffmpeg to extract audio as MP3 (good balance of size and quality)
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'libmp3lame',
            '-ab', '192k',  # 192kbps bitrate
            '-ar', '44100',  # 44.1kHz sample rate
            str(audio_output)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and audio_output.exists():
            size_mb = audio_output.stat().st_size / (1024 * 1024)
            return True, f"extracted ({size_mb:.2f} MB)"
        else:
            # Check for common errors
            if "does not contain any stream" in result.stderr or "Output file is empty" in result.stderr:
                return False, "no audio stream"
            return False, f"ffmpeg error: {result.stderr[:100]}"
            
    except Exception as e:
        return False, f"error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Extract audio from video files in output folder')
    parser.add_argument('--force', '-f', action='store_true', 
                        help='Re-extract audio even if it already exists')
    parser.add_argument('--file', type=str, 
                        help='Extract audio from a specific file only')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Show what would be done without actually extracting')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Audio Extraction Script - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    ensure_audio_dir()
    
    # Get videos to process
    if args.file:
        # Single file mode
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = OUTPUT_DIR / file_path
        
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return 1
        
        videos = [file_path]
    else:
        # Batch mode - all videos
        videos = get_video_files()
    
    if not videos:
        print("No video files found in output directory.")
        return 0
    
    print(f"Found {len(videos)} video file(s) to process\n")
    
    # Track stats
    extracted = 0
    skipped = 0
    failed = 0
    no_audio = 0
    
    for video in videos:
        print(f"Processing: {video.name}")
        
        if args.dry_run:
            audio_output = AUDIO_DIR / f"{video.stem}.mp3"
            if audio_output.exists() and not args.force:
                print(f"  [DRY RUN] Would skip (already exists)")
                skipped += 1
            else:
                print(f"  [DRY RUN] Would extract to: {audio_output.name}")
                extracted += 1
            continue
        
        success, message = extract_audio(video, force=args.force)
        
        if success:
            print(f"  ✓ {message}")
            extracted += 1
        elif "already exists" in message:
            print(f"  - Skipped: {message}")
            skipped += 1
        elif "no audio" in message.lower():
            print(f"  ○ No audio stream in video")
            no_audio += 1
        else:
            print(f"  ✗ Failed: {message}")
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Extracted: {extracted}")
    print(f"  Skipped:   {skipped}")
    print(f"  No audio:  {no_audio}")
    print(f"  Failed:    {failed}")
    print(f"{'='*60}\n")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())

