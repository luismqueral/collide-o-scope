#!/usr/bin/env python3
"""
Add random audio mix to a video.

Picks N random videos from input, extracts audio, crops at random times,
pans them across stereo field, and mixes onto the target video.
"""

import os
import sys
import random
import subprocess
import argparse
from pathlib import Path

def get_duration(file_path):
    """Get duration of audio/video file in seconds."""
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0

def has_audio(file_path):
    """Check if file has audio stream."""
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-select_streams', 'a',
        '-show_entries', 'stream=codec_type',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ], capture_output=True, text=True)
    return 'audio' in result.stdout

def main():
    parser = argparse.ArgumentParser(description='Add random audio mix to video')
    parser.add_argument('-i', '--input', required=True, help='Input video (to add audio to)')
    parser.add_argument('-o', '--output', required=True, help='Output video')
    parser.add_argument('--source-dir', default='library/video', help='Directory to pick random videos from')
    parser.add_argument('--num-tracks', type=int, default=3, help='Number of audio tracks to mix')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--volume', type=float, default=0.7, help='Overall volume (0-1)')
    args = parser.parse_args()
    
    if args.seed:
        random.seed(args.seed)
    
    # Get target video duration
    target_duration = get_duration(args.input)
    print(f"Target video: {args.input}")
    print(f"Duration: {target_duration:.1f}s")
    print()
    
    # Find all videos with audio in source directory
    source_dir = Path(args.source_dir)
    all_videos = list(source_dir.glob('*.mp4')) + list(source_dir.glob('*.MP4'))
    
    # Filter to only those with audio
    videos_with_audio = [v for v in all_videos if has_audio(str(v))]
    print(f"Found {len(videos_with_audio)} videos with audio in {source_dir}")
    
    if len(videos_with_audio) < args.num_tracks:
        print(f"Warning: Only {len(videos_with_audio)} videos with audio, using all")
        selected = videos_with_audio
    else:
        selected = random.sample(videos_with_audio, args.num_tracks)
    
    print(f"\nSelected {len(selected)} tracks:")
    
    # Panning positions: -1 = full left, 0 = center, 1 = full right
    pan_positions = [-0.8, 0, 0.8]  # L, C, R
    if len(selected) > 3:
        # Add more positions if needed
        pan_positions = [i / (len(selected) - 1) * 2 - 1 for i in range(len(selected))]
    
    # Build complex filter for mixing
    filter_parts = []
    inputs = ['-i', args.input]  # Start with video input
    
    for i, video in enumerate(selected):
        video_duration = get_duration(str(video))
        
        # Random start time (ensure we have enough audio)
        max_start = max(0, video_duration - target_duration - 1)
        start_time = random.uniform(0, max_start) if max_start > 0 else 0
        
        pan = pan_positions[i] if i < len(pan_positions) else 0
        pan_name = "L" if pan < -0.3 else "R" if pan > 0.3 else "C"
        
        print(f"  [{i+1}] {video.name}")
        print(f"      Start: {start_time:.1f}s, Pan: {pan_name} ({pan:.1f})")
        
        inputs.extend(['-i', str(video)])
        
        # Filter: trim, pan, and prepare for mixing
        # Pan formula: left = 1-pan, right = 1+pan (normalized)
        left_vol = min(1.0, (1 - pan) * 0.5 + 0.5)
        right_vol = min(1.0, (1 + pan) * 0.5 + 0.5)
        
        filter_parts.append(
            f"[{i+1}:a]atrim=start={start_time}:duration={target_duration},"
            f"asetpts=PTS-STARTPTS,"
            f"pan=stereo|c0={left_vol}*c0|c1={right_vol}*c1,"
            f"volume={args.volume/len(selected):.2f}[a{i}]"
        )
    
    # Mix all audio tracks
    mix_inputs = ''.join(f'[a{i}]' for i in range(len(selected)))
    filter_parts.append(f"{mix_inputs}amix=inputs={len(selected)}:duration=first[amix]")
    
    filter_complex = ';'.join(filter_parts)
    
    print(f"\nMixing audio and adding to video...")
    
    cmd = [
        'ffmpeg', '-y', '-v', 'warning',
        *inputs,
        '-filter_complex', filter_complex,
        '-map', '0:v',
        '-map', '[amix]',
        '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        args.output
    ]
    
    subprocess.run(cmd, check=True)
    
    print(f"\n✓ Done! Output: {args.output}")

if __name__ == "__main__":
    main()

