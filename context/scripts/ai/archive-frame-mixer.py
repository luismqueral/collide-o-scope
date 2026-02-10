"""
archive-frame-mixer.py - Generate chained video transitions from random archive frames

Pipeline:
1. Pick N+1 random videos from projects/archive/output (for N segments)
2. Extract a random frame from each video
3. Chain frames sequentially: frame1→frame2, frame2→frame3, etc.
4. Generate videos that transition from start frame to end frame
5. Concatenate all segments into a seamless final video

Uses fal-ai/kling-video/o1 with start_image_url + end_image_url for true first→last frame interpolation.

Usage:
  python archive-frame-mixer.py                           # Default: 5 chained segments
  python archive-frame-mixer.py --segments 3              # Generate 3 chained segments (needs 4 frames)
  python archive-frame-mixer.py --prompt "dreamy flow"    # Custom prompt
  python archive-frame-mixer.py --concat                  # Concatenate segments into final video
"""

import os
import sys
import subprocess
import random
import json
import base64
import shutil
import urllib.request
from datetime import datetime
from pathlib import Path
import time
import tempfile

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: PIL or numpy not installed")
    print("Install with: pip install pillow numpy")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULTS = {
    'num_source_videos': 6,    # How many archive videos to sample (needs segments + 1)
    'num_segments': 5,          # How many video segments to generate
    'archive_path': 'projects/archive/output',
    'output_dir': 'projects/archive/output',
    'prompt': 'slow organic transition, dreamlike morphing, flowing transformation',
}

MODEL_CONFIG = {
    'id': 'fal-ai/kling-video/o1/image-to-video',
    'name': 'Kling O1 Image-to-Video',
    'approx_duration': 5,  # ~5 seconds per segment
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def encode_image_to_base64(image_path):
    """Encode image to base64 data URI."""
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    ext = Path(image_path).suffix.lower()
    mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
    
    return f"data:{mime_type};base64,{image_data}"


def get_video_duration(video_path):
    """Get video duration in seconds."""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ], capture_output=True, text=True)
        return float(result.stdout.strip())
    except:
        return 0


def find_archive_videos(archive_path, min_duration=2.0):
    """
    Find all video files in the archive directory.
    Returns list of valid video paths.
    """
    video_extensions = ('.mp4', '.mov', '.avi', '.webm', '.mkv')
    videos = []
    
    archive = Path(archive_path)
    if not archive.exists():
        print(f"Error: Archive path does not exist: {archive_path}")
        return []
    
    # Walk through directory (non-recursive to avoid subdirs)
    for item in archive.iterdir():
        if item.is_file() and item.suffix.lower() in video_extensions:
            # Quick duration check
            duration = get_video_duration(str(item))
            if duration >= min_duration:
                videos.append(str(item))
    
    return videos


def extract_random_frame(video_path, output_path, margin_pct=0.15):
    """
    Extract a random frame from a video.
    Avoids the first and last margin_pct of the video.
    """
    duration = get_video_duration(video_path)
    
    if duration <= 0:
        print(f"Warning: Could not get duration for {video_path}")
        return None
    
    # Pick random timestamp avoiding margins
    margin = duration * margin_pct
    timestamp = random.uniform(margin, duration - margin)
    
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-ss', str(timestamp),
        '-i', video_path,
        '-frames:v', '1',
        '-q:v', '2',
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(output_path):
            return timestamp
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frame: {e}")
    
    return None


def concatenate_videos(video_paths, output_path):
    """
    Concatenate multiple videos into a single video.
    Uses ffmpeg concat demuxer for seamless joining.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for video_path in video_paths:
            abs_path = os.path.abspath(video_path)
            escaped_path = abs_path.replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
        concat_list = f.name
    
    try:
        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list,
            '-c:v', 'libx264',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(concat_list)


# =============================================================================
# FAL.AI VIDEO GENERATION
# =============================================================================

def generate_transition_video(start_frame_path, end_frame_path, output_path, prompt, segment_num):
    """
    Generate a video that transitions from start_frame to end_frame.
    Uses MiniMax Hailuo-02-Fast with subject_image for end frame targeting.
    """
    try:
        import fal_client
    except ImportError:
        print("Error: fal-client not installed. Run: pip install fal-client")
        sys.exit(1)
    
    if not os.environ.get('FAL_KEY'):
        print("Error: FAL_KEY not set in environment")
        sys.exit(1)
    
    # Encode both frames
    start_uri = encode_image_to_base64(start_frame_path)
    end_uri = encode_image_to_base64(end_frame_path)
    
    # Build arguments
    # Kling O1 uses start_image_url and end_image_url
    arguments = {
        "start_image_url": start_uri,
        "end_image_url": end_uri,
        "prompt": prompt,
        "duration": "5",  # 5 seconds
    }
    
    print(f"  → Calling fal.ai API for segment {segment_num}...")
    start_time = time.time()
    
    try:
        result = fal_client.subscribe(
            MODEL_CONFIG['id'],
            arguments=arguments,
            with_logs=False,
        )
    except Exception as e:
        print(f"  ✗ API error: {e}")
        return None
    
    elapsed = time.time() - start_time
    print(f"  → API completed in {elapsed:.1f}s")
    
    # Extract video URL from result
    video_url = None
    
    if isinstance(result, dict):
        if 'video' in result:
            video_url = result['video'].get('url') if isinstance(result['video'], dict) else result['video']
        elif 'video_url' in result:
            video_url = result['video_url']
        elif 'output' in result:
            video_url = result['output'].get('url') if isinstance(result['output'], dict) else result['output']
    
    if not video_url:
        print(f"  ✗ Could not extract video URL from response")
        print(f"  Response: {json.dumps(result, indent=2)[:500]}")
        return None
    
    # Download video
    print(f"  → Downloading video...")
    urllib.request.urlretrieve(video_url, output_path)
    
    duration = get_video_duration(output_path)
    print(f"  ✓ Segment {segment_num} complete ({duration:.1f}s)")
    
    return duration


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_archive_mixer(
    num_source_videos,
    num_segments,
    archive_path,
    output_dir,
    prompt,
    concat_output=False,
    keep_frames=False
):
    """
    Main pipeline:
    1. Find and select random archive videos
    2. Extract random frames from each
    3. Generate transition videos between random frame pairs
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create work directory
    work_dir = f"mixer_work_{timestamp}"
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ARCHIVE FRAME MIXER")
    print("=" * 60)
    print(f"Source videos: {num_source_videos}")
    print(f"Segments to generate: {num_segments}")
    print(f"Archive path: {archive_path}")
    print(f"Prompt: {prompt}")
    print("=" * 60)
    print()
    
    try:
        # STEP 1: Find archive videos
        print("[Step 1] Finding archive videos...")
        all_videos = find_archive_videos(archive_path)
        
        if len(all_videos) < num_source_videos:
            print(f"Warning: Only found {len(all_videos)} videos, need {num_source_videos}")
            num_source_videos = len(all_videos)
        
        if len(all_videos) < 2:
            print("Error: Need at least 2 videos to create transitions")
            return None
        
        # Select random videos
        selected_videos = random.sample(all_videos, num_source_videos)
        print(f"✓ Selected {num_source_videos} random videos")
        
        for i, v in enumerate(selected_videos):
            print(f"  {i+1}. {os.path.basename(v)}")
        print()
        
        # STEP 2: Extract random frames
        print("[Step 2] Extracting random frames...")
        extracted_frames = []
        
        for i, video_path in enumerate(selected_videos):
            frame_path = os.path.join(work_dir, f"frame_{i+1:02d}.png")
            timestamp_extracted = extract_random_frame(video_path, frame_path)
            
            if timestamp_extracted is not None:
                extracted_frames.append({
                    'path': frame_path,
                    'source_video': os.path.basename(video_path),
                    'timestamp': timestamp_extracted,
                })
                print(f"  ✓ Frame {i+1}: {os.path.basename(video_path)} @ {timestamp_extracted:.2f}s")
            else:
                print(f"  ✗ Failed to extract from {os.path.basename(video_path)}")
        
        if len(extracted_frames) < 2:
            print("Error: Could not extract enough frames")
            return None
        
        print(f"\n✓ Extracted {len(extracted_frames)} frames")
        print()
        
        # STEP 3: Create sequential pairings for chained transitions
        print("[Step 3] Creating sequential frame pairings...")
        
        # For N segments, we need N+1 frames chained sequentially:
        # Segment 1: frame[0] → frame[1]
        # Segment 2: frame[1] → frame[2]
        # etc.
        # This creates a continuous chain when concatenated
        
        if len(extracted_frames) < num_segments + 1:
            print(f"Warning: Need {num_segments + 1} frames for {num_segments} chained segments")
            print(f"         Only have {len(extracted_frames)} frames, reducing segments")
            num_segments = len(extracted_frames) - 1
        
        pairings = []
        
        for i in range(num_segments):
            start_frame = extracted_frames[i]
            end_frame = extracted_frames[i + 1]
            
            pairings.append({
                'segment': i + 1,
                'start': start_frame,
                'end': end_frame,
            })
            
            print(f"  Segment {i+1}: {start_frame['source_video']} → {end_frame['source_video']}")
        
        print()
        
        # STEP 4: Generate transition videos
        print("[Step 4] Generating transition videos...")
        print()
        
        generated_segments = []
        
        for pairing in pairings:
            seg_num = pairing['segment']
            segment_path = os.path.join(work_dir, f"segment_{seg_num:02d}.mp4")
            
            print(f"[Segment {seg_num}]")
            print(f"  Start: {pairing['start']['source_video']}")
            print(f"  End:   {pairing['end']['source_video']}")
            
            duration = generate_transition_video(
                pairing['start']['path'],
                pairing['end']['path'],
                segment_path,
                prompt,
                seg_num
            )
            
            if duration:
                generated_segments.append({
                    'path': segment_path,
                    'duration': duration,
                    'start_source': pairing['start']['source_video'],
                    'end_source': pairing['end']['source_video'],
                })
            else:
                print(f"  ✗ Failed to generate segment {seg_num}")
            
            print()
        
        if not generated_segments:
            print("Error: No segments generated")
            return None
        
        # STEP 5: Copy segments to output (and optionally concatenate)
        print("[Step 5] Saving outputs...")
        
        output_segments = []
        for i, seg in enumerate(generated_segments):
            output_path = os.path.join(output_dir, f"mixer_{timestamp}_seg{i+1:02d}.mp4")
            shutil.copy(seg['path'], output_path)
            output_segments.append(output_path)
            print(f"  ✓ {os.path.basename(output_path)}")
        
        # Concatenate if requested
        final_output = None
        if concat_output and len(generated_segments) > 1:
            print("\n  Concatenating segments...")
            final_output = os.path.join(output_dir, f"mixer_{timestamp}_final.mp4")
            concatenate_videos([s['path'] for s in generated_segments], final_output)
            final_duration = get_video_duration(final_output)
            print(f"  ✓ Final video: {os.path.basename(final_output)} ({final_duration:.1f}s)")
        
        # Save frames if requested
        if keep_frames:
            frames_dir = os.path.join(output_dir, f"mixer_{timestamp}_frames")
            os.makedirs(frames_dir, exist_ok=True)
            for frame in extracted_frames:
                dest = os.path.join(frames_dir, os.path.basename(frame['path']))
                shutil.copy(frame['path'], dest)
            print(f"  ✓ Frames saved to: {frames_dir}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'model': MODEL_CONFIG['id'],
            'prompt': prompt,
            'num_source_videos': num_source_videos,
            'num_segments': len(generated_segments),
            'total_duration': sum(s['duration'] for s in generated_segments),
            'source_videos': [os.path.basename(v) for v in selected_videos],
            'frames': [
                {
                    'frame': os.path.basename(f['path']),
                    'source': f['source_video'],
                    'timestamp': f['timestamp'],
                }
                for f in extracted_frames
            ],
            'segments': [
                {
                    'segment': i + 1,
                    'start_source': s['start_source'],
                    'end_source': s['end_source'],
                    'duration': s['duration'],
                    'output': os.path.basename(output_segments[i]),
                }
                for i, s in enumerate(generated_segments)
            ],
            'outputs': {
                'segments': [os.path.basename(p) for p in output_segments],
                'final': os.path.basename(final_output) if final_output else None,
            }
        }
        
        metadata_path = os.path.join(output_dir, f"mixer_{timestamp}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'segments': output_segments,
            'final': final_output,
            'metadata': metadata_path,
            'total_duration': sum(s['duration'] for s in generated_segments),
        }
        
    finally:
        # Cleanup work directory
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            print(f"\nCleaned up: {work_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate video transitions from random archive frames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Default: 5 segments from 10 random archive videos
  python archive-frame-mixer.py
  
  # Generate 3 shorter segments
  python archive-frame-mixer.py --segments 3
  
  # Use 20 source videos for more variety
  python archive-frame-mixer.py --num-videos 20
  
  # Custom prompt and concatenate into final video
  python archive-frame-mixer.py --prompt "surreal nightmare transition" --concat
  
  # Keep the extracted frames for inspection
  python archive-frame-mixer.py --keep-frames

HOW IT WORKS:
  1. Scans projects/archive/output for video files
  2. Randomly selects N+1 videos (where N = number of segments)
  3. Extracts a random frame from each video
  4. Chains frames sequentially: frame1→frame2, frame2→frame3, etc.
  5. Each segment ends where the next begins (seamless chain)
  6. Uses Kling O1 with start_image_url + end_image_url for true interpolation
        """
    )
    
    parser.add_argument('--num-videos', '-n', type=int, default=DEFAULTS['num_source_videos'],
                        help=f"Number of archive videos to sample (default: {DEFAULTS['num_source_videos']})")
    parser.add_argument('--segments', '-s', type=int, default=DEFAULTS['num_segments'],
                        help=f"Number of video segments to generate (default: {DEFAULTS['num_segments']})")
    parser.add_argument('--archive', '-a', type=str, default=DEFAULTS['archive_path'],
                        help=f"Path to archive directory (default: {DEFAULTS['archive_path']})")
    parser.add_argument('--output', '-o', type=str, default=DEFAULTS['output_dir'],
                        help=f"Output directory (default: {DEFAULTS['output_dir']})")
    parser.add_argument('--prompt', '-p', type=str, default=DEFAULTS['prompt'],
                        help='Prompt to guide the video generation')
    parser.add_argument('--concat', '-c', action='store_true',
                        help='Concatenate all segments into a final video')
    parser.add_argument('--keep-frames', '-k', action='store_true',
                        help='Keep the extracted frames')
    
    args = parser.parse_args()
    
    result = run_archive_mixer(
        num_source_videos=args.num_videos,
        num_segments=args.segments,
        archive_path=args.archive,
        output_dir=args.output,
        prompt=args.prompt,
        concat_output=args.concat,
        keep_frames=args.keep_frames,
    )
    
    if result:
        print("\n" + "=" * 60)
        print("✓ ARCHIVE FRAME MIXER COMPLETE!")
        print("=" * 60)
        print(f"Segments generated: {len(result['segments'])}")
        print(f"Total duration: {result['total_duration']:.1f}s")
        print(f"Metadata: {result['metadata']}")
        if result['final']:
            print(f"Final video: {result['final']}")
        print("=" * 60)
    else:
        print("\n✗ Archive frame mixer failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

