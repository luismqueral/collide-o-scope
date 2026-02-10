"""
blend-video-multi-vid.py - Multi-Layer Video Compositing

The workhorse version of the video blender. This script handles multiple videos
(default: 5) and intelligently manages both video and audio streams.

WHAT IT DOES:
1. Picks N random videos from library/video (default: 5)
2. For each video:
   - Calculates a random start time
   - Trims to the desired duration
   - Loops the segment if it's too short
   - Applies colorkey (randomly choosing black or white to key out)
3. Stacks all videos on top of each other with overlay
4. Mixes audio from a subset of the videos (default: 2 audio tracks)
5. Outputs the final composited video

KEY FEATURES:
- Handles any number of input videos (not just 2)
- Randomly selects between black (000000) and white (ffffff) for colorkey
- Edge softness control for smoother colorkey transitions
- Audio mixing from multiple sources
- Robust error handling for missing files or audio streams
"""

import os
import random
import subprocess
import datetime
import re
from typing import List, Tuple, Optional

# =============================================================================
# CONFIGURATION - Modify these settings to change the output
# =============================================================================

# Where to find input videos
VIDEO_INPUT = "library/video"

# Where to save output videos
OUTPUT_DIRECTORY = "projects/archive/output"

# Output video dimensions as "WIDTHxHEIGHT" string
OUTPUT_SIZE = "1920x1080"

# Output frame rate (lower = dreamier/choppier, higher = smoother)
FRAME_RATE = 18

# Exact length for output video in seconds
# Set to None to use a random length from VIDEO_LENGTH_RANGE
VIDEO_LENGTH = None

# Random length range (min, max) in seconds
VIDEO_LENGTH_RANGE = (120, 300)

# Where to find audio files (set to None to use audio from input videos)
AUDIO_INPUT = None

# Number of videos to blend together
VIDEO_INPUT_NUM = 5

# Number of video audio tracks to mix into the final output
# This randomly selects N videos to pull audio from
AUDIO_INPUT_NUM = 2

# Colors available for colorkey - script randomly picks one per video
# 'ffffff' = white, '000000' = black
# You can add more colors here, e.g., ('ffffff', '000000', 'ff0000')
COLORKEY = ('ffffff', '000000')

# How similar a pixel must be to the key color to be removed (0.0-1.0)
SIMILARITY = 0.3

# Edge softness for the colorkey effect (0-100)
# 0 = hard edges (pixelated look)
# Higher values = softer, more gradual transparency at edges
EDGE_SOFTNESS = 5

# =============================================================================
# HD FILTERING
# =============================================================================

# Only use HD videos (720p or higher)
HD_ONLY = True

# Minimum resolution (smaller dimension) for HD filtering
# 720 = 720p minimum, 1080 = 1080p minimum
MIN_RESOLUTION = 720

# =============================================================================
# SILENT OUTPUT FALLBACK CONFIGURATION
# =============================================================================

# Volume threshold in dB - outputs below this are considered "silent"
# -60 dB is very quiet, -40 dB is quiet but audible, -20 dB is moderate
SILENCE_THRESHOLD_DB = -50.0

# Range for number of fallback audio tracks to sample (min, max)
# When output is silent, script samples random.randint(*this_range) audio tracks
FALLBACK_AUDIO_RANGE = (1, 2)

# Stereo panning for fallback audio tracks
# 0.0 = full left, 0.5 = center, 1.0 = full right
# Tracks alternate between left and right panning
FALLBACK_PAN_LEFT = 0.4   # 60% left (0.4 means 40% toward center from full left)
FALLBACK_PAN_RIGHT = 0.6  # 60% right (0.6 means 40% toward center from full right)


def sanitize_filename(filename: str) -> str:
    """
    Clean up a filename to avoid FFMPEG errors.
    
    FFMPEG can choke on special characters in filenames. This function:
    - Replaces problematic characters with underscores
    - Removes multiple periods
    - Strips leading/trailing periods
    
    Args:
        filename: The original filename
        
    Returns:
        A sanitized version safe for FFMPEG
    """
    # Keep only alphanumeric, underscore, hyphen, and period
    sanitized = re.sub(r'[^\w\-_.]', '_', filename)
    # Collapse multiple periods into one
    sanitized = re.sub(r'\.{2,}', '.', sanitized)
    # Remove periods at start/end
    return sanitized.strip('.')


def select_random_files(directory: str, num_files: int) -> List[str]:
    """
    Select random video files from a directory.
    
    This function:
    1. Lists all files in the directory (excluding hidden files)
    2. Filters to only video files by extension
    3. Optionally filters by resolution (HD_ONLY mode)
    4. Randomly samples the requested number
    
    Args:
        directory: Path to folder containing videos
        num_files: How many videos to select
        
    Returns:
        List of full paths to selected video files
    """
    # Get all non-hidden files
    all_files = [
        f for f in os.listdir(directory) 
        if os.path.isfile(os.path.join(directory, f)) and not f.startswith('.')
    ]
    
    # Filter to video files only
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.3g2')
    video_files = [f for f in all_files if f.lower().endswith(video_extensions)]
    print(f"Total video files found: {len(video_files)}")
    
    if not video_files:
        print(f"No video files found in {directory}")
        return []
    
    # Filter by resolution if HD_ONLY is enabled
    if HD_ONLY:
        print(f"HD filtering enabled (min {MIN_RESOLUTION}p). Checking resolutions...")
        hd_videos = []
        for f in video_files:
            path = os.path.join(directory, f)
            width, height = get_video_resolution(path)
            min_dim = min(width, height)
            if min_dim >= MIN_RESOLUTION:
                hd_videos.append(f)
        
        print(f"HD videos ({MIN_RESOLUTION}p+): {len(hd_videos)} of {len(video_files)}")
        
        if not hd_videos:
            print(f"No HD videos found! Try setting HD_ONLY = False")
            return []
        
        video_files = hd_videos
    
    # Randomly select up to num_files videos
    selected_files = random.sample(video_files, min(num_files, len(video_files)))
    print(f"Selected {len(selected_files)} videos for blending")
    return [os.path.join(directory, f) for f in selected_files]


def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video file in seconds using ffprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Duration in seconds as a float
    """
    result = subprocess.run(
        [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    return float(result.stdout)


def get_video_resolution(video_path: str) -> Tuple[int, int]:
    """
    Get the resolution (width, height) of a video file using ffprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (width, height) in pixels
    """
    result = subprocess.run(
        [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            video_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    try:
        width, height = result.stdout.strip().split(',')
        return int(width), int(height)
    except:
        return 0, 0


def calculate_output_duration(video_length: Optional[int], 
                              video_length_range: Tuple[int, int]) -> int:
    """
    Calculate the output video duration.
    
    If a specific length is set, use that. Otherwise, pick randomly from range.
    
    Args:
        video_length: Specific length in seconds, or None for random
        video_length_range: (min, max) tuple for random selection
        
    Returns:
        Duration in seconds
    """
    if video_length is not None:
        return video_length
    return random.randint(*video_length_range)


def hex_to_colorkey(hex_color: str) -> str:
    """
    Convert a hex color string to FFMPEG's colorkey format.
    
    FFMPEG expects colors as '0xRRGGBB', so we just add the prefix.
    
    Args:
        hex_color: Color as 'RRGGBB' (without 0x prefix)
        
    Returns:
        Color as '0xRRGGBB' (with prefix)
    """
    return f"0x{hex_color}"


def get_audio_volume(video_path: str) -> Optional[float]:
    """
    Analyze the audio volume of a video file using ffmpeg's volumedetect filter.
    
    Returns the mean volume in dB. Lower values = quieter audio.
    Returns None if the video has no audio or analysis fails.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Mean volume in dB, or None if no audio/error
    """
    try:
        # Use volumedetect filter to analyze audio
        result = subprocess.run(
            [
                'ffmpeg', '-i', video_path,
                '-af', 'volumedetect',
                '-vn',  # No video output
                '-sn',  # No subtitle output
                '-dn',  # No data output
                '-f', 'null',
                '/dev/null'
            ],
            capture_output=True,
            text=True
        )
        
        # volumedetect outputs to stderr
        output = result.stderr
        
        # Look for mean_volume line
        for line in output.split('\n'):
            if 'mean_volume:' in line:
                # Extract the dB value
                # Format: "mean_volume: -XX.X dB"
                parts = line.split('mean_volume:')
                if len(parts) > 1:
                    volume_str = parts[1].strip().replace('dB', '').strip()
                    return float(volume_str)
        
        return None  # No audio stream found
        
    except Exception as e:
        print(f"Error analyzing audio volume: {e}")
        return None


def has_audio_stream(video_path: str) -> bool:
    """
    Check if a video file has an audio stream.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        True if video has audio, False otherwise
    """
    probe = subprocess.run(
        [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a',
            '-count_packets',
            '-show_entries', 'stream=codec_type',
            '-of', 'csv=p=0',
            video_path
        ],
        capture_output=True,
        text=True
    )
    return bool(probe.stdout.strip())


def add_fallback_audio(video_path: str, source_videos: List[str], duration: int) -> str:
    """
    Add fallback audio from random source videos to a silent/quiet video.
    
    This function:
    1. Selects random videos that have audio
    2. Samples audio from them
    3. Pans them alternately 60% left and 60% right
    4. Mixes them and adds to the video
    
    Args:
        video_path: Path to the video that needs audio
        source_videos: List of potential source videos for audio
        duration: Target duration in seconds
        
    Returns:
        Path to the new video with audio (or original if fallback fails)
    """
    print(f"\n⚠️  Output has low/no volume. Adding fallback audio...")
    
    # Find videos with audio
    videos_with_audio = []
    for video in source_videos:
        if os.path.isfile(video) and has_audio_stream(video):
            videos_with_audio.append(video)
    
    if not videos_with_audio:
        print("No source videos with audio found. Keeping silent output.")
        return video_path
    
    # Determine how many audio tracks to sample
    num_tracks = random.randint(*FALLBACK_AUDIO_RANGE)
    num_tracks = min(num_tracks, len(videos_with_audio))
    
    print(f"Sampling audio from {num_tracks} random video(s)...")
    
    # Select random videos for audio
    selected_sources = random.sample(videos_with_audio, num_tracks)
    
    # Build ffmpeg command
    cmd = ['ffmpeg', '-y']
    
    # Add the original video as input
    cmd.extend(['-i', video_path])
    
    # Add audio source videos
    for source in selected_sources:
        cmd.extend(['-i', source])
    
    # Build filter complex for audio processing
    audio_filters = []
    panned_outputs = []
    
    for i, source in enumerate(selected_sources):
        input_idx = i + 1  # +1 because input 0 is the original video
        
        # Get video duration for proper audio trimming
        source_duration = get_video_duration(source)
        start_time = random.uniform(0, max(0, source_duration - duration))
        
        # Determine pan direction (alternate left/right)
        if i % 2 == 0:
            pan_value = FALLBACK_PAN_LEFT
        else:
            pan_value = FALLBACK_PAN_RIGHT
        
        # Audio filter chain:
        # 1. atrim: Start at random point
        # 2. asetpts: Reset timestamps
        # 3. aloop: Loop to fill duration
        # 4. asetpts: Reset again
        # 5. stereopan: Pan left or right using stereotools
        audio_filters.append(
            f"[{input_idx}:a]atrim=start={start_time},asetpts=PTS-STARTPTS,"
            f"aloop=loop=-1:size={duration*44100},asetpts=N/(44100*TB),"
            f"stereotools=mpan={pan_value}[fallback_a{i}]"
        )
        panned_outputs.append(f"[fallback_a{i}]")
    
    # Mix all fallback audio tracks together
    if len(panned_outputs) > 1:
        audio_filters.append(
            "".join(panned_outputs) + f"amix=inputs={len(panned_outputs)}:normalize=1[fallback_mix]"
        )
        final_audio = "[fallback_mix]"
    else:
        final_audio = panned_outputs[0]
    
    # Build the full filter complex
    filter_complex = ";".join(audio_filters)
    
    # Generate output path
    base, ext = os.path.splitext(video_path)
    output_path = f"{base}_with_audio{ext}"
    
    # Build final command
    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '0:v',  # Video from original
        '-map', final_audio,  # Mixed fallback audio
        '-t', str(duration),
        '-c:v', 'copy',  # Copy video (no re-encode)
        '-c:a', 'aac',
        '-b:a', '128k',
        '-ar', '44100',
        output_path
    ])
    
    print(f"Adding panned audio from: {[os.path.basename(s) for s in selected_sources]}")
    print("Executing fallback audio command:", subprocess.list2cmdline(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        
        # Replace original with new version
        os.remove(video_path)
        os.rename(output_path, video_path)
        
        print(f"✓ Fallback audio added successfully!")
        return video_path
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to add fallback audio: {e}")
        # Clean up if output was created
        if os.path.exists(output_path):
            os.remove(output_path)
        return video_path


def process_videos(input_videos: List[str], output_path: str, duration: int) -> None:
    """
    Process and overlay all input videos into a single output.
    
    This is the main processing function. It builds a complex FFMPEG filter
    chain that handles:
    - Trimming each video to a random start point
    - Looping short videos to fill the duration
    - Applying colorkey to each video (randomly choosing black or white)
    - Overlaying all videos on top of each other
    - Mixing audio from selected videos
    
    Args:
        input_videos: List of paths to input video files
        output_path: Where to save the output
        duration: Desired output length in seconds
    """
    # These lists will hold our filter chain components
    video_filter_complex = []  # Video processing filters
    audio_filter_complex = []  # Audio processing filters
    audio_inputs = []          # List of audio streams to mix
    
    # =============================================================================
    # VALIDATION - Check that all input files exist
    # =============================================================================
    valid_input_videos = [video for video in input_videos if os.path.isfile(video)]
    
    if len(valid_input_videos) < len(input_videos):
        print(f"Warning: {len(input_videos) - len(valid_input_videos)} file(s) not found and will be skipped.")
        for video in input_videos:
            if not os.path.isfile(video):
                print(f"File not found: {video}")
    
    if not valid_input_videos:
        print("Error: No valid input videos found.")
        return

    # =============================================================================
    # AUDIO SOURCE SELECTION
    # Randomly decide which videos we'll pull audio from
    # =============================================================================
    audio_video_indices = random.sample(
        range(len(valid_input_videos)), 
        min(AUDIO_INPUT_NUM, len(valid_input_videos))
    )
    
    # =============================================================================
    # BUILD VIDEO FILTERS FOR EACH INPUT
    # =============================================================================
    for i, video in enumerate(valid_input_videos):
        # Randomly choose which color to key out for this video
        color = random.choice(COLORKEY)
        
        # Get the video's duration so we can pick a random start point
        video_duration = get_video_duration(video)
        
        # Pick a random start time (ensuring we have enough footage)
        # random.uniform gives us a float for sub-second precision
        start_time = random.uniform(0, max(0, video_duration - duration))
        
        # =============================================================================
        # VIDEO FILTER CHAIN FOR THIS INPUT
        #
        # The chain is:
        # 1. trim: Start at random point, cut to duration
        # 2. setpts: Reset timestamps (required after trim)
        # 3. loop: Loop the segment to fill duration (in case source is short)
        # 4. setpts: Reset timestamps again (required after loop)
        # 5. scale: Resize to output dimensions
        # 6. colorkey: Make the selected color transparent
        #
        # Output is labeled [v0], [v1], [v2], etc.
        # =============================================================================
        video_filter_complex.extend([
            f"[{i}:v]trim=start={start_time},setpts=PTS-STARTPTS,"
            f"loop=loop=-1:size={duration*FRAME_RATE},setpts=N/({FRAME_RATE}*TB),"
            f"scale={OUTPUT_SIZE},colorkey={hex_to_colorkey(color)}:{SIMILARITY}:{EDGE_SOFTNESS/100}[v{i}]"
        ])
        
        # =============================================================================
        # AUDIO FILTER CHAIN (only for selected videos)
        # =============================================================================
        if i in audio_video_indices:
            # First, check if this video actually has an audio stream
            probe = subprocess.run(
                [
                    'ffprobe', '-v', 'error',
                    '-select_streams', 'a',  # Only audio streams
                    '-count_packets',
                    '-show_entries', 'stream=codec_type',
                    '-of', 'csv=p=0',
                    video
                ], 
                capture_output=True, 
                text=True
            )
            
            # If there's audio output, this video has audio
            if probe.stdout.strip():
                # Audio filter chain:
                # 1. atrim: Start at the same random point as video
                # 2. asetpts: Reset audio timestamps
                # 3. aloop: Loop audio to fill duration
                # 4. asetpts: Reset timestamps again
                audio_filter_complex.append(
                    f"[{i}:a]atrim=start={start_time},asetpts=PTS-STARTPTS,"
                    f"aloop=loop=-1:size={duration*44100},asetpts=N/(44100*TB)[a{i}]"
                )
                audio_inputs.append(f"[a{i}]")

    # =============================================================================
    # BUILD THE OVERLAY CHAIN
    #
    # This stacks all videos on top of each other:
    # - If only 1 video, just pass it through
    # - Otherwise, overlay [v0] + [v1] -> [temp1]
    #             then [temp1] + [v2] -> [temp2]
    #             etc.
    # - Final output is labeled [outv]
    # =============================================================================
    if len(valid_input_videos) == 1:
        # Single video: just pass through with null filter
        video_filter_complex.append(f"[v0]null[outv]")
    else:
        # Start by overlaying first two videos
        overlay = f"[v0][v1]overlay=shortest=1[temp1]"
        
        # Add each subsequent video
        for i in range(2, len(valid_input_videos)):
            overlay += f";[temp{i-1}][v{i}]overlay=shortest=1[temp{i}]"
        
        # Rename the final temp to [outv]
        overlay = overlay.replace(f"[temp{len(valid_input_videos)-1}]", "[outv]")
        video_filter_complex.append(overlay)
    
    # =============================================================================
    # AUDIO MIXING
    #
    # If we have multiple audio streams, mix them together with amix
    # normalize=1 prevents clipping by adjusting volume automatically
    # =============================================================================
    if audio_inputs:
        audio_filter_complex.append(
            "".join(audio_inputs) + f"amix=inputs={len(audio_inputs)}:normalize=1[outa]"
        )
    
    # =============================================================================
    # BUILD THE FFMPEG COMMAND
    # =============================================================================
    cmd = ['ffmpeg', '-y']  # -y = overwrite output without asking
    
    # Add all input videos
    for video in valid_input_videos:
        cmd.extend(['-i', video])
    
    # Combine all filters into the filter_complex
    filter_complex = video_filter_complex + audio_filter_complex
    cmd.extend([
        '-filter_complex',
        ";".join(filter_complex),
        '-map', '[outv]'  # Use our video output
    ])
    
    # Map audio or disable it
    if audio_inputs:
        cmd.extend(['-map', '[outa]'])
    else:
        cmd.extend(['-an'])  # No audio
    
    # Output settings
    cmd.extend([
        '-t', str(duration),      # Total duration
        '-r', str(FRAME_RATE),    # Frame rate
        '-c:v', 'libx264',        # Video codec
        '-c:a', 'aac',            # Audio codec
        '-b:a', '128k',           # Audio bitrate
        '-ar', '44100',           # Audio sample rate
        output_path
    ])
    
    # Show the command for debugging
    print("Executing command:", subprocess.list2cmdline(cmd))
    
    # Run FFMPEG
    subprocess.run(cmd, check=True)


def main():
    """
    Main entry point.
    
    Orchestrates the video selection, duration calculation, and processing.
    Also checks for silent output and adds fallback audio if needed.
    """
    try:
        # Create output directory if needed
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        
        # Select random input videos
        input_videos = select_random_files(VIDEO_INPUT, VIDEO_INPUT_NUM)
        print(f"Selected input videos: {input_videos}")
        
        if not input_videos:
            print("No input videos found. Exiting.")
            return

        # Calculate output duration
        duration = calculate_output_duration(VIDEO_LENGTH, VIDEO_LENGTH_RANGE)
        print(f"Output duration: {duration} seconds")
        
        # Generate unique output filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        output_filename = f"output_{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
        
        # Process!
        process_videos(input_videos, output_path, duration)
        
        print(f"Video processing completed. Output saved to: {output_path}")
        
        # =============================================================================
        # SILENT OUTPUT CHECK & FALLBACK AUDIO
        # =============================================================================
        print("\nAnalyzing output audio levels...")
        volume = get_audio_volume(output_path)
        
        if volume is None:
            print(f"No audio detected in output (volume: None)")
            needs_fallback = True
        elif volume < SILENCE_THRESHOLD_DB:
            print(f"Output volume is very low: {volume:.1f} dB (threshold: {SILENCE_THRESHOLD_DB} dB)")
            needs_fallback = True
        else:
            print(f"Output volume OK: {volume:.1f} dB")
            needs_fallback = False
        
        if needs_fallback:
            # Get a fresh pool of videos to sample audio from
            # We sample from the entire video library for more variety
            all_videos = [
                os.path.join(VIDEO_INPUT, f) 
                for f in os.listdir(VIDEO_INPUT) 
                if os.path.isfile(os.path.join(VIDEO_INPUT, f)) 
                and f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.3g2'))
                and not f.startswith('.')
            ]
            
            add_fallback_audio(output_path, all_videos, duration)
        
        print(f"\n✓ Final output: {output_path}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
