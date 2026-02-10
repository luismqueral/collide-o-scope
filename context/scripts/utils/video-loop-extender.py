"""
video-loop-extender.py - Extend Short Videos by Looping

This utility script takes short video clips and loops them to reach a target
duration. Useful for building up your source library when you have interesting
but short clips.

WHAT IT DOES:
1. Scans the library/video directory for MP4 files
2. For each video shorter than the target duration:
   - Calculates how many loops are needed
   - Creates a new "_looped" version at the target length
3. Videos already longer than the target are skipped

USE CASE:
If you have a bunch of 10-30 second clips but want them to be at least 10
minutes long (for YouTube or for having more footage to pull from), run this
script to extend them all automatically.

NOTE: The looped videos are saved back to the same library/video directory
with "_looped" appended to the filename.
"""

import os
import subprocess
import math

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory containing the video files to process
video_dir = "library/video"

# Target length in seconds (600 = 10 minutes)
desired_length = 600

# Where to save the looped videos
# By default, saves back to the library directory
output_dir = "library/video"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


def get_video_duration(video_path):
    """
    Get the duration of a video file in seconds using ffprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Duration in seconds as a float
    """
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ], stdout=subprocess.PIPE, text=True)
    return float(result.stdout)


def loop_video(video_path, output_path, loops):
    """
    Create a looped version of a video.
    
    Uses FFMPEG's stream_loop feature to repeat the video N times.
    The -c copy flag means we're not re-encoding, just concatenating,
    which is very fast.
    
    Args:
        video_path: Path to the source video
        output_path: Where to save the looped video
        loops: How many additional times to loop (0 = no looping)
    """
    subprocess.run([
        "ffmpeg",
        "-stream_loop", str(loops),  # Loop the input this many times
        "-i", video_path,
        "-c", "copy",          # Copy streams without re-encoding (fast!)
        "-fflags", "+genpts",  # Generate presentation timestamps
        output_path,
        "-y"  # Overwrite output if it exists
    ])


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

# Process each MP4 file in the directory
for video in os.listdir(video_dir):
    # Only process MP4 files (change extension if needed)
    if video.endswith(".mp4"):
        video_path = os.path.join(video_dir, video)
        
        # Create output filename with "_looped" suffix
        output_path = os.path.join(
            output_dir, 
            os.path.splitext(video)[0] + "_looped.mp4"
        )
        
        # Get the current duration
        duration = get_video_duration(video_path)
        
        # Calculate loops needed
        if duration > 0:  # Avoid division by zero
            # ceil() rounds up - we want to meet OR exceed the target
            # -1 because the video plays once before looping starts
            loops = math.ceil(desired_length / duration) - 1
            
            if loops > 0:
                print(f"Looping {video} to reach desired length of {desired_length} seconds...")
                loop_video(video_path, output_path, loops)
            else:
                print(f"{video} is already longer than {desired_length} seconds. Skipping...")

print("Processing completed.")
