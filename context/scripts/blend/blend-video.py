"""
blend-video.py - Basic Video Blending with Colorkey

This is the simplest version of the video blender. It takes multiple random videos
from your input folder, applies colorkey (chroma key) filtering to make certain
colors transparent, and overlays them on top of each other.

WHAT IT DOES:
1. Picks N random videos from library/video
2. Scales them all to the same size
3. Applies colorkey filter to videos 2, 3, etc. (making white pixels transparent)
4. Stacks them on top of each other using overlay
5. Adds random audio from library/audio
6. Outputs the final blended video

COLORKEY EXPLAINED:
- COLOR_KEY_COLOR: The color to make transparent (default: white 'ffffff')
- COLOR_KEY_SIMULARITY: How close a color needs to be to match (0-1, lower = stricter)
- COLOR_KEY_OPACITY: Transparency of the keyed area (0-1, higher = more transparent)
"""

import os
import random
import subprocess
import argparse

# Get the project root directory (two levels up from scripts/blend/)
# This ensures paths work regardless of where you run the script from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# =============================================================================
# CONFIGURATION - Modify these settings to change the output
# =============================================================================
DEFAULT_SETTINGS = {
    # Where to find input videos
    'VID_INPUT_DIR': os.path.join(BASE_DIR, 'library', 'video'),
    
    # Where to find audio files to overlay
    'AUDIO_INPUT_DIR': os.path.join(BASE_DIR, 'library', 'audio'),
    
    # How many videos to blend together (minimum 2)
    'NUMBER_OF_VIDS': 3,
    
    # Output video dimensions (width,height)
    'VID_SIZE': '640,480',
    
    # Random duration range in seconds (min, max)
    # Final video will be a random length between these values
    'VID_DURATION': (30, 35),
    
    # Blend mode (not currently used in FFMPEG command, placeholder for future)
    'BLEND_MODE': 'overlay',
    
    # Colorkey settings - which color to make transparent
    # 'ffffff' = white, '000000' = black
    'COLOR_KEY_COLOR': 'ffffff',
    
    # How similar a pixel must be to the key color to be removed (0.0-1.0)
    # Lower = more strict matching, Higher = more pixels get removed
    'COLOR_KEY_SIMULARITY': '.3',
    
    # How transparent the keyed pixels become (0.0-1.0)
    # Higher = more transparent
    'COLOR_KEY_OPACITY': '.9',
    
    # Where to save the output video
    'OUTPUT_DIR': os.path.join(BASE_DIR, 'projects', 'archive', 'output'),
}


def blend_videos(settings):
    """
    Main function that performs the video blending.
    
    This builds an FFMPEG command that:
    1. Takes multiple video inputs
    2. Scales the first video as the "base" layer
    3. Applies colorkey to subsequent videos (making a color transparent)
    4. Overlays each video on top of the previous result
    5. Adds audio and outputs the final file
    
    Args:
        settings: Dictionary containing all configuration options
    """
    
    # =============================================================================
    # VALIDATION - Make sure all required directories exist
    # =============================================================================
    if not os.path.isdir(settings['VID_INPUT_DIR']):
        print(f"Video input directory {settings['VID_INPUT_DIR']} does not exist.")
        return
    if not os.path.isdir(settings['AUDIO_INPUT_DIR']):
        print(f"Audio input directory {settings['AUDIO_INPUT_DIR']} does not exist.")
        return
    if not os.path.isdir(settings['OUTPUT_DIR']):
        print(f"Output directory {settings['OUTPUT_DIR']} does not exist. Creating...")
        os.makedirs(settings['OUTPUT_DIR'], exist_ok=True)

    # =============================================================================
    # VIDEO SELECTION - Randomly pick videos from the input folder
    # =============================================================================
    
    # Get all video files (mp4, avi, mov) from the input directory
    video_files = [
        os.path.join(settings['VID_INPUT_DIR'], f) 
        for f in os.listdir(settings['VID_INPUT_DIR']) 
        if f.endswith(('.mp4', '.avi', '.mov'))
    ]
    
    # Make sure we have enough videos to blend
    if len(video_files) < settings['NUMBER_OF_VIDS']:
        print("Not enough video files in the directory to blend.")
        return

    # Randomly select the specified number of videos
    selected_videos = random.sample(video_files, settings['NUMBER_OF_VIDS'])

    # =============================================================================
    # AUDIO SELECTION - Pick a random audio file to use as the soundtrack
    # =============================================================================
    audio_files = [
        os.path.join(settings['AUDIO_INPUT_DIR'], f) 
        for f in os.listdir(settings['AUDIO_INPUT_DIR']) 
        if f.endswith(('.mp3', '.wav'))
    ]
    selected_audio = random.choice(audio_files) if audio_files else None

    # =============================================================================
    # BUILD FFMPEG COMMAND
    # =============================================================================
    
    # Convert VID_SIZE from 'width,height' to 'widthxheight' format for FFMPEG
    width, height = settings['VID_SIZE'].split(',')
    vid_size_formatted = f"{width}x{height}"

    # Start building the FFMPEG command
    ffmpeg_cmd = ['ffmpeg']

    # Add each selected video as an input (-i flag)
    for video in selected_videos:
        ffmpeg_cmd += ['-i', video]

    # =============================================================================
    # BUILD THE FILTER COMPLEX
    # 
    # The filter_complex is where the magic happens. It's a chain of operations:
    # 
    # [0:v] = first video's video stream
    # [1:v] = second video's video stream
    # etc.
    #
    # The flow is:
    # 1. Scale video 0 to target size -> call it [base]
    # 2. For each subsequent video:
    #    a. Scale it to target size
    #    b. Apply colorkey (make white pixels transparent)
    #    c. Overlay it on top of [base]
    #    d. The result becomes the new [base]
    # =============================================================================
    
    # Start with the first video as the base layer
    filter_complex = f"[0:v]scale={vid_size_formatted}[base];"
    
    # Process each additional video
    for i in range(1, settings['NUMBER_OF_VIDS']):
        # Scale the video and apply colorkey filter
        # colorkey format: colorkey=color:similarity:blend
        filter_complex += (
            f"[{i}:v]scale={vid_size_formatted},"
            f"colorkey=0x{settings['COLOR_KEY_COLOR']}:"
            f"{settings['COLOR_KEY_SIMULARITY']}:"
            f"{settings['COLOR_KEY_OPACITY']}[ck{i}];"
        )
        
        # Overlay this video on top of the base
        # shortest=1 means stop when the shortest input ends
            filter_complex += f"[base][ck{i}]overlay=shortest=1[base];"

    # Add the filter_complex to the command
    # [:-7] removes the trailing "[base];" to leave just the final output
    ffmpeg_cmd += [
        '-filter_complex', filter_complex[:-7],
        '-t', str(random.randint(*settings['VID_DURATION']))  # Random duration
    ]

    # Add audio input if we found an audio file
    if selected_audio:
        ffmpeg_cmd += ['-i', selected_audio, '-c:a', 'copy']

    # Specify the output file path
    output_file = os.path.join(settings['OUTPUT_DIR'], "blended_video.mp4")
    ffmpeg_cmd += [output_file]

    # =============================================================================
    # EXECUTE FFMPEG
    # =============================================================================
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Blended video created successfully: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to blend videos: {e}")


# =============================================================================
# ENTRY POINT - Run the script
# =============================================================================
if __name__ == "__main__":
    blend_videos(DEFAULT_SETTINGS)
