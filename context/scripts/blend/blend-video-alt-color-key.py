"""
blend-video-alt-color-key.py - Intelligent Colorkey with Machine Learning

The most sophisticated version of the video blender. Instead of using hardcoded
colors (black/white) for the colorkey, this script analyzes each video to find
its dominant colors and uses THOSE for transparency.

HOW IT WORKS:
1. Picks N random videos from library/video
2. For each video:
   - Extracts a random frame as an image
   - Uses K-Means clustering (machine learning) to find dominant colors
   - Randomly selects colors from that palette to key out
3. Builds the colorkey filter using the video's ACTUAL colors
4. Stacks all videos with overlay
5. Creates a debug image showing which colors were extracted

WHY THIS IS COOL:
- Adapts to whatever videos you throw at it
- Keys out colors that actually exist in the footage
- More organic/unpredictable results than fixed black/white keying
- Debug images let you see what colors were detected

MACHINE LEARNING COMPONENT:
Uses K-Means clustering from scikit-learn to identify color clusters in frames.
K-Means groups similar pixels together and finds the "center" of each group,
giving us the N most representative colors in the image.
"""

import os
import random
import subprocess
import datetime
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans

# =============================================================================
# CONFIGURATION - Modify these settings to change the output
# =============================================================================

# Where to find input videos
VIDEO_INPUT = "library/video"

# Where to save output videos
OUTPUT_DIRECTORY = "projects/archive/output"

# Output video dimensions as "WIDTHxHEIGHT" string
OUTPUT_SIZE = "800x800"

# Output frame rate
FRAME_RATE = 18

# Exact length for output video in seconds
# Set to None to use a random length from VIDEO_LENGTH_RANGE
VIDEO_LENGTH = None

# Random length range (min, max) in seconds
VIDEO_LENGTH_RANGE = (60, 180)

# Number of videos to blend together
VIDEO_INPUT_NUM = 5

# Number of video audio tracks to mix into the final output
AUDIO_INPUT_NUM = 2

# =============================================================================
# COLOR PALETTE SETTINGS - These control the ML-based color extraction
# =============================================================================

# How many dominant colors to extract from each video frame
# Higher = more colors to choose from, but slower processing
COLORS_PER_PALETTE = 4

# How many of those colors to randomly select for keying out
# Lower = more of the video remains visible
# Higher = more transparency, more abstract result
COLORS_TO_KEY_OUT = 2

# Generate debug images showing the color palettes?
# Useful for understanding what colors are being detected
DEBUG_MODE = True


def select_random_files(directory: str, num_files: int) -> List[str]:
    """
    Select random video files from a directory.
    
    Args:
        directory: Path to folder containing videos
        num_files: How many videos to select
        
    Returns:
        List of full paths to selected video files
    """
    all_files = [
        f for f in os.listdir(directory) 
        if os.path.isfile(os.path.join(directory, f)) and not f.startswith('.')
    ]
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.3g2')
    video_files = [f for f in all_files if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"No video files found in {directory}")
        return []
    
    selected_files = random.sample(video_files, min(num_files, len(video_files)))
    return [os.path.join(directory, f) for f in selected_files]


def extract_random_frame(video_path: str, output_path: str) -> None:
    """
    Extract a single frame from a random point in the video.
    
    This gives us an image to analyze for color palette extraction.
    We pick a random timestamp to get variety in the colors we detect.
    
    Args:
        video_path: Path to the source video
        output_path: Where to save the extracted frame (as JPEG)
    """
    # First, get the video duration
    duration = float(subprocess.check_output([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]).decode('utf-8'))
    
    # Pick a random time within the video
    random_time = random.uniform(0, duration)
    
    # Extract that frame using FFMPEG
    # -ss = seek to timestamp
    # -frames:v 1 = extract just one frame
    subprocess.run([
        'ffmpeg', '-y',
        '-i', video_path,
        '-ss', str(random_time),
        '-frames:v', '1',
        output_path
    ], check=True)


def generate_color_palette(image_path: str, num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Extract the dominant colors from an image using K-Means clustering.
    
    K-MEANS EXPLAINED:
    K-Means is a clustering algorithm that groups similar data points together.
    For images:
    1. We treat each pixel as a point in 3D space (R, G, B)
    2. K-Means finds N cluster centers (the dominant colors)
    3. Each cluster center represents a common color in the image
    
    The algorithm:
    1. Randomly place N cluster centers
    2. Assign each pixel to its nearest center
    3. Move each center to the average of its assigned pixels
    4. Repeat steps 2-3 until centers stop moving
    
    Args:
        image_path: Path to the image file
        num_colors: How many dominant colors to extract
        
    Returns:
        List of (R, G, B) tuples representing the dominant colors
    """
    # Open and prepare the image
    image = Image.open(image_path)
    image = image.convert('RGB')  # Ensure RGB mode (no alpha)
    image = image.resize((150, 150))  # Downsample for faster processing
    
    # Convert image to array of pixels
    # Shape goes from (150, 150, 3) to (22500, 3) - a list of RGB values
    pixels = np.array(image)
    pixels = pixels.reshape(-1, 3)
    
    # Run K-Means clustering
    # n_clusters = how many color groups to find
    # n_init = how many times to run with different starting positions
    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(pixels)
    
    # Get the cluster centers (the dominant colors)
    colors = kmeans.cluster_centers_.astype(int)
    
    # =============================================================================
    # FILTER SIMILAR COLORS
    #
    # Sometimes K-Means finds colors that are very similar to each other.
    # We filter these out to ensure we get distinct colors.
    # The threshold (2000) is the squared Euclidean distance in RGB space.
    # =============================================================================
    distinct_colors = [colors[0]]
    for color in colors[1:]:
        # Check if this color is different enough from all colors we've kept
        is_distinct = all(np.sum((color - dc)**2) > 2000 for dc in distinct_colors)
        if is_distinct:
            distinct_colors.append(color)
        if len(distinct_colors) == num_colors:
            break
    
    return [tuple(color) for color in distinct_colors]


def create_debug_image(color_palettes: List[Tuple[str, List[Tuple[int, int, int]]]], 
                       output_path: str) -> None:
    """
    Create a visual debug image showing all extracted color palettes.
    
    This creates a single image with:
    - Each row = one video's color palette
    - Colored rectangles showing the extracted colors
    - The video filename below each palette
    
    Useful for understanding what colors the algorithm detected and why
    certain parts of videos became transparent.
    
    Args:
        color_palettes: List of (filename, [colors]) tuples
        output_path: Where to save the debug image
    """
    # Calculate image dimensions
    palette_height = 50  # Height of each color bar
    text_height = 30     # Space for filename
    total_height = (palette_height + text_height) * len(color_palettes)
    width = 500
    
    # Create a white canvas
    image = Image.new('RGB', (width, total_height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("Courier.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw each palette
    for i, (filename, colors) in enumerate(color_palettes):
        y_offset = i * (palette_height + text_height)
        
        # Draw color rectangles
        for j, color in enumerate(colors):
            x_start = j * width // len(colors)
            x_end = (j + 1) * width // len(colors)
            draw.rectangle([x_start, y_offset, x_end, y_offset + palette_height], fill=color)
        
        # Draw the filename below the colors
        text_y = y_offset + palette_height
        draw.text((10, text_y), os.path.basename(filename), font=font, fill='black')
    
    image.save(output_path)


def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video file in seconds using ffprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Duration in seconds as a float
    """
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)


def process_videos(input_videos: List[str], output_path: str, duration: int) -> None:
    """
    Process and overlay all input videos with intelligent colorkeying.
    
    For each video:
    1. Extract a random frame
    2. Analyze it to find dominant colors
    3. Build a colorkey filter using those colors
    4. Apply trim, loop, scale, and colorkey filters
    
    Then overlay all videos and mix selected audio tracks.
    
    Args:
        input_videos: List of paths to input video files
        output_path: Where to save the output
        duration: Desired output length in seconds
    """
    video_filter_complex = []
    audio_filter_complex = []
    audio_inputs = []
    color_palettes = []  # Store palettes for debug image
    
    for i, video in enumerate(input_videos):
        # Get video duration for random start time calculation
        video_duration = get_video_duration(video)
        
        # =============================================================================
        # EXTRACT FRAME AND ANALYZE COLORS
        # =============================================================================
        
        # Extract a random frame from this video
        frame_path = f"temp_frame_{i}.jpg"
        extract_random_frame(video, frame_path)
        
        # Use K-Means to find the dominant colors in that frame
        color_palette = generate_color_palette(frame_path, COLORS_PER_PALETTE)
        color_palettes.append((video, color_palette))
        
        # Clean up the temporary frame
        os.remove(frame_path)
        
        # Randomly select which colors from the palette to key out
        colors_to_key = random.sample(color_palette, COLORS_TO_KEY_OUT)
        
        # =============================================================================
        # BUILD VIDEO FILTER
        # =============================================================================
        
        # Calculate random start time
        max_start_time = max(0, video_duration - duration)
        random_start = random.uniform(0, max_start_time)
        
        # Build colorkey filters for each selected color
        # colorkey format: colorkey=0xRRGGBB:similarity:blend
        color_key_filters = []
        for color in colors_to_key:
            # Convert RGB tuple to hex string
            color_hex = '{:02x}{:02x}{:02x}'.format(*color)
            color_key_filters.append(f"colorkey=0x{color_hex}:0.3:0.2")
        
        # Full video filter chain:
        # trim -> reset pts -> loop -> reset pts -> scale -> colorkey(s)
        video_filter_complex.extend([
            f"[{i}:v]trim=start={random_start}:duration={duration},setpts=PTS-STARTPTS,"
            f"loop=loop=-1:size={duration*FRAME_RATE},"
            f"setpts=N/({FRAME_RATE}*TB),"
            f"scale={OUTPUT_SIZE},{','.join(color_key_filters)}[v{i}]"
        ])
        
        # =============================================================================
        # BUILD AUDIO FILTER (for first N videos)
        # =============================================================================
        if i < AUDIO_INPUT_NUM:
            audio_filter_complex.append(
                f"[{i}:a]atrim=start={random_start}:duration={duration},asetpts=PTS-STARTPTS,"
                f"aloop=loop=-1:size={duration*44100},"
                f"asetpts=N/(44100*TB)[a{i}]"
            )
            audio_inputs.append(f"[a{i}]")

    # =============================================================================
    # BUILD OVERLAY CHAIN
    # Stack all videos: [v0] + [v1] -> [temp1] + [v2] -> [temp2] etc.
    # =============================================================================
    overlay = f"[v0][v1]overlay=shortest=1[temp1]"
    for i in range(2, len(input_videos)):
        overlay += f";[temp{i-1}][v{i}]overlay=shortest=1[temp{i}]"
    overlay = overlay.replace(f"[temp{len(input_videos)-1}]", "[outv]")
    video_filter_complex.append(overlay)
    
    # Mix audio tracks if we have any
    if audio_inputs:
        audio_filter_complex.append(
            "".join(audio_inputs) + f"amix=inputs={len(audio_inputs)}:normalize=1[outa]"
        )
    
    # =============================================================================
    # BUILD AND RUN FFMPEG COMMAND
    # =============================================================================
    cmd = ['ffmpeg', '-y']
    
    # Add inputs
    for video in input_videos:
        cmd.extend(['-i', video])
    
    # Add filter complex
    filter_complex = video_filter_complex + audio_filter_complex
    cmd.extend([
        '-filter_complex',
        ";".join(filter_complex),
        '-map', '[outv]'
    ])
    
    # Map audio or disable
    if audio_inputs:
        cmd.extend(['-map', '[outa]'])
    else:
        cmd.extend(['-an'])
    
    # Output settings
    cmd.extend([
        '-t', str(duration),
        '-r', str(FRAME_RATE),
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-b:a', '128k',
        output_path
    ])
    
    subprocess.run(cmd, check=True)
    
    # =============================================================================
    # CREATE DEBUG IMAGE
    # Shows what colors were extracted from each video
    # =============================================================================
    if DEBUG_MODE:
        debug_image_path = output_path.rsplit('.', 1)[0] + '--debug.png'
        create_debug_image(color_palettes, debug_image_path)


def main():
    """
    Main entry point.
    
    Orchestrates video selection, processing, and output generation.
    """
    try:
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        
        # Select random input videos
        input_videos = select_random_files(VIDEO_INPUT, VIDEO_INPUT_NUM)
        if not input_videos:
            print("No input videos found. Exiting.")
            return

        # Calculate output duration
        duration = random.randint(*VIDEO_LENGTH_RANGE) if VIDEO_LENGTH is None else VIDEO_LENGTH
        
        # Generate unique output filename
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        output_filename = f"output_{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
        
        # Process!
        process_videos(input_videos, output_path, duration)
        
        print(f"Video processing completed. Output saved to: {output_path}")
        if DEBUG_MODE:
            debug_image_path = output_path.rsplit('.', 1)[0] + '--debug.png'
            print(f"Debug image saved to: {debug_image_path}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
