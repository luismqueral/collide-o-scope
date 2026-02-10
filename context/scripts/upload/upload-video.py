"""
upload-video.py - Auto-Upload to YouTube with Word Salad Titles

This script automatically uploads your latest generated video to YouTube.
It creates whimsical, abstract titles using random "word salad" generation.

WHAT IT DOES:
1. Finds the most recently modified MP4 in the output directory
2. Generates a random poetic title like "ephemeral cascade nebula"
3. Uploads the video to YouTube as a private video
4. Returns the video ID for reference

PREREQUISITES:
1. You need a YouTube API OAuth client secret file (client_secret.json)
2. First run will open a browser for Google authentication
3. Credentials are cached in credentials.storage for future runs

SETUP:
1. Go to Google Cloud Console
2. Create a project and enable YouTube Data API v3
3. Create OAuth 2.0 credentials (Desktop application)
4. Download as client_secret.json and place in this directory

NOTE: Videos are uploaded as PRIVATE by default for safety.
Change the privacy_status if you want them public.
"""

from simple_youtube_api.Channel import Channel
from simple_youtube_api.LocalVideo import LocalVideo
import os
import random

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory containing the output videos
video_directory_path = 'output'

# =============================================================================
# WORD SALAD GENERATOR
# =============================================================================

def generate_word_salad(n=5):
    """
    Generate a random "word salad" title from evocative words.
    
    These words are chosen to be:
    - Aesthetically pleasing
    - Evocative/atmospheric
    - Abstract (matching the video art aesthetic)
    
    Args:
        n: How many words to include in the title
        
    Returns:
        A string of n random words separated by spaces
    """
    words = [
        "ephemeral",    # lasting briefly
        "cascade",      # falling water / sequence
        "nebula",       # cosmic cloud
        "quixotic",     # dreamy/idealistic
        "serendipity",  # happy accident
        "luminous",     # glowing
        "sonder",       # realization that everyone has a complex life
        "zenith",       # highest point
        "ethereal",     # otherworldly
        "elixir"        # magical potion
    ]
    return ' '.join(random.sample(words, n))


# =============================================================================
# FIND THE NEWEST VIDEO
# =============================================================================

# Get all files in the output directory
files = [
    os.path.join(video_directory_path, f) 
    for f in os.listdir(video_directory_path) 
    if os.path.isfile(os.path.join(video_directory_path, f))
]

# Filter to only MP4 files
mp4_files = [f for f in files if f.endswith('.mp4')]

# Find the most recently modified file
newest_file = max(mp4_files, key=os.path.getmtime)

# =============================================================================
# YOUTUBE AUTHENTICATION
# =============================================================================

# Create a channel object and authenticate
channel = Channel()
channel.login("client_secret.json", "credentials.storage")

# =============================================================================
# SET UP THE VIDEO
# =============================================================================

# Create a LocalVideo object for the newest output file
video = LocalVideo(file_path=newest_file)

# Set the title to a random word salad
video.set_title(generate_word_salad())

# Set metadata
video.set_default_language("en-US")

# =============================================================================
# PRIVACY AND SETTINGS
# =============================================================================

# Allow the video to be embedded on other sites
video.set_embeddable(True)

# Upload as PRIVATE by default (safer - won't accidentally publish)
# Options: "private", "unlisted", "public"
video.set_privacy_status("private")

# Allow view counts to be visible
video.set_public_stats_viewable(True)

# Optional: Set a thumbnail image
# video.set_thumbnail_path('/path/to/your/thumbnail.jpg')

# =============================================================================
# UPLOAD
# =============================================================================

# Upload the video and print results
uploaded_video = channel.upload_video(video)
print(f"Video ID: {uploaded_video.id}")
print(uploaded_video)

# Optional: Auto-like the video
# uploaded_video.like()
