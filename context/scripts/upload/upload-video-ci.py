"""
upload-video-ci.py - CI/CD Version for GitHub Actions

This is an adapted version of upload-video.py designed to run in
automated CI/CD environments like GitHub Actions.

DIFFERENCES FROM upload-video.py:
1. Uses environment variables for configuration
2. Better error handling for CI environments
3. Explicit logging for debugging workflow runs
4. Supports PRIVACY_STATUS env var for flexible privacy control

USAGE:
This script is called by the GitHub Actions workflow.
It expects credentials files to already exist (created from secrets).
"""

from simple_youtube_api.Channel import Channel
from simple_youtube_api.LocalVideo import LocalVideo
import os
import random
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory containing the output videos
video_directory_path = os.environ.get('VIDEO_DIRECTORY', 'output')

# Privacy status from environment (default: private)
privacy_status = os.environ.get('PRIVACY_STATUS', 'private')

# =============================================================================
# LOGGING HELPERS
# =============================================================================

def log(message):
    """Print with flush for real-time CI logs."""
    print(message, flush=True)

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
        "elixir",       # magical potion
        "velvet",       # soft texture
        "aurora",       # northern lights
        "gossamer",     # delicate fabric
        "phosphene",    # light you see when eyes closed
        "petrichor",    # smell after rain
        "halcyon",      # peaceful/golden
        "liminal",      # threshold/in-between
        "vermillion",   # vivid red-orange
        "cerulean",     # sky blue
        "obsidian"      # volcanic glass
    ]
    return ' '.join(random.sample(words, n))


# =============================================================================
# FIND THE VIDEO
# =============================================================================

log(f"Looking for videos in: {video_directory_path}")

# Check if directory exists
if not os.path.exists(video_directory_path):
    log(f"Error: Directory '{video_directory_path}' does not exist")
    sys.exit(1)

# Get all files in the output directory
files = [
    os.path.join(video_directory_path, f) 
    for f in os.listdir(video_directory_path) 
    if os.path.isfile(os.path.join(video_directory_path, f))
]

# Filter to only MP4 files
mp4_files = [f for f in files if f.endswith('.mp4')]

if not mp4_files:
    log(f"Error: No MP4 files found in '{video_directory_path}'")
    sys.exit(1)

log(f"Found {len(mp4_files)} MP4 file(s)")

# Find the most recently modified file
newest_file = max(mp4_files, key=os.path.getmtime)
log(f"Selected video: {newest_file}")

# =============================================================================
# YOUTUBE AUTHENTICATION
# =============================================================================

log("Authenticating with YouTube...")

# Check for credential files
if not os.path.exists("client_secret.json"):
    log("Error: client_secret.json not found")
    sys.exit(1)

if not os.path.exists("credentials.storage"):
    log("Error: credentials.storage not found")
    sys.exit(1)

try:
    channel = Channel()
    channel.login("client_secret.json", "credentials.storage")
    log("Authentication successful")
except Exception as e:
    log(f"Authentication failed: {e}")
    sys.exit(1)

# =============================================================================
# SET UP THE VIDEO
# =============================================================================

log("Preparing video for upload...")

video = LocalVideo(file_path=newest_file)

# Generate and set title
title = generate_word_salad()
video.set_title(title)
log(f"Title: {title}")

# Set metadata
video.set_default_language("en-US")

# =============================================================================
# PRIVACY AND SETTINGS
# =============================================================================

video.set_embeddable(True)
video.set_privacy_status(privacy_status)
video.set_public_stats_viewable(True)

log(f"Privacy status: {privacy_status}")

# =============================================================================
# UPLOAD
# =============================================================================

log("Starting upload...")

try:
    uploaded_video = channel.upload_video(video)
    log(f"Upload successful!")
    log(f"Video ID: {uploaded_video.id}")
    log(f"Video URL: https://www.youtube.com/watch?v={uploaded_video.id}")
except Exception as e:
    log(f"Upload failed: {e}")
    sys.exit(1)

log("Done!")




