"""
upload-video-default.py - Upload to YouTube with Default Settings

A simplified version of upload-video.py that uses default settings.
This file appears to be a backup/alternative version of the uploader.

See upload-video.py for detailed documentation on how YouTube uploading works.
"""

from simple_youtube_api.Channel import Channel
from simple_youtube_api.LocalVideo import LocalVideo
import os
import random

# Directory containing output videos
video_directory_path = 'output'


def generate_word_salad(n=5):
    """Generate a random title from evocative words."""
    words = [
        "ephemeral", "cascade", "nebula", "quixotic", "serendipity",
        "luminous", "sonder", "zenith", "ethereal", "elixir"
    ]
    return ' '.join(random.sample(words, n))


# Find all MP4 files
files = [
    os.path.join(video_directory_path, f) 
    for f in os.listdir(video_directory_path) 
    if os.path.isfile(os.path.join(video_directory_path, f))
]
mp4_files = [f for f in files if f.endswith('.mp4')]

# Get the newest file
newest_file = max(mp4_files, key=os.path.getmtime)

# Authenticate and upload
channel = Channel()
channel.login("client_secret.json", "credentials.storage")

video = LocalVideo(file_path=newest_file)
video.set_title(generate_word_salad())
video.set_default_language("en-US")
video.set_embeddable(True)
video.set_privacy_status("private")
video.set_public_stats_viewable(True)

uploaded_video = channel.upload_video(video)
print(f"Video ID: {uploaded_video.id}")
print(uploaded_video)
