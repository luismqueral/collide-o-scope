"""
upload-instagram-ci.py - Auto-publish Reels to Instagram

This script uploads videos as Instagram Reels using the Graph API.

REQUIREMENTS:
1. Instagram Professional account (Business or Creator)
2. Connected to a Facebook Page
3. Facebook App with Instagram Graph API permissions

SETUP:
1. Go to https://developers.facebook.com/apps/ and create an app
2. Add "Instagram Graph API" product
3. Get a long-lived access token with these permissions:
   - instagram_basic
   - instagram_content_publish
   - pages_read_engagement
4. Get your Instagram Business Account ID

VIDEO REQUIREMENTS:
- Format: MP4, MOV
- Aspect ratio: 9:16 (vertical) for Reels
- Duration: 3-90 seconds
- Max size: 1GB

ENVIRONMENT VARIABLES:
- INSTAGRAM_ACCESS_TOKEN: Long-lived access token
- INSTAGRAM_ACCOUNT_ID: Your Instagram Business Account ID
- VIDEO_URL: Public URL to the video file
"""

import os
import sys
import time
import random
import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

ACCESS_TOKEN = os.environ.get('INSTAGRAM_ACCESS_TOKEN')
ACCOUNT_ID = os.environ.get('INSTAGRAM_ACCOUNT_ID')
VIDEO_URL = os.environ.get('VIDEO_URL')

GRAPH_API_VERSION = 'v18.0'
BASE_URL = f'https://graph.facebook.com/{GRAPH_API_VERSION}'

# =============================================================================
# LOGGING
# =============================================================================

def log(message):
    """Print with flush for real-time CI logs."""
    print(message, flush=True)

# =============================================================================
# CAPTION GENERATOR
# =============================================================================

def generate_caption():
    """Generate an aesthetic caption with hashtags."""
    
    phrases = [
        "signal to noise",
        "static memories",
        "phosphor dreams",
        "analog artifacts",
        "digital decay",
        "magnetic resonance",
        "cathode visions",
        "tape loops",
        "scan lines",
        "ghost frames",
        "color bleed",
        "sync pulse",
        "vertical hold",
        "horizontal drift",
        "luminance fade"
    ]
    
    hashtags = [
        "#videoart",
        "#experimentalvideo",
        "#glitchart",
        "#analogvideo",
        "#vhs",
        "#videosynthesis",
        "#abstractart",
        "#generativeart",
        "#newmediaart",
        "#digitalart"
    ]
    
    phrase = random.choice(phrases)
    selected_tags = ' '.join(random.sample(hashtags, 5))
    
    return f"{phrase}\n\n{selected_tags}"

# =============================================================================
# INSTAGRAM API FUNCTIONS
# =============================================================================

def create_media_container(video_url, caption):
    """
    Step 1: Create a media container for the Reel.
    Instagram will fetch the video from the URL.
    """
    log("Creating media container...")
    
    url = f"{BASE_URL}/{ACCOUNT_ID}/media"
    params = {
        'media_type': 'REELS',
        'video_url': video_url,
        'caption': caption,
        'access_token': ACCESS_TOKEN,
        'share_to_feed': 'true'  # Also show in main feed
    }
    
    response = requests.post(url, params=params)
    data = response.json()
    
    if 'error' in data:
        log(f"Error creating container: {data['error']['message']}")
        sys.exit(1)
    
    container_id = data['id']
    log(f"Container created: {container_id}")
    return container_id

def check_container_status(container_id):
    """
    Step 2: Wait for Instagram to process the video.
    """
    log("Waiting for video processing...")
    
    url = f"{BASE_URL}/{container_id}"
    params = {
        'fields': 'status_code,status',
        'access_token': ACCESS_TOKEN
    }
    
    max_attempts = 30  # 5 minutes max
    for attempt in range(max_attempts):
        response = requests.get(url, params=params)
        data = response.json()
        
        status = data.get('status_code', 'UNKNOWN')
        log(f"  Status: {status} (attempt {attempt + 1}/{max_attempts})")
        
        if status == 'FINISHED':
            log("Video processing complete!")
            return True
        elif status == 'ERROR':
            log(f"Processing failed: {data.get('status', 'Unknown error')}")
            return False
        
        time.sleep(10)  # Wait 10 seconds between checks
    
    log("Timeout waiting for video processing")
    return False

def publish_media(container_id):
    """
    Step 3: Publish the processed media container.
    """
    log("Publishing to Instagram...")
    
    url = f"{BASE_URL}/{ACCOUNT_ID}/media_publish"
    params = {
        'creation_id': container_id,
        'access_token': ACCESS_TOKEN
    }
    
    response = requests.post(url, params=params)
    data = response.json()
    
    if 'error' in data:
        log(f"Error publishing: {data['error']['message']}")
        sys.exit(1)
    
    media_id = data['id']
    log(f"Published! Media ID: {media_id}")
    return media_id

def get_permalink(media_id):
    """Get the public URL of the published post."""
    url = f"{BASE_URL}/{media_id}"
    params = {
        'fields': 'permalink',
        'access_token': ACCESS_TOKEN
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    return data.get('permalink', 'URL not available')

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Validate environment
    if not ACCESS_TOKEN:
        log("Error: INSTAGRAM_ACCESS_TOKEN not set")
        sys.exit(1)
    
    if not ACCOUNT_ID:
        log("Error: INSTAGRAM_ACCOUNT_ID not set")
        sys.exit(1)
    
    if not VIDEO_URL:
        log("Error: VIDEO_URL not set")
        sys.exit(1)
    
    log(f"Video URL: {VIDEO_URL}")
    
    # Generate caption
    caption = generate_caption()
    log(f"Caption: {caption[:50]}...")
    
    # Upload flow
    container_id = create_media_container(VIDEO_URL, caption)
    
    if not check_container_status(container_id):
        log("Failed to process video")
        sys.exit(1)
    
    media_id = publish_media(container_id)
    permalink = get_permalink(media_id)
    
    log(f"\n✅ Success!")
    log(f"View post: {permalink}")

if __name__ == '__main__':
    main()




