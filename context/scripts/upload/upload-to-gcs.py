"""
upload-to-gcs.py - Upload videos to Google Cloud Storage

This script uploads your output videos to a GCS bucket and generates
a video-manifest.json file for the GitHub Actions workflow.

SETUP:
1. Install: pip install google-cloud-storage
2. Create a GCS bucket at https://console.cloud.google.com/storage
3. Create a service account with Storage Admin role
4. Download the JSON key and save as gcs-key.json

USAGE:
    python upload-to-gcs.py --bucket YOUR_BUCKET_NAME

OPTIONS:
    --bucket     GCS bucket name (required)
    --limit      Max number of videos to upload (default: all)
    --prefix     Path prefix in bucket (default: videos/)
"""

import os
import sys
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Upload videos to Google Cloud Storage')
    parser.add_argument('--bucket', required=True, help='GCS bucket name')
    parser.add_argument('--limit', type=int, default=None, help='Max videos to upload')
    parser.add_argument('--prefix', default='videos/', help='Path prefix in bucket')
    parser.add_argument('--key', default='gcs-key.json', help='Service account key file')
    parser.add_argument('--source', default='output', help='Source directory')
    args = parser.parse_args()

    # Check for google-cloud-storage
    try:
        from google.cloud import storage
    except ImportError:
        print("Installing google-cloud-storage...")
        os.system(f"{sys.executable} -m pip install google-cloud-storage")
        from google.cloud import storage

    # Check for key file
    if not os.path.exists(args.key):
        print(f"Error: Service account key file '{args.key}' not found")
        print("\nTo create one:")
        print("1. Go to https://console.cloud.google.com/iam-admin/serviceaccounts")
        print("2. Create a service account with 'Storage Admin' role")
        print("3. Create a JSON key and save as gcs-key.json")
        sys.exit(1)

    # Initialize client
    client = storage.Client.from_service_account_json(args.key)
    bucket = client.bucket(args.bucket)

    # Get video files
    source_dir = Path(args.source)
    video_files = sorted(source_dir.glob('*.mp4'), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if args.limit:
        video_files = video_files[:args.limit]

    print(f"Found {len(video_files)} videos to upload")
    
    manifest = {
        "description": "Videos uploaded to Google Cloud Storage",
        "bucket": args.bucket,
        "videos": []
    }

    for i, video_path in enumerate(video_files):
        blob_name = f"{args.prefix}{video_path.name}"
        blob = bucket.blob(blob_name)
        
        # Check if already uploaded
        if blob.exists():
            print(f"[{i+1}/{len(video_files)}] Already exists: {video_path.name}")
        else:
            print(f"[{i+1}/{len(video_files)}] Uploading: {video_path.name}")
            blob.upload_from_filename(str(video_path))
        
        # Make publicly readable
        blob.make_public()
        
        manifest["videos"].append({
            "url": blob.public_url,
            "name": video_path.stem
        })

    # Save manifest
    with open('video-manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nDone! Uploaded {len(video_files)} videos")
    print(f"Manifest saved to video-manifest.json")
    print(f"\nPublic URL pattern: https://storage.googleapis.com/{args.bucket}/{args.prefix}FILENAME.mp4")

if __name__ == '__main__':
    main()




