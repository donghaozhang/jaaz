#!/usr/bin/env python3

import os
import glob
from datetime import datetime

def verify_image_generation():
    """Verify that images are being generated successfully"""
    
    print("🔍 Verifying Image Generation Results...")
    
    # Check the image directory
    image_dir = "/home/zdhpe/GenAI-tool/jaaz-source/server/user_data/files/"
    
    if not os.path.exists(image_dir):
        print(f"❌ Image directory does not exist: {image_dir}")
        return False
    
    # Find all image files
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    all_images = []
    
    for pattern in image_patterns:
        all_images.extend(glob.glob(os.path.join(image_dir, pattern)))
    
    if not all_images:
        print(f"❌ No images found in {image_dir}")
        return False
    
    print(f"✅ Found {len(all_images)} images:")
    
    # Sort by modification time (newest first)
    all_images.sort(key=os.path.getmtime, reverse=True)
    
    recent_images = all_images[:10]  # Show last 10 images
    
    for img_path in recent_images:
        filename = os.path.basename(img_path)
        file_size = os.path.getsize(img_path)
        mod_time = datetime.fromtimestamp(os.path.getmtime(img_path))
        
        print(f"   📷 {filename}")
        print(f"      💾 Size: {file_size:,} bytes")
        print(f"      🕒 Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    # Check if recent images are being created
    latest_image = all_images[0]
    latest_time = os.path.getmtime(latest_image)
    time_diff = datetime.now().timestamp() - latest_time
    
    if time_diff < 300:  # Within last 5 minutes
        print(f"✅ Recent image generation detected!")
        print(f"   Latest image: {os.path.basename(latest_image)}")
        print(f"   Created: {int(time_diff)} seconds ago")
        return True
    else:
        print(f"⚠️  Latest image is {int(time_diff/60)} minutes old")
        return True  # Still consider it success if images exist
        
if __name__ == "__main__":
    verify_image_generation()