#!/usr/bin/env python3
"""
Test real image generation and local saving
"""
import sys
import os
import asyncio
import time

sys.path.append(os.path.dirname(__file__))

from tools.image_generators import generate_image
from services.config_service import config_service
from langchain_core.tools import ToolException

async def test_real_image_generation():
    print("🎨 Testing Real Image Generation...")
    
    # Check configuration
    config = config_service.get_config()
    replicate_key = config.get('replicate', {}).get('api_key', '')
    
    if not replicate_key:
        print("❌ No Replicate API key found")
        return
    
    print(f"✅ Replicate API key: {replicate_key[:8]}...")
    
    # Test cases
    test_prompts = [
        {
            "prompt": "A beautiful red dragon with golden scales, fantasy art style, detailed artwork",
            "aspect_ratio": "1:1",
            "description": "Simple dragon test"
        },
        {
            "prompt": "A cute blue cat sitting in a garden, digital art, high quality",
            "aspect_ratio": "16:9", 
            "description": "Cat in garden"
        }
    ]
    
    for i, test in enumerate(test_prompts):
        print(f"\n🧪 Test {i+1}: {test['description']}")
        print(f"📝 Prompt: {test['prompt']}")
        print(f"📐 Aspect ratio: {test['aspect_ratio']}")
        
        start_time = time.time()
        
        try:
            # Call the image generation tool
            result = await generate_image.ainvoke({
                'prompt': test['prompt'],
                'aspect_ratio': test['aspect_ratio'],
                'tool_call_id': f'test_real_{i+1}'
            })
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✅ SUCCESS! Generated in {duration:.1f}s")
            print(f"📁 Result: {result}")
            
            # Check if file exists
            if "im_" in result:
                image_filename = result.split("im_")[1].split()[0] if "im_" in result else "unknown"
                image_path = f"user_data/files/im_{image_filename}"
                
                if os.path.exists(image_path):
                    file_size = os.path.getsize(image_path)
                    print(f"📂 File saved: {image_path} ({file_size} bytes)")
                else:
                    print(f"❓ File not found at expected path: {image_path}")
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"❌ FAILED after {duration:.1f}s")
            print(f"🔍 Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("-" * 50)
    
    # List generated files
    print("\n📁 Generated files in user_data/files/:")
    files_dir = "user_data/files"
    if os.path.exists(files_dir):
        files = os.listdir(files_dir)
        image_files = [f for f in files if f.startswith('im_')]
        
        if image_files:
            for file in sorted(image_files):
                file_path = os.path.join(files_dir, file)
                size = os.path.getsize(file_path)
                print(f"  📎 {file} ({size} bytes)")
        else:
            print("  ❌ No image files found")
    else:
        print(f"  ❌ Directory {files_dir} not found")

if __name__ == "__main__":
    asyncio.run(test_real_image_generation())