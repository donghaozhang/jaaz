#!/usr/bin/env python3
"""
Quick test script for image generation
"""
import sys
import os
import asyncio
from pathlib import Path

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(__file__))

from tools.image_generators import generate_image
from services.config_service import config_service

async def test_image_generation():
    print("🧪 Testing Image Generation...")
    
    # Check config
    print("📋 Current config:")
    config = config_service.get_config()
    for provider in ['replicate', 'fal', 'anthropic']:
        if provider in config:
            api_key = config[provider].get('api_key', '')
            masked_key = api_key[:8] + "..." + api_key[-4:] if api_key else "NOT SET"
            print(f"  {provider}: {masked_key}")
    
    print("\n🎨 Testing image generation...")
    
    try:
        # Test with simple prompt
        result = await generate_image.ainvoke({
            'prompt': 'A beautiful red dragon with golden scales',
            'aspect_ratio': '1:1',
            'tool_call_id': 'test_123'
        })
        
        print(f"✅ SUCCESS! Generated image: {result}")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_image_generation())