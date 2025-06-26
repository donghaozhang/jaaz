#!/usr/bin/env python3

import asyncio
import httpx
import json

async def test_chat_api():
    """Test full chat API with image generation"""
    
    print("💬 Testing Full Chat API...")
    
    base_url = "http://localhost:57988"
    
    # Test payload matching frontend format
    payload = {
        "messages": [{"role": "user", "content": "Generate a simple red dragon"}],
        "canvas_id": "test-canvas",
        "session_id": "test-session-api",
        "text_model": {
            "provider": "anthropic",
            "model": "anthropic/claude-3.5-sonnet",
            "type": "text"
        },
        "image_model": {
            "provider": "replicate", 
            "model": "black-forest-labs/flux-dev",
            "type": "image"
        },
        "system_prompt": "You are a helpful assistant that generates images when requested."
    }
    
    print(f"   📤 Sending chat request...")
    print(f"      Session: {payload['session_id']}")
    print(f"      Text Model: {payload['text_model']['model']}")
    print(f"      Image Model: {payload['image_model']['model']}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/api/chat",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            print(f"   📥 Response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"   ✅ Response: {result}")
                    return True
                except:
                    print(f"   📄 Response text: {response.text}")
                    return True
            else:
                print(f"   ❌ Error response: {response.text}")
                return False
                
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_chat_api())