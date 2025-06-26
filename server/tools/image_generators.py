import random
import base64
import json
import time
import traceback
import os
from mimetypes import guess_type
from typing import Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.runnables import RunnableConfig
import aiofiles
from nanoid import generate

from common import DEFAULT_PORT
from services.config_service import FILES_DIR
from services.db_service import db_service
from services.websocket_service import send_to_websocket, broadcast_session_update

# Import all generators
from .img_generators import (
    ReplicateGenerator,
    ComfyUIGenerator,
    WavespeedGenerator,
    JaazGenerator,
    OpenAIGenerator
)

# 生成唯一文件 ID
def generate_file_id():
    return 'im_' + generate(size=8)


class GenerateImageInputSchema(BaseModel):
    prompt: str = Field(
        description="Required. The prompt for image generation. If you want to edit an image, please describe what you want to edit in the prompt.")
    aspect_ratio: str = Field(
        description="Required. Aspect ratio of the image, only these values are allowed: 1:1, 16:9, 4:3, 3:4, 9:16 Choose the best fitting aspect ratio according to the prompt. Best ratio for posters is 3:4")
    input_image: Optional[str] = Field(default=None, description="Optional; Image to use as reference. Pass image_id here, e.g. 'im_jurheut7.png'. Best for image editing cases like: Editing specific parts of the image, Removing specific objects, Maintaining visual elements across scenes (character/object consistency), Generating new content in the style of the reference (style transfer), etc.")
    tool_call_id: Annotated[str, InjectedToolCallId]


# Initialize provider instances
PROVIDERS = {
    'replicate': ReplicateGenerator(),
    'comfyui': ComfyUIGenerator(),
    'wavespeed': WavespeedGenerator(),
    'jaaz': JaazGenerator(),
    'openai': OpenAIGenerator(),
}


@tool("generate_image",
      description="Generate an image using text prompt or optionally pass an image for reference or for editing",
      args_schema=GenerateImageInputSchema)
async def generate_image(
    prompt: str,
    aspect_ratio: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    input_image: Optional[str] = None,
) -> str:
    print('🛠️ tool_call_id', tool_call_id)
    print('🔍 Config type:', type(config))
    print('🔍 Full config:', config)
    
    try:
        ctx = config.get('configurable', {})
        print('🔍 Context:', ctx)
    except Exception as e:
        print('❌ Error getting configurable:', e)
        ctx = {}
    canvas_id = ctx.get('canvas_id', '')
    session_id = ctx.get('session_id', '')
    print('🛠️canvas_id', canvas_id, 'session_id', session_id)
    # Inject the tool call id into the context
    ctx['tool_call_id'] = tool_call_id

    model_info = ctx.get('model_info', {})
    print('🔍 Model info:', model_info)
    if model_info is None:
        print('❌ model_info is None!')
        raise ValueError("model_info is None")
    
    image_model = model_info.get('image', {})
    print('🔍 Image model:', image_model)
    if image_model is None:
        raise ValueError("Image model is not selected")
    model = image_model.get('model', '')
    provider = image_model.get('provider', 'replicate')

    # Get provider instance
    generator = PROVIDERS.get(provider)
    if not generator:
        raise ValueError(f"Unsupported provider: {provider}")

    try:
        # Prepare input image if provided
        input_image_data = None
        if input_image:
            image_path = os.path.join(FILES_DIR, f'{input_image}')

            if provider == 'openai':
                # OpenAI needs file path
                input_image_data = image_path
            else:
                # Other providers need base64
                async with aiofiles.open(image_path, 'rb') as f:
                    image_data = await f.read()
                b64 = base64.b64encode(image_data).decode('utf-8')
                mime_type, _ = guess_type(image_path)
                if not mime_type:
                    mime_type = "image/png"
                input_image_data = f"data:{mime_type};base64,{b64}"

        # Generate image using the appropriate provider
        extra_kwargs = {}
        if provider == 'comfyui':
            extra_kwargs['ctx'] = ctx
        elif provider == 'wavespeed':
            extra_kwargs['aspect_ratio'] = aspect_ratio

        mime_type, width, height, filename = await generator.generate(
            prompt=prompt,
            model=model,
            aspect_ratio=aspect_ratio,
            input_image=input_image_data,
            **extra_kwargs
        )

        # Use filename as the file_id for consistency
        file_id = filename
        url = f'/api/file/{filename}'

        file_data = {
            'mimeType': mime_type,
            'id': file_id,
            'dataURL': url,
            'created': int(time.time() * 1000),
        }

        # Add image to canvas if canvas_id is provided
        if canvas_id:
            try:
                # Create canvas element for the generated image
                image_element = await generate_new_image_element(canvas_id, file_id, {
                    'width': width,
                    'height': height,
                    'mimeType': mime_type
                })
                
                if image_element:
                    # Add the image element to the canvas
                    canvas = await db_service.get_canvas_data(canvas_id)
                    if canvas:
                        canvas_data = canvas.get('data', {})
                        elements = canvas_data.get('elements', [])
                        elements.append(image_element)
                        
                        # Update canvas with new element
                        canvas_data['elements'] = elements
                        await db_service.save_canvas_data(canvas_id, json.dumps(canvas_data))
                        
                        # Notify frontend about canvas update with proper format for Excalidraw
                        await broadcast_session_update(session_id, canvas_id, {
                            'type': 'image_generated',
                            'canvas_id': canvas_id,
                            'element': image_element,
                            'file': {
                                'id': file_id,
                                'dataURL': url,
                                'mimeType': mime_type,
                                'created': int(time.time() * 1000),
                            }
                        })
                        
                        print(f"🎨 Image added to canvas: {canvas_id}")
                    else:
                        print(f"⚠️  Canvas {canvas_id} not found")
                else:
                    print(f"⚠️  Failed to create canvas element")
            except Exception as canvas_error:
                print(f"⚠️  Canvas integration error: {canvas_error}")
                # Continue even if canvas integration fails
        
        # Use relative URL that works through frontend proxy
        image_url = f"/api/file/{filename}"
        
        print(f"✅ Image generated successfully: {filename}")
        print(f"📁 File path: /home/zdhpe/GenAI-tool/jaaz-source/server/user_data/files/{filename}")
        print(f"🌐 Image URL: {image_url}")

        return f"✅ Image generated successfully! ![{filename}]({image_url})"

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        traceback.print_exc()
        await send_to_websocket(session_id, {
            'type': 'error',
            'error': str(e)
        })
        return f"image generation failed: {str(e)}"

print('🛠️', generate_image.args_schema.model_json_schema())

async def generate_new_image_element(canvas_id: str, fileid: str, image_data: dict):
    canvas = await db_service.get_canvas_data(canvas_id)
    if canvas is None:
        print(f"⚠️  Canvas {canvas_id} not found, skipping canvas element creation")
        return
    
    canvas_data = canvas.get('data', {})
    elements = canvas_data.get('elements', [])

    # find the last image element
    last_x = 0
    last_y = 0
    last_width = 0
    last_height = 0
    image_elements = [
        element for element in elements if element.get('type') == 'image']
    last_image_element = image_elements[-1] if len(
        image_elements) > 0 else None
    if last_image_element is not None:
        last_x = last_image_element.get('x', 0)
        last_y = last_image_element.get('y', 0)
        last_width = last_image_element.get('width', 0)
        last_height = last_image_element.get('height', 0)

    new_x = last_x + last_width + 20

    return {
        'type': 'image',
        'id': fileid,
        'x': new_x,
        'y': last_y,
        'width': image_data.get('width', 0),
        'height': image_data.get('height', 0),
        'angle': 0,
        'fileId': fileid,
        'strokeColor': '#000000',
        'fillStyle': 'solid',
        'strokeStyle': 'solid',
        'boundElements': None,
        'roundness': None,
        'frameId': None,
        'backgroundColor': 'transparent',
        'strokeWidth': 1,
        'roughness': 0,
        'opacity': 100,
        'groupIds': [],
        'seed': int(random.random() * 1000000),
        'version': 1,
        'versionNonce': int(random.random() * 1000000),
        'isDeleted': False,
        'index': None,
        'updated': 0,
        'link': None,
        'locked': False,
        'status': 'saved',
        'scale': [1, 1],
        'crop': None,
    }
