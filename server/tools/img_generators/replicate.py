from typing import Optional
import os
import asyncio
import traceback
from .base import ImageGenerator, get_image_info_and_save, generate_image_id
from services.config_service import config_service, FILES_DIR
from utils.http_client import HttpClient


class ReplicateGenerator(ImageGenerator):
    """Replicate image generator implementation"""

    async def generate(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str = "1:1",
        input_image: Optional[str] = None,
        **kwargs
    ) -> tuple[str, int, int, str]:
        try:
            api_key = config_service.app_config.get(
                'replicate', {}).get('api_key', '')
            if not api_key:
                raise ValueError(
                    "Image generation failed: Replicate API key is not set")

            url = f"https://api.replicate.com/v1/models/{model}/predictions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Prefer": "wait"
            }
            data = {
                "input": {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                }
            }

            if input_image:
                data['input']['input_image'] = input_image
                model = 'black-forest-labs/flux-kontext-pro'

            async with HttpClient.create() as client:
                response = await client.post(url, headers=headers, json=data)
                res = response.json()

                # Handle both immediate completion (200) and async processing (201)
                if response.status_code == 201:
                    # Job started, wait for completion
                    get_url = res.get('urls', {}).get('get', '')
                    if not get_url:
                        raise Exception('Replicate job started but no get URL provided')
                    
                    # Poll for completion (max 60 seconds)
                    for attempt in range(60):
                        await asyncio.sleep(1)
                        poll_response = await client.get(get_url, headers=headers)
                        poll_res = poll_response.json()
                        
                        status = poll_res.get('status', '')
                        if status == 'succeeded':
                            res = poll_res
                            break
                        elif status in ['failed', 'canceled']:
                            error_msg = poll_res.get('error', 'Unknown error')
                            raise Exception(f'Replicate generation failed: {error_msg}')
                        # Continue polling if status is 'starting' or 'processing'
                    
                    if poll_res.get('status') != 'succeeded':
                        raise Exception('Replicate generation timed out after 60 seconds')

            output = res.get('output', '')
            if not output:
                if res.get('detail', '') != '':
                    raise Exception(
                        f'Replicate image generation failed: {res.get("detail", "")}')
                else:
                    raise Exception(
                        'Replicate image generation failed: no output url found')

            # Handle both single URL and list of URLs
            if isinstance(output, list):
                image_url = output[0]  # Use first image
            else:
                image_url = output  # Single URL

            image_id = generate_image_id()
            print('🦄image generation image_id', image_id)

            # get image dimensions
            mime_type, width, height, extension = await get_image_info_and_save(
                image_url, os.path.join(FILES_DIR, f'{image_id}')
            )
            filename = f'{image_id}.{extension}'
            return mime_type, width, height, filename

        except Exception as e:
            print('Error generating image with replicate', e)
            traceback.print_exc()
            raise e
