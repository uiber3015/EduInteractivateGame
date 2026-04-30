import os
import io
import asyncio
import itertools
import threading
from typing import List, Optional

import requests
from PIL import Image
from google import genai
from google.genai import types


class YunwuImageGenerator:
    def __init__(self, api_keys: List[str], model: str = "gemini-3-pro-image-preview"):
        if not api_keys:
            raise ValueError("api_keys 不能为空")
        self.api_keys = list(api_keys)
        self.model = model
        self._key_cycle = itertools.cycle(range(len(self.api_keys)))
        self._key_lock = threading.Lock()
        self._clients = []

        for key in self.api_keys:
            client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    base_url=os.getenv("YUNWU_IMAGE_BASE_URL", "https://yunwu.ai"),
                    api_version=os.getenv("YUNWU_IMAGE_API_VERSION", "v1beta"),
                ),
            )
            self._clients.append(client)

    def _get_next_client(self):
        with self._key_lock:
            idx = next(self._key_cycle)
        return self._clients[idx]

    async def _generate_async(self, prompt: str, reference_image_paths: Optional[List[str]] = None, aspect_ratio: str = "1:1") -> Image.Image:
        client = self._get_next_client()
        reference_images = []
        for path in reference_image_paths or []:
            if path.startswith("http"):
                response = requests.get(path, timeout=30)
                response.raise_for_status()
                reference_images.append(Image.open(io.BytesIO(response.content)))
            elif os.path.exists(path):
                reference_images.append(Image.open(path))

        response = await client.aio.models.generate_content(
            model=self.model,
            contents=reference_images + [prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
            ),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return part.as_image()
        raise ValueError("yunwu 未返回图像数据")

    def generate_image(self, prompt: str, output_path: str, reference_image_paths: Optional[List[str]] = None, aspect_ratio: str = "1:1") -> str:
        image = asyncio.run(self._generate_async(prompt, reference_image_paths=reference_image_paths, aspect_ratio=aspect_ratio))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        return output_path
