import os
import time
import threading
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REFERENCE_CONSISTENCY_PREFIX = "Use the provided reference image only to preserve the same protagonist identity. Match the same face, hairstyle, age, school uniform, and overall silhouette as the reference image. Do not redesign the protagonist into a different girl, different anime character, or realistic photographic person."
MULTI_REFERENCE_GUIDANCE_PREFIX = "Treat the first reference image as the canonical protagonist identity anchor. If additional reference images are provided, use them only for scene continuity, object continuity, pose constraints, or environment carry-over. Do not let later reference images overwrite the protagonist's face, hairstyle, school uniform, or silhouette."
STYLE_ENFORCEMENT_PREFIX = "Render the image as a warm hand-painted Ghibli-like educational illustration with soft cinematic lighting, storybook warmth, and a polished anime-inspired look. Do not render as realistic photography, live-action film, commercial studio photo, or glossy product-shot realism."
QUALITY_ENFORCEMENT_PREFIX = "Keep the entire frame high-definition, clean, refined, complete, and visually polished in a 16:9 cinematic composition. Avoid noise, dirty textures, heavy oil-paint mess, blur, smudges, ghosting, cropped subjects, incomplete objects, and any visual contamination."
NO_TEXT_ENFORCEMENT_PREFIX = "Do not show any readable text, Chinese characters, English letters, numbers, formulas, punctuation, symbols, logos, watermarks, labels, subtitles, UI text, sign text, screen text, or book text anywhere in the image. If a sign, screen, package, scale, poster, or board appears, keep its shape blank with no recognizable marks."


class ParallelYunwuOpenAIImageGenerator:
    """Parallel image generator for yunwu.ai /v1/images/generations."""

    def __init__(self, api_keys: List[str], model: str = "gpt-image-2-all", api_base_url: Optional[str] = None):
        if not api_keys:
            raise ValueError("api_keys cannot be empty")

        self.api_keys = list(api_keys)
        self.model = model
        self.api_base_url = (api_base_url or os.getenv("YUNWU_OPENAI_IMAGE_BASE_URL", "https://yunwu.ai")).rstrip("/")

        self.lock = threading.Lock()
        self.current_key_index = 0
        self.api_usage = {i: 0 for i in range(len(self.api_keys))}

        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=max(20, len(self.api_keys) * 2),
            max_retries=retry_strategy,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        print(f"✓ 使用 {len(self.api_keys)} 个 yunwu.ai OpenAI 图像 API 密钥进行并行生成")
        print(f"✓ API 基础地址: {self.api_base_url}")
        print(f"✓ 图像模型: {self.model}")

    def _get_next_api_key(self) -> tuple:
        with self.lock:
            key = self.api_keys[self.current_key_index]
            index = self.current_key_index
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            self.api_usage[index] += 1
            return key, index

    def _is_doubao_seedream_model(self, model: str) -> bool:
        return (model or "").startswith("doubao-seedream-")

    def _aspect_ratio_to_size(self, aspect_ratio: str, model: Optional[str] = None) -> str:
        ratio = (aspect_ratio or "16:9").strip().lower()
        if self._is_doubao_seedream_model(model or self.model):
            mapping = {
                "1:1": "2048x2048",
                "16:9": "2848x1600",
                "9:16": "1600x2848",
                "3:2": "2496x1664",
                "2:3": "1664x2496",
                "4:3": "2304x1728",
                "3:4": "1728x2304",
                "21:9": "3136x1344",
            }
            return mapping.get(ratio, os.getenv("YUNWU_DOUBAO_DEFAULT_SIZE", "2848x1600"))

        mapping = {
            "1:1": "1024x1024",
            "16:9": "1536x1024",
            "9:16": "1024x1536",
            "3:2": "1536x1024",
            "2:3": "1024x1536",
        }
        return mapping.get(ratio, os.getenv("YUNWU_OPENAI_DEFAULT_SIZE", "1536x1024"))

    def generate_image(self, prompt: str, size: str = "16:9", reference_image_url: Optional[List[str]] = None, model: Optional[str] = None) -> Optional[str]:
        api_key, key_index = self._get_next_api_key()
        request_model = model or self.model
        request_size = self._aspect_ratio_to_size(size, request_model)
        endpoint = f"{self.api_base_url}/v1/images/generations"

        image_inputs = None
        if reference_image_url:
            if isinstance(reference_image_url, list):
                image_inputs = [item for item in reference_image_url if item]
            else:
                image_inputs = [reference_image_url]

        enhanced_prompt = f"{STYLE_ENFORCEMENT_PREFIX}\n\n{QUALITY_ENFORCEMENT_PREFIX}\n\n{NO_TEXT_ENFORCEMENT_PREFIX}\n\n{prompt}"
        if image_inputs:
            enhanced_prompt = f"{REFERENCE_CONSISTENCY_PREFIX}\n\n{MULTI_REFERENCE_GUIDANCE_PREFIX}\n\n{STYLE_ENFORCEMENT_PREFIX}\n\n{QUALITY_ENFORCEMENT_PREFIX}\n\n{NO_TEXT_ENFORCEMENT_PREFIX}\n\n{prompt}"

        payload = {
            "model": request_model,
            "size": request_size,
            "prompt": enhanced_prompt,
        }
        if self._is_doubao_seedream_model(request_model):
            payload.update({
                "output_format": "png",
                "response_format": "url",
                "watermark": False,
            })
        else:
            payload["n"] = 1
        if image_inputs:
            max_reference_images = 14 if self._is_doubao_seedream_model(request_model) else 5
            payload["image"] = image_inputs[:max_reference_images]

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            print("  🔍 请求参数:")
            print(f"    - model: {payload['model']}")
            print(f"    - size: {payload['size']}")
            print(f"    - prompt长度: {len(payload['prompt'])} 字符")
            if image_inputs:
                print(f"    - 参考图数量: {len(payload['image'])}")
                for i, url in enumerate(payload["image"]):
                    print(f"      [{i+1}] {url[:80]}...")

            response = self.session.post(endpoint, headers=headers, json=payload, timeout=180)
            if response.status_code != 200:
                print(f"⚠️ API密钥{key_index + 1}返回错误: {response.status_code} - {response.text}")
                return None

            result = response.json()
            data = result.get("data") or []
            if not data:
                print(f"⚠️ API密钥{key_index + 1}未返回图像数据: {result}")
                return None

            image_url = data[0].get("url")
            if image_url:
                print(f"✓ 使用API密钥{key_index + 1}成功生成图像")
                return image_url

            print(f"⚠️ API密钥{key_index + 1}返回结果缺少 url: {result}")
            return None
        except Exception as e:
            print(f"❌ 使用API密钥{key_index + 1}生成图像失败: {e}")
            return None

    def _download_image(self, image_url: str, output_dir: str, node_id: str, max_retries: int = 3) -> Optional[str]:
        os.makedirs(output_dir, exist_ok=True)
        local_image_path = os.path.join(output_dir, f"{node_id}.png")

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"  🔄 下载重试 {attempt + 1}/{max_retries} ({node_id})...")
                    time.sleep(2)

                response = self.session.get(image_url, timeout=60, stream=True)
                if response.status_code != 200:
                    print(f"⚠️ 下载失败: HTTP {response.status_code} ({node_id})")
                    continue

                with open(local_image_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                if os.path.exists(local_image_path) and os.path.getsize(local_image_path) > 0:
                    print(f"💾 图像已下载: {local_image_path}")
                    return local_image_path

                print(f"⚠️ 文件保存失败或文件为空 ({node_id})")
            except Exception as e:
                print(f"❌ 下载图像异常 ({node_id}): {e}")

        print(f"❌ 下载失败，已尝试 {max_retries} 次 ({node_id})")
        return None

    def generate_batch(self, generation_tasks: List[Dict], nodes: List[Dict], max_workers: Optional[int] = None, batch_label: str = "", model: Optional[str] = None) -> Dict[str, str]:
        request_model = model or self.model
        if max_workers is None:
            max_workers = len(self.api_keys)

        tasks = []
        for task in generation_tasks:
            node_id = task.get("node_id")
            node = next((n for n in nodes if n.get("id") == node_id), None)
            if not node:
                continue
            image_prompt = task.get("image_prompt", "")
            reference_url = task.get("reference_image_url")
            tasks.append((node, node_id, image_prompt, reference_url))

        if not tasks:
            return {}

        label = f"[{batch_label}] " if batch_label else ""
        print(f"\n{label}🚀 开始并行生成 {len(tasks)} 个节点的图像URL...")

        start_time = time.time()
        completed = 0
        url_results: Dict[str, str] = {}
        max_retry_rounds = 3
        pending_tasks = list(tasks)

        for retry_round in range(max_retry_rounds + 1):
            if not pending_tasks:
                break

            if retry_round > 0:
                print(f"\n{label}🔄 第 {retry_round} 轮重试，剩余 {len(pending_tasks)} 个失败任务...")
                time.sleep(10)

            failed_tasks = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {}
                for task in pending_tasks:
                    node, node_id, image_prompt, reference_url = task
                    future = executor.submit(self.generate_image, image_prompt, "16:9", reference_url, request_model)
                    future_to_task[future] = (node, node_id, image_prompt, reference_url)

                for future in as_completed(future_to_task):
                    node, node_id, image_prompt, reference_url = future_to_task[future]
                    try:
                        image_url = future.result()
                        if image_url:
                            url_results[node_id] = image_url
                            if "metadata" not in node:
                                node["metadata"] = {}
                            node["metadata"]["image_url"] = image_url
                            completed += 1
                            print(f"{label}[{completed}/{len(tasks)}] 已生成节点 {node_id} 的图像URL")
                        else:
                            msg = "，将在下一轮重试" if retry_round < max_retry_rounds else "，已达最大重试次数"
                            print(f"{label}⚠️ 节点 {node_id} 生成图像失败{msg}")
                            failed_tasks.append((node, node_id, image_prompt, reference_url))
                    except Exception as e:
                        msg = "，将在下一轮重试" if retry_round < max_retry_rounds else ""
                        print(f"{label}❌ 节点 {node_id} 生成失败: {e}{msg}")
                        failed_tasks.append((node, node_id, image_prompt, reference_url))

            pending_tasks = failed_tasks

        if pending_tasks:
            failed_ids = [t[1] for t in pending_tasks]
            print(f"\n{label}⚠️ {len(pending_tasks)} 个节点在 {max_retry_rounds} 轮重试后仍然失败: {', '.join(failed_ids)}")

        elapsed = time.time() - start_time
        print(f"{label}✅ 已生成 {completed}/{len(tasks)} 个图像URL ({elapsed:.2f}秒)")
        return url_results
