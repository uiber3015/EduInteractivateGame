import requests
import os
import json
import time
from datetime import datetime as dt
from typing import List, Optional, Dict, Any

# 导入SCDN图床上传工具
try:
    from scdn_image_uploader import convert_local_paths_to_urls
    SCDN_AVAILABLE = True
except ImportError:
    SCDN_AVAILABLE = False
    print("⚠️ SCDN图床上传模块不可用，本地图片将无法作为参考图片")


# === 配置部分 ===
# 多个API密钥，支持轮换
API_KEYS = [
    "sk-95356c6d03ff45c09c582004404e2fa2",
    "sk-2a4d123f95fe4f40947af288fa14cbb9",
    "sk-2bf3e79d3a1a4f5aa515350bbb0ca17b"
]

# API端点配置
API_HOST_OVERSEAS = "https://grsaiapi.com"
API_HOST_DOMESTIC = "https://grsai.dakka.com.cn"

# 默认使用国内节点
API_HOST = API_HOST_DOMESTIC

# 当前使用的API密钥索引
current_api_key_index = 0


def _get_draw_api_url(model: str) -> str:
    if model.startswith("nano-banana"):
        return f"{API_HOST}/v1/draw/nano-banana"
    return f"{API_HOST}/v1/draw/completions"


def get_next_api_key():
    """获取下一个可用的API密钥（轮换机制）"""
    global current_api_key_index
    api_key = API_KEYS[current_api_key_index]
    current_api_key_index = (current_api_key_index + 1) % len(API_KEYS)
    return api_key


def get_output_dir():
    """创建带时间戳的保存路径"""
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("data", "output_images", "sora_image", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _build_draw_request_data(model: str, prompt: str, size: str, variants: int, shut_progress: bool, image_size: Optional[str] = None) -> Dict[str, Any]:
    request_data = {
        "model": model,
        "prompt": prompt,
        "aspectRatio": size,
        "variants": variants,
        "shutProgress": shut_progress
    }
    if model.startswith("nano-banana"):
        request_data["imageSize"] = image_size or "1K"
    return request_data


def generate_image_url_sora(
    prompt: str,
    model: str = "gpt-image-2",
    size: str = "1:1",
    variants: int = 1,
    reference_urls: Optional[List[str]] = None,
    use_webhook: bool = False,
    shut_progress: bool = False,
    max_retries: int = 3,
    image_size: str = "1K"
) -> Optional[str]:
    """
    使用图像模型生成图片，只返回URL不下载
    
    Args:
        prompt: 图片生成提示词
        model: 模型名称，支持 "gpt-image-2" 或 "gpt-image-1.5"
        size: 图片比例，可选 "auto", "1:1", "3:2", "2:3"
        variants: 批量生成数量，可选 1 或 2
        reference_urls: 参考图片URL列表
        use_webhook: 是否使用webhook回调（False则使用流式响应）
        shut_progress: 是否关闭进度回复，直接返回最终结果
        max_retries: 最大重试次数（用于API key轮换）
    
    Returns:
        str: 生成的图片URL，如果失败则返回None
    """
    print(f"🎨 开始使用 {model} 生成图片（仅获取URL）")
    print(f"📝 提示词: {prompt}")
    print(f"📐 尺寸比例: {size}")
    print(f"🔢 生成数量: {variants}")
    
    # 处理参考图片：将本地路径转换为URL
    processed_urls = None
    if reference_urls:
        if SCDN_AVAILABLE:
            # 自动将本地路径转换为URL
            processed_urls = convert_local_paths_to_urls(reference_urls, use_cache=True)
            if processed_urls:
                print(f"📸 参考图片数量: {len(processed_urls)}")
                for i, url in enumerate(processed_urls):
                    print(f"   参考图片[{i+1}]: {url}")
            else:
                print("⚠️ 没有有效的参考图片URL")
        else:
            # SCDN不可用，只保留已经是URL的路径
            processed_urls = [url for url in reference_urls if url.startswith(('http://', 'https://'))]
            if processed_urls:
                print(f"📸 参考图片数量: {len(processed_urls)}")
                for i, url in enumerate(processed_urls):
                    print(f"   参考图片[{i+1}]: {url}")
            # 警告本地路径无法使用
            local_paths = [url for url in reference_urls if not url.startswith(('http://', 'https://'))]
            for path in local_paths:
                print(f"⚠️ 本地路径无法使用（SCDN不可用）: {path}")
    
    # 构建请求参数
    request_data = _build_draw_request_data(model, prompt, size, variants, shut_progress, image_size)
    
    # 添加参考图片URL
    if processed_urls:
        request_data["urls"] = processed_urls
    
    # 如果使用webhook，添加webhook参数（这里使用-1表示立即返回id）
    if use_webhook:
        request_data["webHook"] = "-1"
    
    # API端点
    api_url = _get_draw_api_url(model)
    
    # 尝试多次请求，支持API key轮换
    for attempt in range(max_retries):
        api_key = get_next_api_key()
        print(f"🔑 使用API密钥: {api_key[:15]}... (尝试 {attempt + 1}/{max_retries})")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        try:
            # 发送请求
            response = requests.post(
                api_url,
                headers=headers,
                json=request_data,
                timeout=120,
                stream=not use_webhook  # 如果不使用webhook，则使用流式响应
            )
            
            print(f"📡 响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                if use_webhook:
                    # webhook模式：立即返回id，需要轮询结果
                    result = response.json()
                    if result.get("code") == 0:
                        task_id = result.get("data", {}).get("id")
                        print(f"✅ 任务已提交，ID: {task_id}")
                        
                        # 轮询获取结果
                        image_url = poll_result(task_id, api_key)
                        if image_url:
                            print(f"✅ 获取到图片URL: {image_url}")
                            return image_url
                    else:
                        print(f"❌ 任务提交失败: {result.get('msg')}")
                else:
                    # 流式响应模式：解析流式数据
                    image_url = parse_stream_response(response)
                    if image_url:
                        print(f"✅ 获取到图片URL: {image_url}")
                        return image_url
                
                # 如果成功处理，跳出重试循环
                break
            else:
                print(f"❌ 请求失败: {response.status_code}")
                print(f"   响应内容: {response.text[:500]}")
                
                # 如果是最后一次尝试，返回None
                if attempt == max_retries - 1:
                    print(f"❌ 已尝试所有API密钥，生成失败")
                    return None
                
                # 否则继续尝试下一个API key
                print(f"⚠️ 尝试使用下一个API密钥...")
                time.sleep(1)
                
        except requests.exceptions.Timeout:
            print(f"⏱️ 请求超时")
            if attempt == max_retries - 1:
                return None
            print(f"⚠️ 尝试使用下一个API密钥...")
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ 请求异常: {str(e)}")
            if attempt == max_retries - 1:
                return None
            print(f"⚠️ 尝试使用下一个API密钥...")
            time.sleep(1)
    
    return None


def generate_image_sora(
    prompt: str,
    model: str = "gpt-image-2",
    size: str = "1:1",
    variants: int = 1,
    reference_urls: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    use_webhook: bool = False,
    shut_progress: bool = False,
    max_retries: int = 3,
    image_size: str = "1K"
) -> Optional[str]:
    """
    使用图像模型生成图片
    
    Args:
        prompt: 图片生成提示词
        model: 模型名称，支持 "gpt-image-2" 或 "gpt-image-1.5"
        size: 图片比例，可选 "auto", "1:1", "3:2", "2:3"
        variants: 批量生成数量，可选 1 或 2
        reference_urls: 参考图片URL列表
        save_path: 图片保存路径，如果为None则自动生成
        use_webhook: 是否使用webhook回调（False则使用流式响应）
        shut_progress: 是否关闭进度回复，直接返回最终结果
        max_retries: 最大重试次数（用于API key轮换）
    
    Returns:
        str: 生成的图片本地保存路径，如果失败则返回None
    """
    # 如果没有提供保存路径，则自动生成
    if not save_path:
        output_dir = get_output_dir()
        import hashlib
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        filename = f"sora_{prompt_hash}.png"
        save_path = os.path.join(output_dir, filename)
    
    # 确保目录存在
    dir_path = os.path.dirname(save_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"🎨 开始使用 {model} 生成图片")
    print(f"📝 提示词: {prompt}")
    print(f"📐 尺寸比例: {size}")
    print(f"🔢 生成数量: {variants}")
    
    # 处理参考图片：将本地路径转换为URL
    processed_urls = None
    if reference_urls:
        if SCDN_AVAILABLE:
            # 自动将本地路径转换为URL
            processed_urls = convert_local_paths_to_urls(reference_urls, use_cache=True)
            if processed_urls:
                print(f"📸 参考图片数量: {len(processed_urls)}")
                for i, url in enumerate(processed_urls):
                    print(f"   参考图片[{i+1}]: {url}")
            else:
                print("⚠️ 没有有效的参考图片URL")
        else:
            # SCDN不可用，只保留已经是URL的路径
            processed_urls = [url for url in reference_urls if url.startswith(('http://', 'https://'))]
            if processed_urls:
                print(f"📸 参考图片数量: {len(processed_urls)}")
                for i, url in enumerate(processed_urls):
                    print(f"   参考图片[{i+1}]: {url}")
            # 警告本地路径无法使用
            local_paths = [url for url in reference_urls if not url.startswith(('http://', 'https://'))]
            for path in local_paths:
                print(f"⚠️ 本地路径无法使用（SCDN不可用）: {path}")
    
    # 构建请求参数
    request_data = _build_draw_request_data(model, prompt, size, variants, shut_progress, image_size)
    
    # 添加参考图片URL
    if processed_urls:
        request_data["urls"] = processed_urls
    
    # 如果使用webhook，添加webhook参数（这里使用-1表示立即返回id）
    if use_webhook:
        request_data["webHook"] = "-1"
    
    # API端点
    api_url = _get_draw_api_url(model)
    
    # 尝试多次请求，支持API key轮换
    for attempt in range(max_retries):
        api_key = get_next_api_key()
        print(f"🔑 使用API密钥: {api_key[:15]}... (尝试 {attempt + 1}/{max_retries})")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        try:
            # 发送请求
            response = requests.post(
                api_url,
                headers=headers,
                json=request_data,
                timeout=120,
                stream=not use_webhook  # 如果不使用webhook，则使用流式响应
            )
            
            print(f"📡 响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                if use_webhook:
                    # webhook模式：立即返回id，需要轮询结果
                    result = response.json()
                    if result.get("code") == 0:
                        task_id = result.get("data", {}).get("id")
                        print(f"✅ 任务已提交，ID: {task_id}")
                        
                        # 轮询获取结果
                        image_url = poll_result(task_id, api_key)
                        if image_url:
                            # 下载图片
                            if download_image(image_url, save_path):
                                print(f"✅ 图片已保存到: {save_path}")
                                return save_path
                    else:
                        print(f"❌ 任务提交失败: {result.get('msg')}")
                else:
                    # 流式响应模式：解析流式数据
                    image_url = parse_stream_response(response)
                    if image_url:
                        # 下载图片
                        if download_image(image_url, save_path):
                            print(f"✅ 图片已保存到: {save_path}")
                            return save_path
                
                # 如果成功处理，跳出重试循环
                break
            else:
                print(f"❌ 请求失败: {response.status_code}")
                print(f"   响应内容: {response.text[:500]}")
                
                # 如果是最后一次尝试，返回None
                if attempt == max_retries - 1:
                    print(f"❌ 已尝试所有API密钥，生成失败")
                    return None
                
                # 否则继续尝试下一个API key
                print(f"⚠️ 尝试使用下一个API密钥...")
                time.sleep(1)
                
        except requests.exceptions.Timeout:
            print(f"⏱️ 请求超时")
            if attempt == max_retries - 1:
                return None
            print(f"⚠️ 尝试使用下一个API密钥...")
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ 请求异常: {str(e)}")
            if attempt == max_retries - 1:
                return None
            print(f"⚠️ 尝试使用下一个API密钥...")
            time.sleep(1)
    
    return None


def parse_stream_response(response: requests.Response, timeout: int = 300) -> Optional[str]:
    """
    解析流式响应，获取生成的图片URL
    
    Args:
        response: requests响应对象（流式）
        timeout: 超时时间（秒），默认300秒
    
    Returns:
        str: 图片URL，如果失败则返回None
    """
    print("📥 正在接收流式响应...")
    
    start_time = time.time()
    last_progress = 0
    
    try:
        for line in response.iter_lines(decode_unicode=True):
            # 检查超时
            if time.time() - start_time > timeout:
                print(f"⏱️ 流式响应超时（{timeout}秒）")
                return None
            
            if line:
                # 跳过空行和注释
                if not line.strip() or line.startswith(':'):
                    continue
                
                # 解析data行
                if line.startswith('data: '):
                    data_str = line[6:]  # 去掉 "data: " 前缀
                    
                    try:
                        data = json.loads(data_str)
                        
                        # 显示进度
                        progress = data.get('progress', 0)
                        status = data.get('status', 'unknown')
                        
                        # 只在进度变化时打印，避免刷屏
                        if progress > last_progress:
                            print(f"⏳ 生成进度: {progress}% - 状态: {status}")
                            last_progress = progress
                        
                        # 检查是否成功
                        if status == 'succeeded':
                            # 优先使用results中的第一个结果
                            results = data.get('results', [])
                            if results and len(results) > 0:
                                image_url = results[0].get('url')
                                if image_url:
                                    print(f"✅ 图片生成成功!")
                                    print(f"🔗 图片URL: {image_url}")
                                    return image_url
                            
                            # 如果results为空，使用旧的url字段
                            image_url = data.get('url')
                            if image_url:
                                print(f"✅ 图片生成成功!")
                                print(f"🔗 图片URL: {image_url}")
                                return image_url
                            
                            # 如果都没有，说明响应格式有问题
                            print(f"⚠️ 状态为成功但未找到图片URL")
                            print(f"   响应数据: {json.dumps(data, ensure_ascii=False)[:200]}")
                            return None
                        
                        # 检查是否失败
                        elif status == 'failed':
                            failure_reason = data.get('failure_reason', 'unknown')
                            error = data.get('error', '')
                            print(f"❌ 生成失败: {failure_reason}")
                            if error:
                                print(f"   错误详情: {error}")
                            return None
                    
                    except json.JSONDecodeError as e:
                        # 记录无法解析的行，但继续处理
                        print(f"⚠️ 无法解析JSON: {data_str[:100]}")
                        continue
        
        print("⚠️ 流式响应结束，但未获取到图片URL")
        return None
        
    except requests.exceptions.ChunkedEncodingError as e:
        print(f"❌ 流式响应中断: {str(e)}")
        return None
        
    except Exception as e:
        print(f"❌ 解析流式响应时出错: {str(e)}")
        import traceback
        print(f"   详细错误: {traceback.format_exc()}")
        return None


def poll_result(task_id: str, api_key: str, max_wait_time: int = 300, poll_interval: int = 5) -> Optional[str]:
    """
    轮询获取任务结果
    
    Args:
        task_id: 任务ID
        api_key: API密钥
        max_wait_time: 最大等待时间（秒）
        poll_interval: 轮询间隔（秒）
    
    Returns:
        str: 图片URL，如果失败则返回None
    """
    api_url = f"{API_HOST}/v1/draw/result"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    start_time = time.time()
    
    print(f"🔄 开始轮询任务结果，任务ID: {task_id}")
    
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json={"id": task_id},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("code") == 0:
                    data = result.get("data", {})
                    status = data.get("status", "unknown")
                    progress = data.get("progress", 0)
                    
                    print(f"⏳ 任务进度: {progress}% - 状态: {status}")
                    
                    if status == "succeeded":
                        # 优先使用results中的第一个结果
                        results = data.get('results', [])
                        if results and len(results) > 0:
                            image_url = results[0].get('url')
                            if image_url:
                                print(f"✅ 任务完成!")
                                return image_url
                        
                        # 如果results为空，使用旧的url字段
                        image_url = data.get("url")
                        if image_url:
                            print(f"✅ 任务完成!")
                            return image_url
                    
                    elif status == "failed":
                        failure_reason = data.get("failure_reason", "unknown")
                        error = data.get("error", "")
                        print(f"❌ 任务失败: {failure_reason}")
                        if error:
                            print(f"   错误详情: {error}")
                        return None
                
                elif result.get("code") == -22:
                    print(f"❌ 任务不存在: {task_id}")
                    return None
            
            # 等待后继续轮询
            time.sleep(poll_interval)
            
        except Exception as e:
            print(f"⚠️ 轮询出错: {str(e)}")
            time.sleep(poll_interval)
    
    print(f"⏱️ 轮询超时（{max_wait_time}秒）")
    return None


def download_image(url: str, save_path: str, max_retries: int = 3) -> bool:
    """
    从URL下载图片并保存到本地
    
    Args:
        url: 图片URL
        save_path: 保存路径
        max_retries: 最大重试次数
    
    Returns:
        bool: 是否成功保存
    """
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"🔄 下载重试 {attempt + 1}/{max_retries}...")
                time.sleep(2)  # 重试前等待2秒
            
            print(f"📥 正在下载图片...")
            response = requests.get(url, timeout=60, stream=True)
            
            if response.status_code == 200:
                # 确保目录存在
                dir_path = os.path.dirname(save_path)
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                
                # 使用流式下载，避免大文件内存问题
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # 验证文件是否成功保存
                if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                    print(f"✅ 图片下载成功")
                    return True
                else:
                    print(f"⚠️ 文件保存失败或文件为空")
                    if attempt < max_retries - 1:
                        continue
                    return False
            else:
                print(f"❌ 下载失败: 状态码 {response.status_code}")
                if attempt < max_retries - 1:
                    continue
                return False
                
        except requests.exceptions.Timeout:
            print(f"⏱️ 下载超时")
            if attempt < max_retries - 1:
                continue
            return False
            
        except requests.exceptions.ConnectionError as e:
            print(f"🔌 连接错误: {str(e)}")
            if attempt < max_retries - 1:
                continue
            return False
            
        except Exception as e:
            print(f"❌ 下载图片时出错: {str(e)}")
            if attempt < max_retries - 1:
                continue
            return False
    
    print(f"❌ 下载失败，已尝试 {max_retries} 次")
    return False


def generate_images_batch(
    prompts: List[Dict[str, Any]],
    model: str = "gpt-image-2",
    reference_urls: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> List[Optional[str]]:
    """
    批量生成图片
    
    Args:
        prompts: 提示词列表，每个元素是包含prompt、scene等信息的字典
        model: 模型名称
        reference_urls: 参考图片URL列表
        output_dir: 输出目录
    
    Returns:
        List[Optional[str]]: 生成的图片路径列表
    """
    if not output_dir:
        output_dir = get_output_dir()
    
    print(f"📁 批量生成图片，保存目录: {output_dir}")
    print(f"📊 总共需要生成 {len(prompts)} 张图片")
    
    if reference_urls:
        print(f"📸 使用 {len(reference_urls)} 张参考图片")
    
    results = []
    
    for idx, item in enumerate(prompts):
        prompt = item.get("prompt", "")
        scene = item.get("scene", f"场景{idx+1}")
        size = item.get("size", "1:1")
        variants = item.get("variants", 1)
        
        filename = f"{scene}.png".replace(" ", "_")
        save_path = os.path.join(output_dir, filename)
        
        print(f"\n{'='*60}")
        print(f"▶ 开始生成第 {idx + 1}/{len(prompts)} 张图: {scene}")
        print(f"{'='*60}")
        
        result_path = generate_image_sora(
            prompt=prompt,
            model=model,
            size=size,
            variants=variants,
            reference_urls=reference_urls,
            save_path=save_path
        )
        
        results.append(result_path)
        
        if result_path:
            print(f"✅ 第 {idx + 1} 张图片生成成功")
        else:
            print(f"❌ 第 {idx + 1} 张图片生成失败")
        
        # 添加延迟，避免请求过快
        if idx < len(prompts) - 1:
            time.sleep(2)
    
    # 统计结果
    success_count = sum(1 for r in results if r is not None)
    print(f"\n{'='*60}")
    print(f"📊 批量生成完成: 成功 {success_count}/{len(prompts)} 张")
    print(f"{'='*60}")
    
    return results


# === 测试函数 ===
def test_sora_image_api():
    """测试sora-image API"""
    print("="*60)
    print("🧪 测试 sora-image API")
    print("="*60)
    
    # 测试1: 基本文本生成图片
    print("\n【测试1】基本文本生成图片")
    test_prompt = "一只可爱的卡通猫咪在草地上玩耍，吉卜力风格"
    result1 = generate_image_sora(
        prompt=test_prompt,
        model="sora-image",
        size="1:1"
    )
    
    if result1:
        print(f"✅ 测试1通过: {result1}")
    else:
        print("❌ 测试1失败")
    
    # 等待一下
    time.sleep(3)
    
    # 测试2: 使用参考图片生成
    print("\n【测试2】使用参考图片生成")
    reference_url = "https://zczczc17508.oss-cn-hangzhou.aliyuncs.com/%E4%B8%BB%E8%A7%92_%E5%90%89%E5%8D%9C%E5%8A%9B.png"
    test_prompt2 = "动漫形象抱着一个小狗，温馨的场景"
    result2 = generate_image_sora(
        prompt=test_prompt2,
        model="sora-image",
        size="1:1",
        reference_urls=[reference_url]
    )
    
    if result2:
        print(f"✅ 测试2通过: {result2}")
    else:
        print("❌ 测试2失败")


def test_batch_generation():
    """测试批量生成"""
    print("\n" + "="*60)
    print("🧪 测试批量生成")
    print("="*60)
    
    test_data = [
        {
            "scene": "小狗玩耍",
            "prompt": "一只可爱的小狗在草地上玩耍，阳光明媚",
            "size": "1:1"
        },
        {
            "scene": "猫咪休息",
            "prompt": "一只优雅的猫咪坐在窗台上休息，温暖的阳光",
            "size": "3:2"
        },
        {
            "scene": "风景画",
            "prompt": "美丽的山水风景，有山有水有树木，水墨画风格",
            "size": "2:3"
        }
    ]
    
    results = generate_images_batch(test_data, model="sora-image")
    
    print(f"\n批量生成结果:")
    for i, result in enumerate(results):
        if result:
            print(f"  {i+1}. ✅ {result}")
        else:
            print(f"  {i+1}. ❌ 生成失败")


if __name__ == "__main__":
    # 运行测试
    print("🚀 开始测试 sora-image 图像生成器")
    print("="*60)
    
    # 测试基本功能
    test_sora_image_api()
    
    # 测试批量生成（可选，取消注释以运行）
    # test_batch_generation()
    
    print("\n" + "="*60)
    print("✅ 测试完成")
    print("="*60)
