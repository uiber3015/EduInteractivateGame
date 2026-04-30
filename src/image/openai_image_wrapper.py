"""
OpenAI图像生成器封装模块
提供简单易用的接口来生成故事节点的图片
支持多种图像生成模型：
- OpenAI gpt-image-1 (原有接口)
- gpt-image-2 (默认接口)
- gpt-image-1.5 (新增接口)
- yunwu-image (新增接口)

注意：此为非并行版本，适合单线程顺序生成
"""

import os
import json
import requests
from openai_image_generation import generate_image_with_http_client, save_image_from_url
from sora_image_generator import generate_image_sora
from utils.model_provider_config import get_image_provider, get_image_api_keys
from image.yunwu_image_generator import YunwuImageGenerator

# 导入SCDN图床上传工具
try:
    from scdn_image_uploader import upload_local_image
    SCDN_AVAILABLE = True
except ImportError:
    SCDN_AVAILABLE = False


def download_image_from_url(image_url: str, output_path: str) -> bool:
    """
    从URL下载图像到本地
    
    Args:
        image_url: 图像URL
        output_path: 输出文件路径
        
    Returns:
        是否下载成功
    """
    try:
        print(f"  📥 下载图像: {image_url}")
        response = requests.get(image_url, timeout=30)
        
        if response.status_code == 200:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"  ✓ 已保存: {output_path}")
            return True
        else:
            print(f"  ✗ 下载失败 (HTTP {response.status_code})")
            return False
            
    except Exception as e:
        print(f"  ✗ 下载异常: {e}")
        return False

def generate_story_node_image(prompt, output_path, reference_image_paths=None, style="动漫风格", model="gpt-image-2", return_url=False, image_provider=None):
    """
    为故事节点生成图片
    
    Args:
        prompt: 图片生成提示词
        output_path: 输出图片路径
        reference_image_paths: 参考图片路径列表（可选），支持多张图片
        style: 图片风格，默认为"动漫风格"
        model: 使用的模型，可选 "gpt-image-1", "gpt-image-2", "gpt-image-1.5", "nano-banana-pro"
        return_url: 是否返回图像URL而不是下载（用于并行处理）
        image_provider: 图像提供商，可选 "yunwu"，默认为 None
    
    Returns:
        如果return_url=True，返回图像URL；否则返回bool表示是否成功生成图片
    """
    # 添加风格前缀
    no_text_rule = "【硬性要求】画面中绝对不要出现任何可读文字、中文、英文、数字、logo、水印、标签、海报文字、屏幕文字、路牌文字、书本文字或UI文案；若场景包含牌子、屏幕、书本、展板，只保留其外形，不显示任何可识别内容。"
    styled_prompt = f"{style}:{prompt}\n\n{no_text_rule}"
    provider = get_image_provider(image_provider)
    
    try:
        if provider == "yunwu":
            if return_url:
                print("⚠️ yunwu 图像通道当前仅支持直接保存本地文件，不支持 return_url=True")
                return None

            api_keys = get_image_api_keys("yunwu")
            if not api_keys:
                print("⚠️ yunwu 图像提供商的API key未配置，回退到 legacy")
                provider = "legacy"
            else:
                generator = YunwuImageGenerator(api_keys=api_keys)
                result_path = generator.generate_image(
                    prompt=styled_prompt,
                    output_path=output_path,
                    reference_image_paths=reference_image_paths,
                    aspect_ratio="16:9"
                )
                print(f"✅ yunwu 图片生成成功: {result_path}")
                return True

        # 根据模型选择不同的生成方式
        if model in ["gpt-image-2", "gpt-image-1.5", "nano-banana-pro"]:
            # 使用新的图像接口
            print(f"正在使用 {model} 模型生成图片: {styled_prompt}")
            
            # 将本地路径转换为URL（如果需要）
            reference_urls = None
            if reference_image_paths and len(reference_image_paths) > 0:
                print(f"使用参考图片数量: {len(reference_image_paths)}")
                reference_urls = []
                for i, img_path in enumerate(reference_image_paths):
                    if img_path.startswith('http'):
                        reference_urls.append(img_path)
                        print(f"参考图片[{i+1}] (URL): {img_path}")
                    elif os.path.exists(img_path):
                        # 本地路径，使用SCDN上传
                        if SCDN_AVAILABLE:
                            print(f" 上传本地参考图片[{i+1}]: {img_path}")
                            url = upload_local_image(img_path, use_cache=True)
                            if url:
                                reference_urls.append(url)
                                print(f" 参考图片[{i+1}] 上传成功: {url}")
                            else:
                                print(f" 参考图片[{i+1}] 上传失败，跳过")
                        else:
                            print(f" 参考图片[{i+1}] 是本地路径，SCDN不可用: {img_path}")
                    else:
                        print(f" 参考图片[{i+1}] 路径不存在: {img_path}")
            
            # 如果return_url=True，直接返回URL
            if return_url:
                from sora_image_generator import generate_image_url_sora
                image_url = generate_image_url_sora(
                    prompt=styled_prompt,
                    model=model,
                    size="16:9",
                    variants=1,
                    reference_urls=reference_urls if reference_urls else None
                )
                return image_url
            else:
                result_path = generate_image_sora(
                    prompt=styled_prompt,
                    model=model,
                    size="16:9",
                    variants=1,
                    reference_urls=reference_urls if reference_urls else None,
                    save_path=output_path
                )
                
                if result_path:
                    print(f"✅ 图片生成成功: {result_path}")
                    return True
                else:
                    print(f"❌ 图片生成失败")
                    return False
        
        else:
            # 使用原有的gpt-image-1接口
            print(f"正在使用 {model} 模型生成图片: {styled_prompt}")
            if reference_image_paths and len(reference_image_paths) > 0:
                print(f"使用参考图片数量: {len(reference_image_paths)}")
                for i, img_path in enumerate(reference_image_paths):
                    print(f"参考图片[{i+1}]: {img_path}")
                image_result = generate_image_with_http_client(
                    prompt=styled_prompt,
                    image_paths=reference_image_paths
                )
            else:
                image_result = generate_image_with_http_client(
                    prompt=styled_prompt
                )
            
            if image_result:
                if return_url:
                    return image_result
                else:
                    # 保存图片
                    success = save_image_from_url(image_result, output_path)
                    if success:
                        print(f"✅ 图片生成成功: {output_path}")
                        return True
                    else:
                        print(f"❌ 图片保存失败: {output_path}")
                        return False
            else:
                print(f"❌ 图片生成失败: {styled_prompt}")
                return False if not return_url else None
            
    except Exception as e:
        print(f"❌ 生成图片时出错: {str(e)}")
        return False if not return_url else None

def generate_images_for_story_nodes(story_nodes, output_dir, reference_image_paths=None, use_scene_transition=False, model="gpt-image-2", max_retries=3, download_images=True, edges=None, image_provider=None):
    """
    为多个故事节点批量生成图片
    
    Args:
        story_nodes: 故事节点列表，每个节点包含id和image_prompt
        output_dir: 输出目录
        reference_image_paths: 参考图片路径列表（可选），支持多张图片（主角参考图）
        use_scene_transition: 是否使用场景转换逻辑
        model: 使用的模型，可选 "gpt-image-1", "gpt-image-2", "gpt-image-1.5", "nano-banana-pro"
        max_retries: 每个节点的最大重试次数，默认3次
        download_images: 是否立即下载图片，False则只获取URL
        edges: 故事图的边关系，格式为 {parent_id: [child_id1, child_id2, ...]}
        image_provider: 图像提供商，可选 "yunwu"，默认为 None
    
    Returns:
        dict: 包含生成结果的字典，包括image_mapping（节点ID到图片路径的映射）和image_urls（节点ID到URL的映射）
    
    场景一致性策略：
        1. node_1（起始节点）：使用主角参考图生成，作为整体风格基准
        2. 非场景转换节点（scene_transition=0）：参考父节点图片 + 主角参考图，保持场景连贯
        3. 场景转换节点（scene_transition=1）：参考node_1图片 + 主角参考图，保持角色和风格一致
    """
    generated_images = {}
    image_urls = {}
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计成功和失败的节点
    success_count = 0
    fail_count = 0
    failed_nodes = []  # 记录失败的节点
    
    # 创建节点ID到节点对象的映射
    node_map = {node["id"]: node for node in story_nodes}
    
    # 使用edges数据构建正确的父子关系映射
    parent_map = {}
    if use_scene_transition and edges:
        # 从edges构建子节点到父节点的反向映射
        for parent_id, child_ids in edges.items():
            for child_id in child_ids:
                if child_id not in parent_map:
                    parent_map[child_id] = []
                parent_map[child_id].append(parent_id)
        print(f"📊 使用edges数据构建父子关系映射，共 {len(parent_map)} 个节点有父节点")
    elif use_scene_transition:
        print("⚠️ 未提供edges数据，无法建立正确的父子关系映射")
    
    for node in story_nodes:
        node_id = node["id"]
        
        # 如果当前节点已经有image_path，跳过该节点
        if "metadata" in node and node["metadata"].get("image_path"):
            print(f"⏭️ 节点 {node_id} 已有图片，跳过生成")
            continue
        
        # 获取图片提示词
        if "metadata" in node and "image_prompt" in node["metadata"]:
            image_prompt = node["metadata"]["image_prompt"]
        else:
            image_prompt = node.get("content", "")
        
        # 构建输出路径（使用 {node_id}.png 格式）
        output_path = os.path.join(output_dir, f"{node_id}.png")
        
        # 确定使用的参考图片 - 实现场景一致性策略
        current_reference_images = reference_image_paths.copy() if reference_image_paths else []
        
        # 判断是否是起始节点（通常是node_1）
        is_start_node = (node_id == "node_1" or node_id not in parent_map)
        
        if use_scene_transition and "metadata" in node:
            scene_transition = node["metadata"].get("scene_transition", 1)  # 默认为场景转换
            
            if is_start_node:
                # 策略1: 起始节点 - 只使用主角参考图，作为整体风格基准
                print(f"🎬 节点 {node_id} 是起始节点，使用主角参考图作为风格基准")
                # current_reference_images 已经包含主角参考图，不需要额外处理
                
            elif scene_transition == 0:
                # 策略2: 非场景转换 - 参考父节点图片 + 主角参考图，保持场景连贯
                if node_id in parent_map:
                    parent_ids = parent_map[node_id]
                    for parent_id in parent_ids:
                        if parent_id in generated_images:
                            # 将父节点图片作为参考，保持场景连贯性
                            parent_image_path = generated_images[parent_id]
                            if parent_image_path not in current_reference_images:
                                current_reference_images.append(parent_image_path)
                                print(f"📸 节点 {node_id} (同场景) 参考父节点 {parent_id} 的图片: {parent_image_path}")
                
            else:
                # 策略3: 场景转换 - 参考node_1图片 + 主角参考图，保持角色和风格一致
                # 场景转换时，需要保持角色形象和整体风格的一致性
                if "node_1" in generated_images:
                    node1_image_path = generated_images["node_1"]
                    if node1_image_path not in current_reference_images:
                        current_reference_images.append(node1_image_path)
                        print(f"🎨 节点 {node_id} (新场景) 参考node_1的图片保持角色和风格一致: {node1_image_path}")
                else:
                    print(f"⚠️ 节点 {node_id} 是场景转换节点，但node_1图片尚未生成")
        
        # 打印当前使用的参考图片信息
        if current_reference_images:
            print(f"📷 节点 {node_id} 使用 {len(current_reference_images)} 张参考图片")
        
        # 重试机制：尝试最多max_retries次
        success = False
        image_url = None
        
        for retry in range(max_retries):
            if retry > 0:
                print(f"🔄 节点 {node_id} 第 {retry + 1}/{max_retries} 次重试...")
                import time
                time.sleep(2)  # 重试前等待2秒
            
            # 生成图片（如果download_images=False，只获取URL）
            if download_images:
                success = generate_story_node_image(
                    prompt=image_prompt,
                    output_path=output_path,
                    reference_image_paths=current_reference_images,
                    model=model,
                    return_url=False,
                    image_provider=image_provider
                )
                
                if success:
                    generated_images[node_id] = output_path
                    success_count += 1
                    if retry > 0:
                        print(f"✅ 节点 {node_id} 在第 {retry + 1} 次尝试后成功生成")
                    break
            else:
                # 只获取URL，不下载
                image_url = generate_story_node_image(
                    prompt=image_prompt,
                    output_path=output_path,
                    reference_image_paths=current_reference_images,
                    model=model,
                    return_url=True,
                    image_provider=image_provider
                )
                
                if image_url:
                    image_urls[node_id] = image_url
                    success_count += 1
                    if retry > 0:
                        print(f"✅ 节点 {node_id} 在第 {retry + 1} 次尝试后成功获取URL")
                    break
        
        # 如果所有重试都失败
        if not success and not image_url:
            fail_count += 1
            failed_nodes.append(node_id)
            print(f"❌ 节点 {node_id} 在 {max_retries} 次尝试后仍然失败")
    
    print(f"\n📊 图片生成统计:")
    print(f"✅ 成功: {success_count} 个节点")
    print(f"❌ 失败: {fail_count} 个节点")
    
    if failed_nodes:
        print(f"\n⚠️ 失败的节点列表:")
        for node_id in failed_nodes:
            print(f"   - {node_id}")
    
    return {
        "image_mapping": generated_images,
        "image_urls": image_urls,
        "success_count": success_count,
        "fail_count": fail_count,
        "failed_nodes": failed_nodes
    }

def update_story_graph_with_images(story_graph_path, output_path, image_mapping, image_urls=None):
    """
    更新故事图JSON文件，添加图片路径和URL信息
    
    Args:
        story_graph_path: 原始故事图JSON文件路径
        output_path: 输出JSON文件路径
        image_mapping: 节点ID到图片路径的映射
        image_urls: 节点ID到图片URL的映射（可选）
    
    Returns:
        bool: 是否成功更新
    """
    try:
        # 读取故事图
        with open(story_graph_path, 'r', encoding='utf-8') as f:
            story_graph = json.load(f)
        
        # 更新每个节点的metadata
        updated_count = 0
        for node in story_graph["nodes"]:
            node_id = node["id"]
            
            # 确保metadata字段存在
            if "metadata" not in node:
                node["metadata"] = {}
            
            # 添加图片路径
            if node_id in image_mapping:
                node["metadata"]["image_path"] = image_mapping[node_id]
                updated_count += 1
                print(f"✅ 更新节点 {node_id} 图片路径: {image_mapping[node_id]}")
            
            # 添加图片URL
            if image_urls and node_id in image_urls:
                node["metadata"]["image_url"] = image_urls[node_id]
                print(f"✅ 更新节点 {node_id} 图片URL: {image_urls[node_id]}")
        
        # 保存更新后的故事图
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(story_graph, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 故事图已更新，共 {updated_count} 个节点添加了图片信息")
        print(f"📁 保存位置: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 更新故事图失败: {str(e)}")
        return False

# 主要功能函数
def generate_story_images_pipeline(story_graph_path, output_dir, reference_image_paths=None, update_graph=True, use_scene_transition=False, model="gpt-image-2", download_images=True, image_provider=None):
    """
    完整的故事图片生成管道（非并行版本）
    
    Args:
        story_graph_path: 故事图JSON文件路径
        output_dir: 图片输出目录
        reference_image_paths: 参考图片路径列表（可选）- 可以是本地路径或URL，支持多张图片
        update_graph: 是否更新原始JSON文件
        use_scene_transition: 是否使用场景转换逻辑（仅在openai模式下使用）
        model: 使用的模型，可选 "gpt-image-1", "gpt-image-2" (默认), "gpt-image-1.5"
        download_images: 是否立即下载图片到本地（True）或只获取URL（False）
        image_provider: 图像提供商，可选 "yunwu"，默认为 None
    
    Returns:
        dict: 包含结果信息的字典
    """
    result = {
        "success": False,
        "generated_images": {},
        "image_urls": {},
        "output_dir": output_dir,
        "updated_graph_path": None,
        "reference_images_used": []
    }
    
    try:
        # 检查参考图片
        if reference_image_paths:
            # 确保reference_image_paths是列表
            if not isinstance(reference_image_paths, list):
                reference_image_paths = [reference_image_paths]
                
            valid_images = []
            for img_path in reference_image_paths:
                if os.path.exists(img_path):
                    print(f"📸 使用本地参考图片: {img_path}")
                    valid_images.append(img_path)
                elif img_path.startswith('http'):
                    print(f"🌐 使用网络参考图片: {img_path}")
                    valid_images.append(img_path)
                else:
                    print(f"⚠️ 参考图片不存在: {img_path}")
            
            if valid_images:
                result["reference_images_used"] = valid_images
                reference_image_paths = valid_images
            else:
                print("⚠️ 没有有效的参考图片")
                reference_image_paths = None
        
        # 读取故事图
        with open(story_graph_path, 'r', encoding='utf-8') as f:
            story_graph = json.load(f)
        
        print(f"📖 加载故事图: {story_graph_path}")
        print(f"📊 共 {len(story_graph['nodes'])} 个节点")
        
        # 获取edges数据用于构建正确的父子关系
        edges = story_graph.get("edges", {})
        if edges:
            print(f"🔗 加载edges数据，共 {len(edges)} 个父节点有子节点")
        else:
            print("⚠️ 故事图中没有edges数据，无法建立精确的父子关系")
        
        # 生成图片
        print(f"\n🎨 开始使用 {model} 模型生成图片...")
        print(f"🎯 场景一致性策略: use_scene_transition={use_scene_transition}")
        generation_result = generate_images_for_story_nodes(
            story_nodes=story_graph["nodes"],
            output_dir=output_dir,
            reference_image_paths=reference_image_paths,
            use_scene_transition=use_scene_transition,
            model=model,
            download_images=download_images,
            edges=edges,  # 传递edges数据用于构建正确的父子关系
            image_provider=image_provider
        )
        
        result["generated_images"] = generation_result["image_mapping"]
        result["image_urls"] = generation_result["image_urls"]
        result["success_count"] = generation_result["success_count"]
        result["fail_count"] = generation_result["fail_count"]
        result["failed_nodes"] = generation_result["failed_nodes"]
        
        # 更新故事图（如果要求）
        if update_graph and (generation_result["image_mapping"] or generation_result["image_urls"]):
            print(f"\n📝 更新故事图...")
            updated_path = os.path.join(output_dir, "story_graph_with_images.json")
            success = update_story_graph_with_images(
                story_graph_path=story_graph_path,
                output_path=updated_path,
                image_mapping=generation_result["image_mapping"],
                image_urls=generation_result["image_urls"]
            )
            
            if success:
                result["updated_graph_path"] = updated_path
        
        result["success"] = generation_result["success_count"] > 0
        return result
        
    except Exception as e:
        print(f"❌ 管道执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
        return result

# 测试用例
if __name__ == "__main__":
    """
    测试多张参考图片功能
    """
    print("=" * 50)
    print("测试多张参考图片功能")
    print("=" * 50)
    
    # 查找项目中的参考图片
    possible_refs = [
        "data/主角_吉卜力.png",
        "output\\物理test3\\output_images\\node_1.png"
    ]
    
    valid_refs = [ref for ref in possible_refs if os.path.exists(ref)]
    
    # 创建输出目录
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试提示词
    test_prompt_with_ref = "一个勇敢的骑士，手持剑，阳光明媚，人物形象参考第一张，背景参考第二张"
    test_prompt_no_ref = "一个勇敢的骑士站在宏伟的城堡前，手持剑，阳光明媚，吉卜力风格"
    
    if len(valid_refs) >= 2:
        print(f"使用多张参考图片: {valid_refs}")
        
        # 调用生成函数
        output_path = os.path.join(output_dir, "knight_with_multiple_refs.png")
        success = generate_story_node_image(
            prompt=test_prompt_with_ref,
            output_path=output_path,
            reference_image_paths=valid_refs,
            style="吉卜力风格"
        )
        
        print(f"\n生成结果:")
        print(f"- 成功: {success}")
        print(f"- 输出路径: {output_path}")
        print(f"- 使用的参考图片数量: {len(valid_refs)}")
        
        # 测试单张参考图片
        print(f"\n测试单张参考图片: {valid_refs[0]}")
        output_path_single = os.path.join(output_dir, "knight_with_single_ref.png")
        success_single = generate_story_node_image(
            prompt=test_prompt_with_ref,
            output_path=output_path_single,
            reference_image_paths=[valid_refs[0]],
            style="吉卜力风格"
        )
        
        print(f"\n生成结果:")
        print(f"- 成功: {success_single}")
        print(f"- 输出路径: {output_path_single}")
        print(f"- 使用的参考图片数量: 1")
        
    else:
        print("未找到足够多的参考图片进行测试")
        print(f"找到的参考图片: {valid_refs}")
        
        # 即使只有一张参考图片，也进行测试
        if valid_refs:
            print(f"\n使用单张参考图片进行测试: {valid_refs[0]}")
            
            output_path = os.path.join(output_dir, "knight_with_single_ref.png")
            success = generate_story_node_image(
                prompt=test_prompt_with_ref,
                output_path=output_path,
                reference_image_paths=valid_refs,
                style="吉卜力风格"
            )
            
            print(f"\n生成结果:")
            print(f"- 成功: {success}")
            print(f"- 输出路径: {output_path}")
            print(f"- 使用的参考图片: {valid_refs[0]}")
    
    # # 测试不使用参考图片
    # print(f"\n测试不使用参考图片:")
    # output_path_no_ref = os.path.join(output_dir, "knight_no_ref.png")
    # success_no_ref = generate_story_node_image(
    #     prompt=test_prompt_no_ref,
    #     output_path=output_path_no_ref,
    #     reference_image_paths=None,
    #     style="吉卜力风格"
    # )
    
    # print(f"\n生成结果:")
    # print(f"- 成功: {success_no_ref}")
    # print(f"- 输出路径: {output_path_no_ref}")
    
    # print("\n" + "=" * 50)
    # print("测试完成")
    # print("=" * 50)