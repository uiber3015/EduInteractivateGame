"""
COT完整流程Pipeline

基于RAG+COT的交互式物理教育故事生成完整流程：
1. 使用cot_web_story_generator_v2生成交互式题目
2. 转换为StoryGraph格式
3. 生成图片提示词
4. 生成故事图片
5. 启动GUI展示

此流程独立于原有的full_pipeline.py，不影响原有代码结构。
"""

import os
import sys
import json
import time
import re
from datetime import datetime
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from generation.cot_web_story_generator_v2 import GeminiCoTGenerator
from generation.cot_to_storygraph_converter import CoTToStoryGraphConverter
from image.intelligent_image_reference_selector import IntelligentImageReferenceSelector
from image_consistency import (
    apply_image_consistency_enrichment,
    load_character_reference_urls,
    save_image_consistency_artifacts,
)
from utils.model_provider_config import get_text_model_config, get_image_api_keys

NO_TEXT_IMAGE_RULE = "画面中绝对不要出现任何可读文字、中文、英文、数字、logo、水印、标签、海报文字、屏幕文字、路牌文字、书本文字或UI文案；若场景包含牌子、屏幕、书本、展板，只保留其外形，不显示任何可识别内容。"
CHARACTER_PATTERNS = [
    r"爷爷", r"奶奶", r"外公", r"外婆", r"爸爸", r"妈妈", r"叔叔", r"阿姨", r"老师", r"校长",
    r"同学", r"班主任", r"主角", r"男孩", r"女孩", r"小[A-Za-z\u4e00-\u9fa5]"
]

def load_json_if_exists(file_path: str):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_json(file_path: str, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(_sanitize_story_graph_text(data), f, ensure_ascii=False, indent=2)


def _sanitize_story_graph_text(value):
    if isinstance(value, str):
        cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", value, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"!\[[^\]]*image[^\]]*\]\([^\)]*\)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[\s*image[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<image[^>]*>", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        fenced_match = re.search(r"```(?:text|markdown|md)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fenced_match:
            cleaned = fenced_match.group(1).strip()
        return cleaned.strip()

    if isinstance(value, list):
        return [_sanitize_story_graph_text(item) for item in value]

    if isinstance(value, dict):
        return {key: _sanitize_story_graph_text(item) for key, item in value.items()}

    return value


def extract_explicit_characters(text: str):
    if not text:
        return []

    found = []
    for pattern in CHARACTER_PATTERNS:
        for match in re.findall(pattern, text):
            role = match.strip()
            if role and role not in found:
                found.append(role)
    return found


def annotate_recurring_characters(story_graph: dict, min_occurrences: int = 2) -> dict:
    character_counts = {}

    for node in story_graph.get("nodes", []):
        content = node.get("content", "")
        roles = extract_explicit_characters(content)
        for role in roles:
            character_counts[role] = character_counts.get(role, 0) + 1

    recurring_characters = {
        role: count for role, count in character_counts.items()
        if count >= min_occurrences and role not in {"同学", "男孩", "女孩", "主角"}
    }

    for node in story_graph.get("nodes", []):
        content = node.get("content", "")
        roles = extract_explicit_characters(content)
        recurring_for_node = [role for role in roles if role in recurring_characters]
        if recurring_for_node:
            if "metadata" not in node or not isinstance(node.get("metadata"), dict):
                node["metadata"] = {}
            node["metadata"]["recurring_characters"] = recurring_for_node

    return recurring_characters


def build_character_reference_prompt(character_name: str) -> str:
    if character_name == "爷爷":
        role_desc = "一位中国老年爷爷，灰白短发，和蔼沉稳，日常朴素穿着"
    elif character_name == "奶奶":
        role_desc = "一位中国老年奶奶，温和慈祥，短卷发，日常朴素穿着"
    elif character_name in {"老师", "班主任", "校长"}:
        role_desc = f"一位中国校园里的{character_name}，气质亲切干练，穿着简洁得体"
    elif character_name in {"爸爸", "妈妈", "叔叔", "阿姨"}:
        role_desc = f"一位中国家庭中的{character_name}，形象自然稳定，穿着日常"
    else:
        role_desc = f"中国校园科普故事中的固定角色“{character_name}”，形象清晰稳定，穿着日常"

    return f"吉卜力风格，单人角色设定图，{role_desc}，半身像，中性干净背景，正面或微侧面，便于后续多场景保持角色一致。{NO_TEXT_IMAGE_RULE}"


def generate_recurring_character_reference_images(recurring_characters: dict, output_dir: str, generator, model: str) -> dict:
    recurring_character_refs = {}
    if not recurring_characters:
        return recurring_character_refs

    refs_dir = os.path.join(output_dir, "output", "character_references")
    os.makedirs(refs_dir, exist_ok=True)

    for character_name in recurring_characters.keys():
        prompt = build_character_reference_prompt(character_name)
        safe_name = re.sub(r"[^\w\u4e00-\u9fa5-]+", "_", character_name)
        output_path = os.path.join(refs_dir, f"{safe_name}.png")
        try:
            image_url = generator.generate_image(prompt=prompt, size="1:1", reference_image_url=None, model=model)
            if image_url:
                recurring_character_refs[character_name] = image_url
        except Exception as e:
            print(f"⚠️ 角色参考图生成失败 {character_name}: {e}")

    if recurring_character_refs:
        save_json(os.path.join(refs_dir, "character_reference_urls.json"), recurring_character_refs)

    return recurring_character_refs


def update_resume_state(output_dir: str, **kwargs):
    state_file = os.path.join(output_dir, "output", "resume_state.json")
    state = load_json_if_exists(state_file) or {}
    state.update(kwargs)
    state["updated_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_json(state_file, state)


def persist_image_generation_progress(output_dir: str, story_graph_with_prompts: dict, story_graph_with_transitions_path: str, output_images_dir: str):
    output_file = os.path.join(output_images_dir, "story_graph_with_images.json")
    save_json(output_file, story_graph_with_prompts)

    input_basename = os.path.splitext(os.path.basename(story_graph_with_transitions_path))[0]
    final_output_file = os.path.join(output_dir, "output", f"{input_basename}_with_images.json")
    save_json(final_output_file, story_graph_with_prompts)
    return output_file, final_output_file


def remove_knowledge_prefix(text: str) -> str:
    """
    移除知识点文本开头的"知识点名称："格式
    
    例如：
    - "自然导航原理：受地球公转影响..." -> "受地球公转影响..."
    - "水的净化：通过过滤操作..." -> "通过过滤操作..."
    - "没有冒号的内容" -> "没有冒号的内容"
    
    Args:
        text: 原始文本
        
    Returns:
        移除前缀后的文本
    """
    if not text:
        return text
    
    # 检测中文冒号
    if '：' in text:
        # 找到第一个冒号的位置
        colon_pos = text.index('：')
        # 检查冒号前面是否是简短的标题（通常少于20个字符）
        prefix = text[:colon_pos]
        if len(prefix) < 20:
            # 移除前缀和冒号，返回剩余内容
            return text[colon_pos + 1:].strip()
    
    # 检测英文冒号
    if ':' in text:
        colon_pos = text.index(':')
        prefix = text[:colon_pos]
        if len(prefix) < 20:
            return text[colon_pos + 1:].strip()
    
    # 没有冒号或前缀过长，返回原文
    return text


def adapt_fatal_knowledge(story_graph: dict, text_provider: str = None) -> dict:
    """
    使用LLM为fatal节点生成适配的知识点说明。
    
    正确答案的知识点是从"选对了"的视角写的，直接复用到错误节点会产生语义冲突。
    此函数根据错误原因和正确知识点，生成一段中性的、适合错误场景的知识点说明。
    """
    text_config = get_text_model_config(text_provider)
    client = OpenAI(
        api_key=text_config["api_key"],
        base_url=text_config["base_url"]
    )
    
    fatal_nodes = [n for n in story_graph.get("nodes", []) if n.get("type") == "fatal"]
    if not fatal_nodes:
        return story_graph
    
    print(f"\n🔧 为 {len(fatal_nodes)} 个失败节点生成适配知识点...")
    
    for node in fatal_nodes:
        meta = node.get("metadata", {})
        knowledge_point = meta.get("knowledge_point", "")
        wrong_explanation = meta.get("wrong_explanation", "")
        correct_knowledge_ref = meta.get("correct_knowledge_ref", "")
        
        if not knowledge_point or not correct_knowledge_ref:
            continue
        
        prompt = f"""你是一位中学教师，精通物理、化学、生物、地理等学科知识。学生在一道关于"{knowledge_point}"的题目中选择了错误答案。

**学生的错误原因**：
{wrong_explanation}

**正确答案对应的知识点解析（仅供参考，不能直接复用）**：
{correct_knowledge_ref}

请你为这位选错的学生写一段知识点说明，要求：
1. 客观解释"{knowledge_point}"的核心概念、定义和公式
2. 不要使用"同学"、"你"等称呼，使用客观、中性的语言
3. 不要说"你正确使用了..."或"本题利用的是..."这类暗示学生选对了的话
4. 可以点明错误与正确做法的区别，但不要说教
5. 100-150字，适合中学生阅读
6. 直接输出知识点内容，不要加"知识点："前缀
7. 使用专业、客观的学术语言风格"""

        try:
            response = client.chat.completions.create(
                model=text_config["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000,
                temperature=0.3
            )
            finish_reason = response.choices[0].finish_reason
            adapted_knowledge = response.choices[0].message.content.strip()
            
            if finish_reason == "length":
                print(f"  ⚠️ {node['id']}: LLM输出被截断(finish_reason=length)，保留原始知识点")
            elif adapted_knowledge:
                # 移除知识点前缀（如果LLM仍然生成了"XXX："格式），但保留"知识点："标签
                adapted_knowledge = remove_knowledge_prefix(adapted_knowledge)
                
                # 重新构建 physics_knowledge
                wrong_physics = f"{knowledge_point}\n\n"
                wrong_physics += f"错误原因：{wrong_explanation}\n\n"
                wrong_physics += f"知识点：{adapted_knowledge}"
                meta["physics_knowledge"] = wrong_physics
                print(f"  ✅ {node['id']}: 知识点已适配")
            else:
                print(f"  ⚠️ {node['id']}: LLM返回为空，保留原始知识点")
        except Exception as e:
            print(f"  ⚠️ {node['id']}: LLM调用失败({e})，保留原始知识点")
    
    return story_graph


def load_all_knowledge_points(knowledge_file_path: str = None) -> list:
    """
    加载所有知识点池
    
    Args:
        knowledge_file_path: 知识点文件路径，默认为 data/rag_data/aggregated/00_all_knowledge_points.json
        
    Returns:
        知识点列表
    """
    if knowledge_file_path is None:
        # 默认使用项目根目录下的知识点文件
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        knowledge_file_path = os.path.join(project_root, "data", "rag_data", "aggregated", "00_all_knowledge_points.json")
    
    if not os.path.exists(knowledge_file_path):
        print(f"⚠️ 知识点文件不存在: {knowledge_file_path}")
        return []
    
    try:
        with open(knowledge_file_path, 'r', encoding='utf-8') as f:
            knowledge_pool = json.load(f)
        print(f"✅ 成功加载知识点池: {len(knowledge_pool)} 个知识点")
        return knowledge_pool
    except Exception as e:
        print(f"❌ 加载知识点池失败: {e}")
        return []


def cot_pipeline(
    knowledge_points: list = None,
    scenario: str = None,
    num_questions: int = 8,
    output_dir: str = None,
    generate_images: bool = True,
    generator_type: str = "openai",
    reference_image_path: str = None,
    model: str = "gpt-image-2",
    text_provider: str = None,
    image_provider: str = None,
    auto_select_knowledge: bool = False,
    enable_resume: bool = True
):
    """
    COT完整流程 - 分步生成策略
    
    Args:
        knowledge_points: 知识点列表（当auto_select_knowledge=False时使用）
        scenario: 故事场景描述
        num_questions: 题目数量，默认4个
        output_dir: 输出目录，默认为output/cot_{timestamp}
        generate_images: 是否生成图片
        generator_type: 图片生成器类型 (openai/wanx/tencent)
        reference_image_path: 参考图片路径
        model: 图像生成模型 ("gpt-image-1", "gpt-image-2", "gpt-image-1.5")，仅当generator_type="openai"时有效
        text_provider: 文本模型提供商
        image_provider: 图像模型提供商
        auto_select_knowledge: 是否自动从知识点池中选择知识点（默认False）
        
    Returns:
        包含所有生成结果的字典
    """
    
    # 记录开始时间
    start_time = datetime.now()
    print("\n" + "=" * 70)
    print("COT交互式物理教育故事生成流程")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output/cot_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "output"), exist_ok=True)

    output_subdir = os.path.join(output_dir, "output")
    selected_kp_file = os.path.join(output_subdir, "selected_knowledge_points.json")
    cot_output_file = os.path.join(output_subdir, "cot_raw_result.json")
    story_graph_file = os.path.join(output_subdir, "enhanced_story_graph.json")
    story_graph_levels_file = os.path.join(output_subdir, "story_graph_from_levels.json")
    image_prompts_file = os.path.join(output_subdir, "story_graph_with_image_prompts.json")
    transitions_file = os.path.join(output_subdir, "story_graph_with_image_prompts_with_scene_transitions.json")

    update_resume_state(
        output_dir,
        status="running",
        scenario=scenario,
        num_questions=num_questions,
        text_provider=text_provider,
        image_provider=image_provider,
        generate_images=generate_images,
        auto_select_knowledge=auto_select_knowledge,
    )
    
    # 读取续跑状态
    resume_state = None
    if enable_resume:
        resume_state_file = os.path.join(output_dir, "output", "resume_state.json")
        resume_state = load_json_if_exists(resume_state_file)
    
    # 如果续跑时修改了 generate_images 配置，需要重新生成图片相关文件
    force_regenerate_images = False
    if enable_resume and resume_state:
        old_generate_images = resume_state.get('generate_images', False)
        if old_generate_images != generate_images:
            print(f"\n⚠️ 检测到图片配置变更（{old_generate_images} -> {generate_images}），将重新生成图片相关文件")
            force_regenerate_images = True
    
    # 自动选择知识点模式
    if auto_select_knowledge:
        candidate_pool_size = max(4, num_questions + max(2, num_questions // 2))
        candidate_min_count = max(4, min(num_questions, candidate_pool_size - 1))
        print(f"\n🤖 启用自动选择知识点模式")
        print(f"场景: {scenario}")
        print(f"候选知识点选择区间: {candidate_min_count}-{candidate_pool_size}")
        print(f"最终题目数量: {num_questions}")

        selected_payload = load_json_if_exists(selected_kp_file) if enable_resume else None
        if selected_payload:
            knowledge_points = selected_payload.get("candidate_knowledge_points", []) or selected_payload.get("knowledge_points", [])
            if knowledge_points:
                print(f"🚀 检测到已保存的知识点选择结果，直接复用")
                print(f"\n✅ 自动选择的候选知识点: {', '.join(knowledge_points)}")
        
        if knowledge_points:
            pass
        else:
            # 加载知识点池
            knowledge_pool = load_all_knowledge_points()
            if not knowledge_pool:
                print("❌ 知识点池为空，无法自动选择知识点")
                return None
            
            # 初始化生成器并选择知识点
            generator = GeminiCoTGenerator(text_provider=text_provider)
            knowledge_points = generator.select_knowledge_points_from_pool(
                knowledge_pool=knowledge_pool,
                scenario=scenario,
                num_questions=candidate_pool_size,
                min_count=candidate_min_count
            )
            save_json(selected_kp_file, {
                "knowledge_points": knowledge_points,
                "candidate_knowledge_points": knowledge_points,
                "scenario": scenario,
                "num_questions": num_questions,
                "candidate_pool_size": candidate_pool_size,
                "candidate_min_count": candidate_min_count
            })

        print(f"\n✅ 自动选择的候选知识点: {', '.join(knowledge_points)}")
    elif knowledge_points is None:
        print("❌ 错误: auto_select_knowledge=False 但未提供 knowledge_points")
        return None
    
    # 设置默认值
    if num_questions is None:
        num_questions = len(knowledge_points)
    
    print(f"\n📁 输出目录: {output_dir}")
    print(f"📚 知识点: {', '.join(knowledge_points)}")
    print(f"🎬 场景: {scenario}")
    print(f"📝 题目数量: {num_questions}")
    
    # ========================================
    # 步骤1: 使用COT生成器生成交互式题目
    # ========================================
    print("\n" + "-" * 50)
    print("[步骤1/6] 使用RAG+COT生成交互式题目...")
    print("-" * 50)
    
    # 如果不是自动选择模式，需要初始化生成器
    if not auto_select_knowledge:
        generator = GeminiCoTGenerator(text_provider=text_provider)

    cot_saved = load_json_if_exists(cot_output_file) if enable_resume else None
    if cot_saved:
        print("🚀 检测到已有COT结果，跳过重新生成")
        cot_result = {
            "knowledge_points": cot_saved.get("knowledge_points", knowledge_points),
            "selected_knowledge_points": cot_saved.get("selected_knowledge_points", cot_saved.get("knowledge_points", knowledge_points)),
            "candidate_knowledge_points": cot_saved.get("candidate_knowledge_points", cot_saved.get("knowledge_points", knowledge_points)),
            "level_knowledge_points": cot_saved.get("level_knowledge_points", []),
            "scenario": cot_saved.get("scenario", scenario),
            "rag_knowledge": {"summary": cot_saved.get("rag_knowledge_summary", "")},
            "story_arc": cot_saved.get("story_arc", {}),
            "story_framework": cot_saved.get("story_framework", {}),
            "initial_result": {"parsed": cot_saved.get("initial_result", {})},
            "reflection": {"parsed": cot_saved.get("reflection", {})},
            "final_result": {"parsed": cot_saved.get("final_result", {})}
        }
    else:
        cot_result = generator.generate_interactive_story(
            knowledge_points=knowledge_points,
            scenario=scenario,
            num_questions=num_questions,
            checkpoint_dir=output_dir,
            enable_resume=enable_resume
        )

        save_data = {
            "knowledge_points": cot_result["knowledge_points"],
            "selected_knowledge_points": cot_result.get("selected_knowledge_points", cot_result.get("knowledge_points", knowledge_points)),
            "candidate_knowledge_points": cot_result.get("candidate_knowledge_points", knowledge_points),
            "level_knowledge_points": cot_result.get("level_knowledge_points", []),
            "scenario": cot_result["scenario"],
            "rag_knowledge_summary": cot_result["rag_knowledge"]["summary"],
            "story_arc": cot_result.get("story_arc", {}),
            "story_framework": cot_result.get("story_framework", {}),
            "initial_result": cot_result["initial_result"]["parsed"],
            "reflection": cot_result["reflection"]["parsed"],
            "final_result": cot_result["final_result"]["parsed"]
        }
        save_json(cot_output_file, save_data)
        print(f"✅ COT原始结果已保存: {cot_output_file}")
    update_resume_state(output_dir, step_1_completed=True)
    
    # ========================================
    # 步骤2: 转换为StoryGraph格式
    # ========================================
    print("\n" + "-" * 50)
    print("[步骤2/6] 转换为StoryGraph格式...")
    print("-" * 50)
    
    story_graph = load_json_if_exists(story_graph_file) if enable_resume else None
    if story_graph is not None:
        print("🚀 检测到已有StoryGraph，跳过转换")
    else:
        converter = CoTToStoryGraphConverter()
        story_graph = converter.convert(cot_result)
        
        # 为fatal节点生成适配的知识点说明（避免直接复用正确答案知识点导致语义冲突）
        story_graph = adapt_fatal_knowledge(story_graph, text_provider=text_provider)
        
        # 验证转换结果
        if not converter.validate_conversion(story_graph):
            print("⚠️ 警告：转换结果验证失败，但仍继续流程")
        
        save_json(story_graph_file, story_graph)
        print(f"✅ StoryGraph已保存: {story_graph_file}")
        
        # 同时保存一份story_graph_from_levels.json（兼容原有流程）
        save_json(story_graph_levels_file, story_graph)
    update_resume_state(output_dir, step_2_completed=True)
    
    if not generate_images:
        print("\n⏭️ 跳过图片生成步骤")
        end_time = datetime.now()
        print("\n" + "=" * 70)
        print(f"COT流程完成（无图片）")
        print(f"总耗时: {end_time - start_time}")
        print("=" * 70)
        return {
            "cot_result": cot_result,
            "story_graph": story_graph,
            "output_dir": output_dir,
            "total_time": str(end_time - start_time)
        }
    
    # ========================================
    # 步骤3: 生成图片提示词
    # ========================================
    print("\n" + "-" * 50)
    print("[步骤3/6] 生成图片提示词...")
    print("-" * 50)
    
    try:
        if enable_resume and os.path.exists(image_prompts_file) and not force_regenerate_images:
            print("🚀 检测到已有图片提示词结果，跳过生成")
        else:
            if force_regenerate_images:
                print("🔄 强制重新生成图片提示词...")
            # 延迟导入，避免langchain依赖问题
            import importlib
            from image import generate_image_prompts_parallel
            importlib.reload(generate_image_prompts_parallel)
            from image.generate_image_prompts_parallel import main_pipeline as generate_image_prompts_pipeline
            
            print("🚀 使用并行版本生成图片提示词...")
            generate_image_prompts_pipeline(
                output_dir=output_dir,
                prompt_template_version=generator_type,
                context_nodes=5,
                text_provider=text_provider
            )
            print("✅ 图片提示词生成完成（并行版本）")
        update_resume_state(output_dir, step_3_completed=True)
    except Exception as e:
        print(f"⚠️ 图片提示词生成失败: {e}")
        import traceback
        traceback.print_exc()
        print("继续流程...")
    
    # ========================================
    # 步骤4: 场景转换检测（与旧版一致）
    # ========================================
    print("\n" + "-" * 50)
    print("[步骤4/6] 检测场景转换点...")
    print("-" * 50)
    
    story_graph_path = image_prompts_file
    
    try:
        if not os.path.exists(story_graph_path):
            print(f"⚠️ 未找到图片提示词文件: {story_graph_path}")
            story_graph_with_transitions_path = story_graph_path
        elif enable_resume and os.path.exists(transitions_file) and not force_regenerate_images:
            story_graph_with_transitions_path = transitions_file
            print("🚀 检测到已有场景转换结果，跳过检测")
        else:
            if force_regenerate_images:
                print("🔄 强制重新检测场景转换...")
            from archive.generate_story_images import detect_scene_transitions
            print("为了生成故事图片的场景一致性，先进行场景转换点的判断...")
            story_graph_with_transitions_path = detect_scene_transitions(story_graph_path)
            print("✅ 场景转换检测完成")
        update_resume_state(output_dir, step_4_completed=True)
    except Exception as e:
        print(f"⚠️ 场景转换检测失败: {e}")
        import traceback
        traceback.print_exc()
        # 如果失败，使用原始文件继续
        story_graph_with_transitions_path = story_graph_path
    
    # ========================================
    # 步骤5: 生成故事图片（并行版本）
    # ========================================
    print("\n" + "-" * 50)
    print("[步骤5/6] 生成故事图片（并行版本）...")
    print("-" * 50)

    resume_image_graph_path = os.path.join(
        output_dir,
        "output",
        f"{os.path.splitext(os.path.basename(story_graph_with_transitions_path))[0]}_with_images.json"
    )
    image_generation_input_path = story_graph_with_transitions_path
    if enable_resume and os.path.exists(resume_image_graph_path):
        image_generation_input_path = resume_image_graph_path
        print(f"🚀 检测到已有带图片结果的故事图，续跑时优先复用: {image_generation_input_path}")
    
    # 检查场景转换文件是否存在，如果不存在说明前面的步骤失败了
    if not os.path.exists(image_generation_input_path):
        print(f"⚠️ 未找到图片生成输入文件: {image_generation_input_path}")
        print("⏭️ 跳过图片生成步骤")
    else:
        # 设置参考图片URL（主角）
        character_reference_url = "https://pub-141831e61e69445289222976a15b6fb3.r2.dev/Image_to_url_V2/--_----imagetourl.cloud-1770284168770-zpdnnd.png"
        
        print(f"📸 图片生成器: {generator_type}")
        if generator_type == "openai":
            print(f"🎨 使用模型: {model}")
        print(f"🖼️ 主角参考图片: {character_reference_url}")

        if image_provider == "yunwu":
            try:
                from image.openai_image_wrapper import generate_story_images_pipeline

                output_images_dir = os.path.join(output_dir, "output_images")
                result = generate_story_images_pipeline(
                    story_graph_path=story_graph_with_transitions_path,
                    output_dir=output_images_dir,
                    reference_image_paths=[character_reference_url],
                    update_graph=True,
                    use_scene_transition=True,
                    model=model,
                    download_images=True,
                    image_provider="yunwu"
                )

                if result.get("updated_graph_path"):
                    final_output_file = os.path.join(output_dir, "output", f"{os.path.splitext(os.path.basename(story_graph_with_transitions_path))[0]}_with_images.json")
                    with open(result["updated_graph_path"], 'r', encoding='utf-8') as src_f:
                        final_graph = _sanitize_story_graph_text(json.load(src_f))
                    with open(final_output_file, 'w', encoding='utf-8') as dst_f:
                        json.dump(final_graph, dst_f, ensure_ascii=False, indent=2)
                    print(f"✅ yunwu 图片生成完成: {final_output_file}")
            except Exception as e:
                print(f"⚠️ yunwu 故事图片生成失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            try:
                from archive.openai_image_wrapper_parallel import ParallelSoraImageGenerator

                print("🚀 使用两阶段并行版本生成故事图片...")

                output_images_dir = os.path.join(output_dir, "output_images")

                # ========================================
                # 步骤5.1: 分析故事图，基于scene_transition分两阶段
                # ========================================
                print("\n📊 分析故事图结构，基于scene_transition分两阶段生成...")

                with open(image_generation_input_path, 'r', encoding='utf-8') as f:
                    story_graph_with_prompts = json.load(f)

                recurring_characters = annotate_recurring_characters(story_graph_with_prompts, min_occurrences=2)
                if recurring_characters:
                    print(f"👥 检测到高频角色: {', '.join([f'{name}({count})' for name, count in recurring_characters.items()])}")
                else:
                    print("👥 未检测到需要统一的高频角色")

                external_character_reference_urls = load_character_reference_urls(output_dir)
                if external_character_reference_urls:
                    print(f"🧷 读取到外部角色参考图配置: {', '.join(external_character_reference_urls.keys())}")

                story_graph_with_prompts, consistency_plan = apply_image_consistency_enrichment(
                    story_graph=story_graph_with_prompts,
                    character_reference_urls=external_character_reference_urls
                )
                plan_path, enriched_graph_path = save_image_consistency_artifacts(
                    output_dir=output_dir,
                    story_graph=story_graph_with_prompts,
                    plan=consistency_plan
                )
                print(f"🧭 一致性计划已生成: {plan_path}")
                print(f"📝 增强提示词故事图已保存: {enriched_graph_path}")

                with open(image_generation_input_path, 'w', encoding='utf-8') as f:
                    json.dump(_sanitize_story_graph_text(story_graph_with_prompts), f, ensure_ascii=False, indent=2)

                # ========================================
                # 步骤5.2: 获取Sora API密钥
                # ========================================
                api_keys = get_image_api_keys(image_provider)

                if not api_keys:
                    print("⚠️ 未找到图像提供商 API Keys，请在.env文件中配置")
                    print("继续流程...")
                else:
                    generator = ParallelSoraImageGenerator(api_keys=api_keys, model=model)
                    nodes = story_graph_with_prompts.get("nodes", [])
                    os.makedirs(output_images_dir, exist_ok=True)

                    recurring_character_refs = generate_recurring_character_reference_images(
                        recurring_characters=recurring_characters,
                        output_dir=output_dir,
                        generator=generator,
                        model=model
                    )
                    if recurring_character_refs:
                        print(f"🧷 已生成 {len(recurring_character_refs)} 个高频角色参考图")

                    selector = IntelligentImageReferenceSelector(
                        story_graph=story_graph_with_prompts,
                        character_reference_url=character_reference_url,
                        recurring_character_refs=recurring_character_refs
                    )

                    for node in nodes:
                        metadata = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
                        existing_image_url = metadata.get("image_url")
                        if existing_image_url:
                            selector.set_node_image_url(node.get("id"), existing_image_url)

                    phase_a_tasks = selector.prepare_phase_a_tasks()
                    print(f"✅ 任务分析完成:")
                    print(f"  - 阶段A（新场景 scene_transition=1）: {len(phase_a_tasks)} 个节点")

                    # ========================================
                    # 步骤5.3: 阶段A - 并行生成新场景节点（只参考主角）
                    # ========================================
                    print(f"\n{'='*50}")
                    print(f"[阶段A] 并行生成新场景节点（scene_transition=1，只参考主角）")
                    print(f"{'='*50}")

                    phase_a_results = generator.generate_batch(
                        generation_tasks=phase_a_tasks,
                        nodes=nodes,
                        max_workers=len(api_keys),
                        batch_label="阶段A-新场景",
                        model=model
                    )

                    # 将阶段A的结果注册到selector，供阶段B查找参考图
                    for node_id, image_url in phase_a_results.items():
                        selector.set_node_image_url(node_id, image_url)

                    persist_image_generation_progress(
                        output_dir=output_dir,
                        story_graph_with_prompts=story_graph_with_prompts,
                        story_graph_with_transitions_path=story_graph_with_transitions_path,
                        output_images_dir=output_images_dir
                    )

                    # ========================================
                    # 步骤5.4: 阶段B - 并行生成同场景节点（参考主角+前图）
                    # ========================================
                    phase_b_tasks = selector.prepare_phase_b_tasks()
                    print(f"\n  - 阶段B（同场景 scene_transition=0）: {len(phase_b_tasks)} 个节点")

                    if phase_b_tasks:
                        print(f"\n{'='*50}")
                        print(f"[阶段B] 并行生成同场景节点（scene_transition=0，参考主角+前图）")
                        print(f"{'='*50}")

                        phase_b_results = generator.generate_batch(
                            generation_tasks=phase_b_tasks,
                            nodes=nodes,
                            max_workers=len(api_keys),
                            batch_label="阶段B-同场景",
                            model=model
                        )

                        persist_image_generation_progress(
                            output_dir=output_dir,
                            story_graph_with_prompts=story_graph_with_prompts,
                            story_graph_with_transitions_path=story_graph_with_transitions_path,
                            output_images_dir=output_images_dir
                        )

                        # 合并结果
                        all_url_results = {**phase_a_results, **phase_b_results}
                    else:
                        all_url_results = phase_a_results
                        print("  ℹ️ 没有同场景节点，跳过阶段B")

                    # ========================================
                    # 步骤5.5: 下载所有图像到本地
                    # ========================================
                    print(f"\n📥 下载所有图像到本地...")
                    downloaded = 0
                    failed_downloads = {}  # node_id -> image_url

                    for node_id, image_url in all_url_results.items():
                        try:
                            local_path = generator._download_image(image_url, output_images_dir, node_id)
                            if local_path:
                                node = next((n for n in nodes if n.get('id') == node_id), None)
                                if node and "metadata" in node:
                                    node["metadata"]["image_path"] = local_path
                                downloaded += 1
                                print(f"  [{downloaded}/{len(all_url_results)}] 已下载 {node_id}")
                            else:
                                failed_downloads[node_id] = image_url
                        except Exception as e:
                            print(f"  ❌ 下载 {node_id} 失败: {e}")
                            failed_downloads[node_id] = image_url

                    # 对失败的下载进行额外重试
                    if failed_downloads:
                        print(f"\n🔄 对 {len(failed_downloads)} 个失败的下载进行额外重试...")
                        time.sleep(5)
                        for node_id, image_url in list(failed_downloads.items()):
                            try:
                                local_path = generator._download_image(image_url, output_images_dir, node_id, max_retries=5)
                                if local_path:
                                    node = next((n for n in nodes if n.get('id') == node_id), None)
                                    if node and "metadata" in node:
                                        node["metadata"]["image_path"] = local_path
                                    downloaded += 1
                                    del failed_downloads[node_id]
                                    print(f"  ✅ 重试成功 {node_id} [{downloaded}/{len(all_url_results)}]")
                            except Exception as e:
                                print(f"  ❌ 重试下载 {node_id} 仍然失败: {e}")

                        if failed_downloads:
                            print(f"\n⚠️ 仍有 {len(failed_downloads)} 个图像下载失败: {', '.join(failed_downloads.keys())}")

                    # ========================================
                    # 步骤5.6: 保存最终JSON
                    # ========================================
                    output_file, final_output_file = persist_image_generation_progress(
                        output_dir=output_dir,
                        story_graph_with_prompts=story_graph_with_prompts,
                        story_graph_with_transitions_path=story_graph_with_transitions_path,
                        output_images_dir=output_images_dir
                    )

                    print(f"\n✅ 两阶段图片生成完成")
                    print(f"  - 总生成: {len(all_url_results)} 个URL")
                    print(f"  - 总下载: {downloaded} 个图像")
                    print(f"  📁 最终故事图: {final_output_file}")

            except Exception as e:
                print(f"⚠️ 故事图片生成失败: {e}")
                import traceback
                traceback.print_exc()
    
    # ========================================
    # 步骤6: 验证图片生成结果
    # ========================================
    print("\n" + "-" * 50)
    print("[步骤6/6] 验证图片生成结果...")
    print("-" * 50)
    
    try:
        # 检查最终的JSON文件（与旧版路径一致）
        # 文件名基于输入文件名追加 _with_images 后缀
        input_basename = os.path.splitext(os.path.basename(story_graph_with_transitions_path))[0]
        json_with_images = os.path.join(output_dir, "output", f"{input_basename}_with_images.json")
        
        if not os.path.exists(json_with_images):
            print(f"⚠️ 未找到图片JSON文件: {json_with_images}")
            print("⏭️ 跳过验证")
        else:
            with open(json_with_images, 'r', encoding='utf-8') as f:
                result_graph = json.load(f)
            
            nodes_with_images = sum(1 for node in result_graph.get('nodes', []) 
                                   if 'metadata' in node and 'image_path' in node.get('metadata', {}))
            total_nodes = len(result_graph.get('nodes', []))
            
            print(f"✅ 图片生成验证完成")
            print(f"📊 统计:")
            print(f"  - 总节点数: {total_nodes}")
            print(f"  - 已生成图像: {nodes_with_images}")
            print(f"  - 成功率: {nodes_with_images/total_nodes*100:.1f}%")
            print(f"📁 最终故事图: {json_with_images}")
    except Exception as e:
        print(f"⚠️ 结果验证失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================
    # 完成
    # ========================================
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print(f"COT完整流程完成于: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_time}")
    print("=" * 70)
    print(f"\n📂 所有文件已生成在 {output_dir} 目录中:")
    print("  - cot_raw_result.json: COT生成器原始输出")
    print("  - enhanced_story_graph.json: StoryGraph格式")
    print("  - story_graph_with_image_prompts.json: 带图片提示词")
    print("  - output_images/: 生成的图片（{node_id}.png格式）")
    print("  - story_graph_with_images.json: 带图片URL的完整故事图")
    print("\n💡 提示: 可以使用 story_visualizer.py 启动GUI查看结果")
    
    return {
        "cot_result": cot_result,
        "story_graph": story_graph,
        "output_dir": output_dir,
        "total_time": str(total_time)
    }


def main():
    """主函数 - 交互式运行"""
    print("\n" + "=" * 70)
    print("COT交互式物理教育故事生成系统")
    print("=" * 70)
    
    # 默认配置 - 恢复4个知识点，添加滑轮组
    default_knowledge_points = [
        "摩擦力", "杠杆原理", "滑轮组", "重心与稳定性"
    ]
    default_scenario = "帮爷爷翻修老房子：需要移动沉重的木箱、撇起巨大的石板、把材料吊到二楼、搬动高大的衣柜，在这个过程中学习物理知识"
    default_num_questions = 8
    
    print(f"\n默认知识点: {', '.join(default_knowledge_points)}")
    print(f"默认场景: {default_scenario}")

    # 初始化配置变量
    scenario = default_scenario
    num_questions = default_num_questions
    text_provider = "aigcbest"
    image_provider = "legacy"
    generate_images = False
    auto_select_knowledge = False
    knowledge_points = default_knowledge_points
    model = "gpt-image-1"

    resume_dir = None
    resume_input = input("\n是否从已有输出目录断点续跑? (y/n, 默认n): ").strip().lower()
    if resume_input == 'y':
        resume_dir_input = input("请输入已有输出目录路径: ").strip()
        if resume_dir_input:
            resume_dir = resume_dir_input
    
    # 如果选择续跑，尝试从 resume_state.json 读取配置
    if resume_dir:
        resume_state_file = os.path.join(resume_dir, "output", "resume_state.json")
        resume_state = load_json_if_exists(resume_state_file)
        
        if resume_state:
            print(f"\n🚀 检测到续跑配置，之前的配置：")
            print(f"  - 场景: {resume_state.get('scenario', 'N/A')}")
            print(f"  - 题目数量: {resume_state.get('num_questions', 'N/A')}")
            print(f"  - 文本提供商: {resume_state.get('text_provider', 'N/A')}")
            print(f"  - 图像提供商: {resume_state.get('image_provider', 'N/A')}")
            print(f"  - 是否生成图片: {resume_state.get('generate_images', 'N/A')}")
            print(f"  - 是否自动选择知识点: {resume_state.get('auto_select_knowledge', 'N/A')}")
            
            # 询问是否修改配置
            modify_config = input("\n是否要修改配置? (y/n, 默认n): ").strip().lower()
            
            if modify_config == 'y':
                # 允许修改配置
                print("\n可修改的配置项：")
                print("  1. 是否生成图片")
                print("  2. 文本提供商")
                print("  3. 图像提供商")
                print("  4. 修改其他配置（跳过）")
                
                modify_choice = input("请选择要修改的配置项 (1/2/3/4, 默认4): ").strip()
                
                if modify_choice == '1':
                    gen_images = input("是否生成图片? (y/n): ").strip().lower()
                    generate_images = gen_images == 'y'
                    
                    if generate_images:
                        print("\n可用的图像生成模型:")
                        print("  1. gpt-image-1")
                        print("  2. gpt-image-2 (默认，推荐)")
                        print("  3. gpt-image-1.5")
                        model_choice = input("请选择模型 (1/2/3, 默认2): ").strip()
                        
                        if model_choice == "1":
                            model = "gpt-image-1"
                        elif model_choice == "3":
                            model = "gpt-image-1.5"
                        else:
                            model = "gpt-image-2"
                elif modify_choice == '2':
                    print("\n可用的文本提供商:")
                    print("  1. aigcbest")
                    print("  2. yunwu")
                    text_provider_choice = input("请选择文本提供商 (1/2): ").strip()
                    text_provider = "yunwu" if text_provider_choice == "2" else "aigcbest"
                elif modify_choice == '3':
                    print("\n可用的图像提供商:")
                    print("  1. legacy")
                    print("  2. yunwu")
                    image_provider_choice = input("请选择图像提供商 (1/2): ").strip()
                    image_provider = "yunwu" if image_provider_choice == "2" else "legacy"
                
                # 使用续跑配置（可能被修改）
                scenario = resume_state.get('scenario', default_scenario)
                num_questions = resume_state.get('num_questions', default_num_questions)
                text_provider = text_provider or resume_state.get('text_provider', 'aigcbest')
                image_provider = image_provider or resume_state.get('image_provider', 'legacy')
                generate_images = generate_images if 'generate_images' in locals() else resume_state.get('generate_images', False)
                auto_select_knowledge = resume_state.get('auto_select_knowledge', False)
                knowledge_points = None if auto_select_knowledge else default_knowledge_points
                model = resume_state.get('model', 'gpt-image-2')
            else:
                # 直接使用续跑配置
                scenario = resume_state.get('scenario', default_scenario)
                num_questions = resume_state.get('num_questions', default_num_questions)
                text_provider = resume_state.get('text_provider', 'aigcbest')
                image_provider = resume_state.get('image_provider', 'legacy')
                generate_images = resume_state.get('generate_images', False)
                auto_select_knowledge = resume_state.get('auto_select_knowledge', False)
                knowledge_points = None if auto_select_knowledge else default_knowledge_points
                model = resume_state.get('model', 'gpt-image-2')
        else:
            print(f"\n⚠️ 未找到续跑配置文件，将使用默认配置")
            # 询问是否修改配置
            modify_config = input("\n是否要修改配置? (y/n, 默认n): ").strip().lower()
            
            if modify_config == 'y':
                # 允许修改配置
                print("\n可修改的配置项：")
                print("  1. 是否生成图片")
                print("  2. 文本提供商")
                print("  3. 图像提供商")
                print("  4. 修改其他配置（跳过）")
                
                modify_choice = input("请选择要修改的配置项 (1/2/3/4, 默认4): ").strip()
                
                if modify_choice == '1':
                    gen_images = input("是否生成图片? (y/n): ").strip().lower()
                    generate_images = gen_images == 'y'
                    
                    if generate_images:
                        print("\n可用的图像生成模型:")
                        print("  1. gpt-image-1")
                        print("  2. gpt-image-2 (默认，推荐)")
                        print("  3. gpt-image-1.5")
                        model_choice = input("请选择模型 (1/2/3, 默认2): ").strip()
                        
                        if model_choice == "1":
                            model = "gpt-image-1"
                        elif model_choice == "3":
                            model = "gpt-image-1.5"
                        else:
                            model = "gpt-image-2"
                elif modify_choice == '2':
                    print("\n可用的文本提供商:")
                    print("  1. aigcbest")
                    print("  2. yunwu")
                    text_provider_choice = input("请选择文本提供商 (1/2): ").strip()
                    text_provider = "yunwu" if text_provider_choice == "2" else "aigcbest"
                elif modify_choice == '3':
                    print("\n可用的图像提供商:")
                    print("  1. legacy")
                    print("  2. yunwu")
                    image_provider_choice = input("请选择图像提供商 (1/2): ").strip()
                    image_provider = "yunwu" if image_provider_choice == "2" else "legacy"
            else:
                # 使用默认配置
                scenario = default_scenario
                num_questions = default_num_questions
                text_provider = "aigcbest"
                image_provider = "legacy"
                generate_images = False
                auto_select_knowledge = False
                knowledge_points = default_knowledge_points
                model = "gpt-image-2"
    else:
        # 新建任务，询问配置
        # 询问是否使用默认配置
        use_default = input("\n是否使用默认配置? (y/n, 默认y): ").strip().lower()
        
        if use_default != 'n':
            auto_select_knowledge = False
            knowledge_points = default_knowledge_points
            scenario = default_scenario
        else:
            # 自定义配置
            auto_select_input = input("是否让大模型从200个知识点池中自动选择知识点? (y/n, 默认n): ").strip().lower()
            auto_select_knowledge = auto_select_input == 'y'
            
            if auto_select_knowledge:
                knowledge_points = None
                scenario = input("请输入故事场景: ").strip()
                if not scenario:
                    scenario = default_scenario
            else:
                kp_input = input("请输入知识点（用逗号分隔）: ").strip()
                knowledge_points = [kp.strip() for kp in kp_input.split(",") if kp.strip()]
                if not knowledge_points:
                    knowledge_points = default_knowledge_points
                
                scenario = input("请输入故事场景: ").strip()
                if not scenario:
                    scenario = default_scenario
        
        # 询问是否生成图片
        gen_images = input("\n是否生成图片? (y/n, 默认n): ").strip().lower()
        generate_images = gen_images == 'y'

        print("\n可用的文本提供商:")
        print("  1. aigcbest (默认，当前旧通道)")
        print("  2. yunwu")
        text_provider_choice = input("请选择文本提供商 (1/2, 默认1): ").strip()
        text_provider = "yunwu" if text_provider_choice == "2" else "aigcbest"

        image_provider = "legacy"
        
        # 如果生成图片，询问使用的模型
        model = "gpt-image-2"  # 默认模型
        if generate_images:
            print("\n可用的图像提供商:")
            print("  1. legacy (默认，当前旧通道)")
            print("  2. yunwu")
            image_provider_choice = input("请选择图像提供商 (1/2, 默认1): ").strip()
            image_provider = "yunwu" if image_provider_choice == "2" else "legacy"

            print("\n可用的图像生成模型:")
            print("  1. gpt-image-1")
            print("  2. gpt-image-2 (默认，推荐)")
            print("  3. gpt-image-1.5")
            model_choice = input("请选择模型 (1/2/3, 默认2): ").strip()
            
            if model_choice == "1":
                model = "gpt-image-1"
            elif model_choice == "3":
                model = "gpt-image-1.5"
            else:
                model = "gpt-image-2"
            
            print(f"✅ 已选择模型: {model}")
    
    # 运行流程
    result = cot_pipeline(
        knowledge_points=knowledge_points,
        scenario=scenario,
        num_questions=num_questions,
        output_dir=resume_dir,
        generate_images=generate_images,
        model=model,
        text_provider=text_provider,
        image_provider=image_provider,
        auto_select_knowledge=auto_select_knowledge,
        enable_resume=True
    )
    
    print(f"\n✅ 完成！输出目录: {result['output_dir']}")


if __name__ == "__main__":
    main()
