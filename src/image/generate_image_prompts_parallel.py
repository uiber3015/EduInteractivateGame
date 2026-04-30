#!/usr/bin/env python3
"""
画面级提示词生成器 - 并行版本
用于为故事图中的每个节点生成画面级提示词，并添加到metadata中
支持多API Key并行生成以提高速度（Gemini + Sora）
"""

import json
import os
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from utils.model_provider_config import get_text_model_config, get_text_api_keys

# 加载环境变量
load_dotenv()

NO_TEXT_IMAGE_RULE = "画面中绝对不要出现任何可读文字、字母、数字、标志、logo、水印、标签、海报文字、屏幕文字、路牌文字、书本文字或UI文案；如果场景中有展板、屏幕、书本、牌子，只保留物体外形，不渲染任何可识别内容。"

def _sanitize_llm_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.strip()

    fenced_match = re.search(r"```(?:text|markdown|md)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    return cleaned.strip()


class ParallelImagePromptGenerator:
    """画面级提示词生成器（并行版本）- 支持Gemini和Sora"""
    
    def __init__(self, 
                 gemini_api_keys: List[str] = None,
                 sora_api_keys: List[str] = None,
                 context_nodes: int = 2, 
                 prompt_template_version: int = 1,
                 text_provider: str = None):
        """
        初始化提示词生成器
        
        Args:
            gemini_api_keys: Gemini API密钥列表
            sora_api_keys: Sora API密钥列表
            context_nodes: 用于参考的上下文节点数量，默认为2
            prompt_template_version: 提示词模板版本
        """
        self.context_nodes = context_nodes
        text_config = get_text_model_config(text_provider)
        self.text_provider = text_config["provider"]
        self.gemini_api_keys = gemini_api_keys or get_text_api_keys(text_provider)
        self.sora_api_keys = sora_api_keys or []
        
        print(f"✓ 使用 {len(self.gemini_api_keys)} 个文本模型 API密钥进行并行生成")
        if self.sora_api_keys:
            print(f"✓ 使用 {len(self.sora_api_keys)} 个Sora API密钥进行并行生成")
        
        # 为每个Gemini API密钥创建一个LLM客户端
        self.llm_clients = []
        for i, api_key in enumerate(self.gemini_api_keys):
            llm = ChatOpenAI(
                model=text_config["model"],
                temperature=0.7,
                openai_api_key=api_key,
                openai_api_base=text_config["base_url"],
                request_timeout=60
            )
            self.llm_clients.append(llm)
        
        print(f"✓ 已创建 {len(self.llm_clients)} 个 {text_config['model']} 客户端")
        
        # 线程锁，用于轮换API密钥
        self.lock = threading.Lock()
        self.current_client_index = 0
        self.api_usage_stats = {
            'gemini': {i: 0 for i in range(len(self.gemini_api_keys))},
            'sora': {i: 0 for i in range(len(self.sora_api_keys))}
        }
        
        # 创建提示词模板 - 优化版：减少技术细节，让描述更自然
        self.image_prompt_template = PromptTemplate(
            input_variables=["node_content", "node_type", "context_content", "future_content", "no_text_rule"],
            template="""
            请为**当前**故事节点内容生成一个生动自然的画面级提示词，用于AI图像生成。
            
            前面节点内容（用于参考）：{context_content}
            后续节点内容（用于校准当前画面，避免与正确情节冲突）：{future_content}
            当前节点内容：{node_content}
            节点类型：{node_type}
            
            **核心要求**：
            1. 提示词必须包含三个要素：
               - 主体：人物的动作、表情、姿态
               - 背景：环境、光线、氛围
               - 构图：视角（近景/中景/远景）
            
            2. **重要：描述要自然流畅**
               - 用日常语言描述动作，不要用技术术语
               - 不要写具体尺寸（如"约1.5米"、"直径10厘米"）
               - 不要写过于技术化的物理结构描述
               - 用感受和情感来描述，而不是精确数据
            
            3. **工具描述的正确方式**：
               - 撇棍/杠杆："结实的铁棍"、"长长的撇棍"
               - 圆木棍："几根圆滑的木棍"、"粗壮的圆木"
               - 滑轮："挂在阳台上的滑轮"、"铁质滑轮"
               - 绳子："粗糙的绳子"、"结实的麻绳"
            
            4. 吉卜力风格，温馨怀旧的氛围
            5. 原节点中的"你"用"人物"替换
            6. 提示词长度30-50字
            7. 最后加一句"人物形象参考输入的第一张，背景参考第二张（如果有的话）"
            8. 如果后续节点已经透露了正确路线、地形、水流速度、坡度、障碍物、目标位置等关键信息，当前画面必须与这些事实保持兼容
            9. 优先画出对后续正确情节仍然成立的稳定场景事实，不要把环境夸张成会与后续正确结果矛盾的样子
            10. 例如后续是平缓小溪，就不要画成湍急大河；后续是缓坡通路，就不要画成断崖绝壁
            11. {no_text_rule}
            
            **错误示例**（不要这样写）：
            × "人物双手紧握一根笔直的长铁棍（约1.5米长）"
            × "几根直径约10厘米的圆柱形木棍"
            
            **正确示例**（应该这样写）：
            √ "人物弯腰用力，用结实的铁棍撇动沉重的石板"
            √ "人物将圆滑的木棍垫在柜子底下，轻松推动"
            
            请只返回画面级提示词，不要添加任何解释或其他文字。
            """
        )
    
    def _get_next_client(self):
        """获取下一个可用的LLM客户端（轮换）"""
        with self.lock:
            client = self.llm_clients[self.current_client_index]
            index = self.current_client_index
            self.current_client_index = (self.current_client_index + 1) % len(self.llm_clients)
            self.api_usage_stats['gemini'][index] += 1
            return client
    
    def get_context_nodes(self, story_graph: Dict, current_node_id: str) -> str:
        """
        获取当前节点的上下文节点内容
        
        Args:
            story_graph: 故事图数据结构
            current_node_id: 当前节点ID
            
        Returns:
            上下文节点内容的字符串表示
        """
        context_parts = []
        edges = story_graph.get("edges", {})
        nodes = {node["id"]: node for node in story_graph.get("nodes", [])}
        
        # 获取前面的节点（使用BFS查找前驱节点）
        prev_nodes = self._find_previous_nodes(edges, current_node_id, self.context_nodes)
        for node_id in prev_nodes:
            node = nodes.get(node_id, {})
            context_parts.append(node.get('content', ''))
        
        return "\n".join(context_parts)

    def get_future_nodes(self, story_graph: Dict, current_node_id: str) -> str:
        """
        获取当前节点后续的关键节点内容，用于校准当前画面与正确情节的兼容性
        """
        future_parts = []
        edges = story_graph.get("edges", {})
        nodes = {node["id"]: node for node in story_graph.get("nodes", [])}

        next_nodes = self._find_next_nodes(edges, current_node_id, self.context_nodes)
        for node_id in next_nodes:
            node = nodes.get(node_id, {})
            content = node.get('content', '')
            metadata = node.get('metadata', {}) if isinstance(node.get('metadata'), dict) else {}
            choice_option = metadata.get('choice_option', '')
            combined = content
            if choice_option:
                combined = f"选项：{choice_option}；后续：{content}"
            if combined:
                future_parts.append(combined)

        return "\n".join(future_parts)
    
    def _find_previous_nodes(self, edges: Dict, target_node_id: str, max_count: int) -> List[str]:
        """
        查找目标节点的前驱节点
        
        Args:
            edges: 边集合
            target_node_id: 目标节点ID
            max_count: 最大查找数量
            
        Returns:
            前驱节点ID列表，按故事发展顺序排序
        """
        # 构建反向边映射
        reverse_edges = {}
        for source, targets in edges.items():
            for target in targets:
                if target not in reverse_edges:
                    reverse_edges[target] = []
                reverse_edges[target].append(source)
        
        # 简化逻辑：每次只取第一个前驱节点，递归查找k个节点
        result = []
        current_id = target_node_id
        
        for _ in range(max_count):
            # 获取当前节点的前驱节点
            predecessors = reverse_edges.get(current_id, [])
            
            # 如果没有前驱节点，结束查找
            if not predecessors:
                break
                
            # 取第一个前驱节点
            prev_id = predecessors[0]
            result.append(prev_id)
            
            # 继续基于这个前驱节点往前找
            current_id = prev_id
        
        # 反转结果，使其按故事发展顺序排列
        return result[::-1]

    def _find_next_nodes(self, edges: Dict, source_node_id: str, max_count: int) -> List[str]:
        """
        沿主线查找目标节点的后继节点。
        对问题节点而言，默认取第一个后继分支，通常就是正确选项路径。
        """
        result = []
        current_id = source_node_id

        for _ in range(max_count):
            successors = edges.get(current_id, [])
            if not successors:
                break

            next_id = successors[0]
            result.append(next_id)
            current_id = next_id

        return result

    def generate_single_prompt(self, node_content: str, node_type: str, context_content: str = "", future_content: str = "") -> str:
        """
        为单个节点生成画面级提示词
        
        Args:
            node_content: 节点内容
            node_type: 节点类型
            context_content: 上下文内容
            
        Returns:
            生成的画面级提示词
        """
        try:
            # 获取一个LLM客户端
            llm = self._get_next_client()
            
            # 创建提示词生成链
            chain = self.image_prompt_template | llm
            
            # 调用LLM
            response = chain.invoke({
                "node_content": node_content,
                "node_type": node_type,
                "context_content": context_content,
                "future_content": future_content,
                "no_text_rule": NO_TEXT_IMAGE_RULE
            })
            
            if hasattr(response, 'content'):
                return _sanitize_llm_text(response.content)
            else:
                return _sanitize_llm_text(str(response))
                
        except Exception as e:
            print(f"⚠️ 生成提示词时出错: {e}")
            return f"吉卜力风格，中景镜头，描绘'{node_content[:30]}...'的场景，{node_type}类型节点，温暖怀旧的氛围。人物形象参考输入的第一张，背景参考第二张（如果有的话）。{NO_TEXT_IMAGE_RULE}"
    
    def add_image_prompts_parallel(self, json_file_path: str, output_file_path: Optional[str] = None, max_workers: Optional[int] = None) -> None:
        """
        并行为JSON文件中的所有节点添加画面级提示词
        
        Args:
            json_file_path: 输入JSON文件路径
            output_file_path: 输出JSON文件路径，如果为None则覆盖原文件
            max_workers: 最大并发数，默认为API密钥数量
        """
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            story_graph = json.load(f)
        
        # 确定输出文件路径
        if output_file_path is None:
            output_file_path = json_file_path
        
        # 确定最大并发数
        if max_workers is None:
            max_workers = len(self.gemini_api_keys)
        
        nodes = story_graph.get("nodes", [])
        total_nodes = len(nodes)
        
        print(f"\n🚀 开始并行生成 {total_nodes} 个节点的图片提示词...")
        print(f"📊 并发数: {max_workers}")
        
        # 准备任务列表
        tasks = []
        for i, node in enumerate(nodes):
            node_id = node.get("id", "")
            if "metadata" in node and isinstance(node.get("metadata"), dict):
                existing_prompt = node["metadata"].get("image_prompt")
                if isinstance(existing_prompt, str):
                    node["metadata"]["image_prompt"] = _sanitize_llm_text(existing_prompt)
            
            # 检查是否已有image_prompt，跳过已处理的节点
            if "metadata" in node and "image_prompt" in node.get("metadata", {}):
                continue
            
            node_content = node.get("content", "")
            node_type = node.get("type", "normal")
            context_content = self.get_context_nodes(story_graph, node_id)
            
            future_content = self.get_future_nodes(story_graph, node_id)

            tasks.append((i, node, node_id, node_content, node_type, context_content, future_content))
        
        if not tasks:
            print("✅ 所有节点已有图片提示词，无需生成")
            return
        
        print(f"📝 需要生成 {len(tasks)} 个节点的提示词")
        
        # 使用线程池并行生成
        start_time = time.time()
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {}
            for task in tasks:
                i, node, node_id, node_content, node_type, context_content, future_content = task
                future = executor.submit(self.generate_single_prompt, node_content, node_type, context_content, future_content)
                future_to_task[future] = (i, node, node_id)
            
            # 收集结果
            for future in as_completed(future_to_task):
                i, node, node_id = future_to_task[future]
                try:
                    image_prompt = _sanitize_llm_text(future.result())
                    
                    # 添加到metadata中
                    if "metadata" not in node:
                        node["metadata"] = {}
                    node["metadata"]["image_prompt"] = image_prompt
                    
                    completed += 1
                    print(f"[{completed}/{len(tasks)}] 已为节点 {node_id} 生成画面提示词\n{image_prompt}")
                    
                except Exception as e:
                    print(f"❌ 节点 {node_id} 生成失败: {e}")
        
        # 保存结果
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(story_graph, f, ensure_ascii=False, indent=2)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ 已成功为所有节点添加画面级提示词")
        print(f"⏱️ 总耗时: {elapsed_time:.2f}秒")
        print(f"📁 结果保存到: {output_file_path}")
        
        # 打印API使用统计
        print(f"\n📊 API使用统计:")
        for i, count in self.api_usage_stats['gemini'].items():
            if count > 0:
                print(f"  Gemini API密钥{i+1}: {count}次调用")


def main_pipeline(output_dir: str, 
                 prompt_template_version="tencent", 
                 context_nodes=5, 
                 input_file: str = None, 
                 gemini_api_keys: List[str] = None,
                 sora_api_keys: List[str] = None,
                 text_provider: str = None):
    """主函数
    
    Args:
        output_dir: 输出目录
        prompt_template_version: 提示词模板版本
        context_nodes: 上下文节点数量
        input_file: 输入文件名
        gemini_api_keys: Gemini API密钥列表
        sora_api_keys: Sora API密钥列表
    """
    # 如果没有提供Gemini API密钥，从环境变量读取
    if gemini_api_keys is None:
        gemini_api_keys = get_text_api_keys(text_provider)
        if not gemini_api_keys:
            print("⚠️ 未设置文本模型 API Keys，请在.env文件中配置")
            return
    
    # 如果没有提供Sora API密钥，从环境变量读取
    if sora_api_keys is None:
        sora_keys_env = os.getenv("SORA_API_KEYS", "")
        sora_api_keys = [k.strip() for k in sora_keys_env.split(",") if k.strip()] if sora_keys_env else []
    
    # 确定输入文件
    if input_file is None:
        input_file = "enhanced_story_graph.json"
    
    input_path = os.path.join(output_dir, "output", input_file)
    
    if not os.path.exists(input_path):
        print(f"找不到输入文件: {input_path}")
        return
    
    print(f"找到输入文件: {input_file}")
    
    # 确定输出文件
    output_file = "story_graph_with_image_prompts.json"
    output_path = os.path.join(output_dir, "output", output_file)
    
    # 创建并行生成器
    generator = ParallelImagePromptGenerator(
        gemini_api_keys=gemini_api_keys,
        sora_api_keys=sora_api_keys,
        context_nodes=context_nodes,
        prompt_template_version=1,
        text_provider=text_provider
    )
    
    # 并行生成提示词
    generator.add_image_prompts_parallel(input_path, output_path)
    
    print(f"已成功为所有节点添加画面级提示词，结果保存到: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "output/cot_20260205_145208"
    
    main_pipeline(output_dir)
