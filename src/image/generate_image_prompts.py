#!/usr/bin/env python3
"""
画面级提示词生成器
用于为故事图中的每个节点生成画面级提示词，并添加到metadata中
支持并行生成以提高速度
"""

import json
import os
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

NO_TEXT_IMAGE_RULE = "画面中绝对不要出现任何可读文字、字母、数字、标志、logo、水印、标签、海报文字、屏幕文字、路牌文字、书本文字或UI文案；如果场景中有展板、屏幕、书本、牌子，只保留物体外形，不渲染任何可识别内容。"

# 加载环境变量
load_dotenv()


def _sanitize_llm_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.strip()

    fenced_match = re.search(r"```(?:text|markdown|md)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    return cleaned.strip()


class ImagePromptGenerator:
    """画面级提示词生成器（支持并行）"""
    
    # 类变量：决定使用哪套提示词模板 (1: 原始模板, 2: 新增模板)
    PROMPT_TEMPLATE_VERSION = 1
    
    def __init__(self, api_keys: Optional[List[str]] = None, context_nodes: int = 2, prompt_template_version: int = 1, use_gemini: bool = True):
        """
        初始化提示词生成器
        
        Args:
            api_keys: API密钥列表，如果为None则从环境变量获取单个密钥
            context_nodes: 用于参考的上下文节点数量，默认为2
            prompt_template_version: 提示词模板版本
            use_gemini: 是否使用Gemini模型（默认True，速度更快）
        """
        self.context_nodes = context_nodes
        self.use_gemini = use_gemini
        
        # 处理API密钥
        if api_keys and len(api_keys) > 0:
            self.api_keys = api_keys
            print(f"✓ 使用 {len(api_keys)} 个API密钥进行并行生成")
        else:
            # 单个密钥模式
            if use_gemini:
                single_key = os.getenv("GEMINI_API_KEY")
            else:
                single_key = os.getenv("DEEPSEEK_API_KEY")
            
            if not single_key:
                raise ValueError("请提供API密钥")
            self.api_keys = [single_key]
            print(f"✓ 使用单个API密钥")
        
        # 为每个API密钥创建一个LLM客户端
        self.llm_clients = []
        for api_key in self.api_keys:
            if use_gemini:
                llm = ChatOpenAI(
                    model="gemini-2.5-flash",
                    temperature=0.7,
                    openai_api_key=api_key,
                    openai_api_base="https://api2.aigcbest.top/v1"
                )
            else:
                llm = ChatOpenAI(
                    model="deepseek-chat",
                    temperature=0.7,
                    openai_api_key=api_key,
                    openai_api_base="https://api.deepseek.com/v1"
                )
            self.llm_clients.append(llm)
        
        model_name = "Gemini-2.5-flash" if use_gemini else "DeepSeek"
        print(f"✓ 使用{model_name}模型生成图片提示词")
        
        # 线程锁，用于轮换API密钥
        self.lock = threading.Lock()
        self.current_client_index = 0
        
        # 创建两套画面级提示词生成模板
        self.image_prompt_template_v1 = PromptTemplate(
            input_variables=["node_content", "node_type", "context_content", "future_content", "no_text_rule"],
            template="""
            请为**当前**故事节点内容生成一个详细的画面级提示词，用于AI图像生成。
            
            前面节点内容（用于参考）：{context_content}
            后续节点内容（用于校准当前画面，避免与正确情节冲突）：{future_content}
            当前节点内容：{node_content}
            节点类型：{node_type}
            
            要求：
            1. 提示词必须明确包含三个核心要素：
               - 主体：场景中的主要人物或物体，包括其特征、动作和表情
               - 背景：主体所处的环境、场景和氛围
               - 构图：画面布局、视角和镜头类型（如近景、中景、远景、特写、仰拍、俯拍等）
            2. 三个要素应有机结合，形成完整的视觉描述
            3. 使用具体的形容词和细节描述，突出关键场景和情感氛围
            4. 适合用于AI图像生成（如DALL-E、Midjourney等）
            5. 提示词长度在30-50字之间
            6. 如果是决策节点，要突出表现决策的瞬间和紧张感
            7. 如果是结局节点，要体现出最终结果氛围
             8. 原节点内容中的"你"，用"人物"替换，这个与文生图模型的要求有关
             9. 参考前面节点内容，确保提示词与故事情节连贯一致
             10. 最后加一句“人物形象参考输入的第一张，背景参考第二张（如果有的话）”
             11. 吉卜力风格
             12. 如果后续节点已经透露了正确路线、地形、水流速度、坡度、障碍物、目标位置等关键信息，当前画面必须与这些事实保持兼容
             13. 优先画出对后续正确情节仍然成立的稳定场景事实，不要把环境夸张成会与后续正确结果矛盾的样子
             14. 例如后续是平缓小溪，就不要画成湍急大河；后续是缓坡通路，就不要画成断崖绝壁
             15. **重要：对于物理工具和器械，必须明确描述其形状和使用姿势**：
                - 撬棍/杠杆：描述为"一根笔直的长铁棍（约1.5米长）"，并描述使用姿势（一端插入重物下方，中间有支点，人物握住另一端向下压）
                - 滑轮：描述为"圆形的金属滑轮，中间有轴，边缘有凹槽用于穿绳"
                - 圆木棍：描述为"几根直径约10厘米的圆柱形木棍"
                - 绳索：描述为"粗麻绳或尼龙绳"
                - 确保工具的物理结构正确，避免生成变形或不合理的工具形状
             16. {no_text_rule}
             
             请只返回画面级提示词，不要添加任何解释或其他文字。
             """
        )
        
        # 新增的第二套提示词模板
        self.image_prompt_template_v2 = PromptTemplate(
            input_variables=["node_content", "node_type", "context_content", "future_content", "no_text_rule"],
            template="""
            请为**当前**故事节点内容生成一个详细的画面级提示词，用于AI图像生成。
            
            前面节点内容（用于参考）：{context_content}
            后续节点内容（用于校准当前画面，避免与正确情节冲突）：{future_content}
            当前节点内容：{node_content}
            节点类型：{node_type}
            
            要求：
            1. 生成一个适合儿童教育的卡通风格画面提示词
            2. 描述必须生动有趣，色彩鲜明
            3. 主体：明确指出场景中的主要卡通角色及其特征、动作和表情
            4. 背景：详细描述主体所处的环境、场景和氛围，包括时间、地点和天气等
            5. 构图：指定画面布局、视角和镜头类型（如近景、远景、特写、仰拍等）
            6. 风格：采用吉卜力工作室风格的动漫画面
            7. 情感：突出场景的情感氛围，如快乐、紧张、悲伤或兴奋等
            8. 细节：添加具体的细节描述，如服装、道具、颜色等
             9. 原节点内容中的"你"，用"主角"替换
             10. 参考前面节点内容，确保提示词与故事情节连贯一致
             11. 提示词长度在50-80字之间
             12. 如果后续节点已经透露了正确路线、地形、水流速度、坡度、障碍物、目标位置等关键信息，当前画面必须与这些事实保持兼容
             13. 优先画出对后续正确情节仍然成立的稳定场景事实，不要把环境夸张成会与后续正确结果矛盾的样子
             14. 例如后续是平缓小溪，就不要画成湍急大河；后续是缓坡通路，就不要画成断崖绝壁
             15. {no_text_rule}
            
             请只返回画面级提示词，不要添加任何解释或其他文字。
             """
        )

        self.PROMPT_TEMPLATE_VERSION = prompt_template_version

        # 根据类变量选择使用的模板
        if self.PROMPT_TEMPLATE_VERSION == 1:
            selected_template = self.image_prompt_template_v1
        else:  # 默认使用第二套模板
            selected_template = self.image_prompt_template_v2
        
        # 创建提示词生成链（使用新版langchain的管道操作符）
        self.image_prompt_chain = selected_template | self.llm
    
    def switch_prompt_template(self, version: int) -> None:
        """
        动态切换提示词模板
        
        Args:
            version: 模板版本号 (1 或 2)
        """
        if version not in [1, 2]:
            raise ValueError("模板版本号必须是 1 或 2")
        
        self.PROMPT_TEMPLATE_VERSION = version
        
        # 重新选择模板并更新链
        if version == 1:
            selected_template = self.image_prompt_template_v1
        else:
            selected_template = self.image_prompt_template_v2
            
        self.image_prompt_chain = selected_template | self.llm
        
        print(f"已切换到提示词模板版本 {version}")
    
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

    def generate_image_prompt(self, node_content: str, node_type: str, context_content: str = "", future_content: str = "", timeout: int = 60) -> str:
        """
        为故事节点生成画面级提示词
        
        Args:
            node_content: 节点内容
            node_type: 节点类型（normal, decision, ending等）
            context_content: 上下文节点内容
            timeout: API调用超时时间（秒）
            
        Returns:
            生成的画面级提示词
        """
        import concurrent.futures
        
        def _call_llm():
            response = self.image_prompt_chain.invoke({
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
        
        try:
            # 使用线程池执行API调用，添加超时保护
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_llm)
                image_prompt = future.result(timeout=timeout)
            
            return image_prompt
        except concurrent.futures.TimeoutError:
            print(f"⚠️ API调用超时（{timeout}秒），使用默认提示词")
            return f"吉卜力风格，中景镜头，描绘'{node_content[:30]}...'的场景，{node_type}类型节点，温暖怀旧的氛围。{NO_TEXT_IMAGE_RULE}"
        except Exception as e:
            print(f"生成画面级提示词时出错: {e}")
            # 如果LLM调用失败，返回一个默认提示词
            return f"吉卜力风格，中景镜头，描绘'{node_content[:30]}...'的场景，{node_type}类型节点，温暖怀旧的氛围。{NO_TEXT_IMAGE_RULE}"
    
    def add_image_prompts_to_json(self, json_file_path: str, output_file_path: Optional[str] = None) -> None:
        """
        为JSON文件中的所有节点添加画面级提示词
        
        Args:
            json_file_path: 输入JSON文件路径
            output_file_path: 输出JSON文件路径，如果为None则覆盖原文件
        """
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            story_graph = json.load(f)
        
        # 确定输出文件路径
        if output_file_path is None:
            output_file_path = json_file_path
        
        nodes = story_graph.get("nodes", [])
        total_nodes = len(nodes)
        processed = 0
        skipped = 0
        
        # 为每个节点生成画面级提示词并添加到metadata中
        for i, node in enumerate(nodes):
            node_id = node.get("id", "")
            node_content = node.get("content", "")
            node_type = node.get("type", "normal")
            if "metadata" in node and isinstance(node.get("metadata"), dict):
                existing_prompt = node["metadata"].get("image_prompt")
                if isinstance(existing_prompt, str):
                    node["metadata"]["image_prompt"] = _sanitize_llm_text(existing_prompt)
            
            # 检查是否已有image_prompt，跳过已处理的节点
            if "metadata" in node and "image_prompt" in node.get("metadata", {}):
                skipped += 1
                continue
            
            # 获取上下文节点内容
            context_content = self.get_context_nodes(story_graph, node_id)
            future_content = self.get_future_nodes(story_graph, node_id)
            
            # 生成画面级提示词
            image_prompt = self.generate_image_prompt(node_content, node_type, context_content, future_content)
            
            # 添加到metadata中
            if "metadata" not in node:
                node["metadata"] = {}
            
            node["metadata"]["image_prompt"] = image_prompt
            processed += 1
            
            print(f"[{i+1}/{total_nodes}] 已为节点 {node_id} 生成画面提示词\n{image_prompt}")
            
            # 每处理5个节点保存一次（增量保存，防止中断丢失）
            if processed % 5 == 0:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(story_graph, f, ensure_ascii=False, indent=2)
        
        # 最终保存
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(story_graph, f, ensure_ascii=False, indent=2)
        
        print(f"已成功为所有节点添加画面级提示词，结果保存到: {output_file_path}")


def main_pipeline(output_dir: str, prompt_template_version="tencent", context_nodes=5, input_file: str = None):
    """主函数
    
    Args:
        output_dir: 输出目录
        prompt_template_version: 提示词模板版本
        context_nodes: 上下文节点数
        input_file: 输入文件名（默认自动检测）
    """
    # 输入JSON文件路径 - 支持多种文件名
    if input_file:
        json_file_path = os.path.join(output_dir, "output", input_file)
    else:
        # 按优先级尝试不同的文件名
        possible_files = [
            "enhanced_story_graph.json",  # COT流程生成的文件
            "story_graph_with_choices.json",  # 原流程生成的文件
            "story_graph_from_levels.json"  # 备选文件
        ]
        json_file_path = None
        for fname in possible_files:
            fpath = os.path.join(output_dir, "output", fname)
            if os.path.exists(fpath):
                json_file_path = fpath
                print(f"找到输入文件: {fname}")
                break
        
        if json_file_path is None:
            raise FileNotFoundError(f"在 {output_dir}/output 中未找到可用的故事图文件")
    
    # 输出JSON文件路径
    output_file_path = os.path.join(output_dir, "output", "story_graph_with_image_prompts.json")
    
    # 根据模板版本创建提示词生成器实例，使用传入的context_nodes参数
    if prompt_template_version == "tencent" or prompt_template_version == "openai":
        print("image generate prompt使用tencent或openai模板")
        prompt_generator = ImagePromptGenerator(context_nodes=context_nodes, prompt_template_version=1)
    elif prompt_template_version == "wanx":
        print("image generate prompt使用wanx模板")
        prompt_generator = ImagePromptGenerator(context_nodes=context_nodes, prompt_template_version=2)
    else:
        raise ValueError("不支持的提示词模板版本")
    
    # 为JSON文件中的所有节点添加画面级提示词
    prompt_generator.add_image_prompts_to_json(json_file_path, output_file_path)


if __name__ == "__main__":
    # 输入JSON文件路径（使用原始字符串避免转义问题）
    json_file_path = r"Output1\250912火灾1\output\final_story_graph_with_image_prompts.json"
    
    # 输出JSON文件路径
    output_file_path = r"Output1\250912火灾1\output\final_story_graph_with_image_prompts.json"
    
    # 创建提示词生成器实例，使用默认的context_nodes参数值
    prompt_generator = ImagePromptGenerator(context_nodes=100)
    
    # 为JSON文件中的所有节点添加画面级提示词
    prompt_generator.add_image_prompts_to_json(json_file_path, output_file_path)
