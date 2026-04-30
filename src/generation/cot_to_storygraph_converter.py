"""
COT输出到StoryGraph格式转换器

将cot_web_story_generator_v2.py生成的JSON格式转换为StoryGraph格式，
以便复用现有的图片生成和GUI展示功能。
"""

import os
import sys
import json
import re
from typing import Dict, List, Any

# 添加src目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.StoryGraph import StoryGraph, StoryNode, NodeType


class CoTToStoryGraphConverter:
    """COT输出到StoryGraph格式转换器"""
    
    def __init__(self):
        self.node_counter = 0
    
    def _next_node_id(self) -> str:
        """生成下一个节点ID"""
        self.node_counter += 1
        return f"node_{self.node_counter}"

    def _sanitize_text(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"!\[[^\]]*image[^\]]*\]\([^\)]*\)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[\s*image[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<image[^>]*>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*image\s*[:：].*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _sanitize_payload(self, value):
        if isinstance(value, str):
            return self._sanitize_text(value)
        if isinstance(value, list):
            return [self._sanitize_payload(item) for item in value]
        if isinstance(value, dict):
            return {key: self._sanitize_payload(item) for key, item in value.items()}
        return value

    def _extract_question_knowledge(self, question: Dict, level: Dict) -> Dict[str, Any]:
        primary = (
            question.get("primary_knowledge_point", "")
            or level.get("primary_knowledge_point", "")
            or question.get("knowledge_point", "")
            or level.get("knowledge_point", "")
        )
        supporting = question.get("supporting_knowledge_points", []) or level.get("supporting_knowledge_points", []) or []
        if isinstance(supporting, str):
            supporting = [item.strip() for item in re.split(r"[,，、\n]+", supporting) if item.strip()]
        supporting = [kp for kp in supporting if kp and kp != primary]
        return {
            "primary": primary,
            "supporting": supporting[:2],
            "role": question.get("knowledge_role_in_this_level", "") or level.get("knowledge_role_in_this_level", ""),
        }
    
    def _remove_knowledge_prefix(self, text: str) -> str:
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
    
    def _generate_chapter_title(self, question_id: int, knowledge_point: str, story_context: str, llm_chapter_title: str = None) -> str:
        """
        生成章节标题
        优先使用LLM生成的chapter_title，否则使用知识点名称
        
        Args:
            question_id: 问题ID
            knowledge_point: 知识点名称
            story_context: 场景描述文本
            llm_chapter_title: LLM生成的章节标题（优先使用）
            
        Returns:
            生成的章节标题
        """
        # 优先使用LLM生成的标题
        if llm_chapter_title and llm_chapter_title.strip():
            title = llm_chapter_title.strip()
            # 如果已经包含"第N关"前缀，直接返回
            if title.startswith("第"):
                return title
            return f"第{question_id}关 {title}"
        
        # 备选：使用知识点名称
        return f"第{question_id}关 {knowledge_point}"
    
    def convert(self, cot_result: Dict) -> Dict:
        """
        将COT生成器的输出转换为StoryGraph格式
        
        Args:
            cot_result: COT生成器的输出，包含final_result.parsed
            
        Returns:
            StoryGraph格式的字典
        """
        # 获取最终生成的内容
        if 'final_result' in cot_result:
            content = cot_result['final_result'].get('parsed', {})
        elif 'parsed' in cot_result:
            content = cot_result['parsed']
        else:
            content = cot_result

        content = self._sanitize_payload(content)
        story_framework = self._sanitize_payload(cot_result.get("story_framework", {})) if isinstance(cot_result, dict) else {}
        levels = story_framework.get("levels", []) if isinstance(story_framework, dict) else []
        
        # 重置计数器
        self.node_counter = 0
        
        nodes = []
        edges = {}
        
        # 1. 故事开头节点
        intro_id = self._next_node_id()
        nodes.append({
            "id": intro_id,
            "content": content.get("story_intro", "故事开始..."),
            "type": "normal",
            "metadata": {}
        })
        
        prev_node_id = intro_id
        questions = content.get("questions", [])
        
        # 2. 遍历每个问题
        for q_idx, question in enumerate(questions):
            question_id = question.get("question_id", q_idx + 1)
            level = levels[q_idx] if q_idx < len(levels) else {}
            knowledge_meta = self._extract_question_knowledge(question, level)
            knowledge_point = knowledge_meta["primary"]
            supporting_knowledge_points = knowledge_meta["supporting"]
            knowledge_role = knowledge_meta["role"]
            
            # 2.1 章节开始节点 (story_context)
            ep_start_id = self._next_node_id()
            story_context = question.get("story_context", "")
            
            # 如果有上一题的过渡，合并到story_context
            if q_idx > 0:
                prev_transition = questions[q_idx - 1].get("transition_to_next", "")
                if prev_transition:
                    story_context = prev_transition + " " + story_context
            
            # 优先使用LLM生成的chapter_title
            llm_chapter_title = question.get("chapter_title", None)
            chapter_title = self._generate_chapter_title(question_id, knowledge_point, story_context, llm_chapter_title)
            
            nodes.append({
                "id": ep_start_id,
                "content": story_context,
                "type": "ep_start",
                "metadata": {
                    "chapter_title": chapter_title,
                    "knowledge_point": knowledge_point,
                    "primary_knowledge_point": knowledge_point,
                    "supporting_knowledge_points": supporting_knowledge_points,
                    "knowledge_role_in_this_level": knowledge_role
                }
            })
            edges[prev_node_id] = [ep_start_id]
            
            # 2.2 问题描述节点
            question_node_id = self._next_node_id()
            nodes.append({
                "id": question_node_id,
                "content": question.get("question_text", ""),
                "type": "normal",
                "metadata": {
                    "knowledge_point": knowledge_point,
                    "primary_knowledge_point": knowledge_point,
                    "supporting_knowledge_points": supporting_knowledge_points
                }
            })
            edges[ep_start_id] = [question_node_id]
            
            # 2.3 处理选项
            options = question.get("options", [])
            correct_opt = None
            wrong_opts = []
            
            for opt in options:
                if opt.get("is_correct", False):
                    correct_opt = opt
                else:
                    wrong_opts.append(opt)
            
            if not correct_opt:
                # 如果没有标记正确选项，默认第一个为正确
                if options:
                    correct_opt = options[0]
                    wrong_opts = options[1:]
                else:
                    continue
            
            # 记录决策点前的节点ID，用于错误后返回
            pre_decision_node_id = question_node_id
            
            # 正确选项 -> decision节点
            correct_decision_id = self._next_node_id()
            correct_option_text = correct_opt.get("option_text", "")
            # 优先使用 action_feedback 作为 decision 节点正文，兼容旧版 result
            correct_transition = correct_opt.get("action_feedback", "") or correct_opt.get("result", correct_option_text)

            nodes.append({
                "id": correct_decision_id,
                "content": correct_transition,
                "type": "decision",
                "metadata": {
                    "choice_option": correct_option_text
                }
            })
            
            # 正确选项的结果 -> ep_end节点
            correct_result_id = self._next_node_id()
            
            # 构建物理知识说明（优先使用新字段 analysis + knowledge，兼容旧字段 explanation）
            analysis_text = correct_opt.get('analysis', '') or correct_opt.get('explanation', '')
            knowledge_text = correct_opt.get('knowledge', '')
            
            # 去掉知识点文本开头的"知识点名称："格式，但保留"知识点："标签
            # 例如："自然导航原理：受地球公转影响..." -> "知识点：受地球公转影响..."
            if knowledge_text:
                # 检测并移除开头的"XXX："格式
                knowledge_text = self._remove_knowledge_prefix(knowledge_text)
            
            knowledge_header = knowledge_point or question.get("knowledge_point", "")
            if supporting_knowledge_points:
                knowledge_header += f"\n辅助知识点：{'、'.join(supporting_knowledge_points)}"

            physics_knowledge = f"{knowledge_header}\n\n"
            physics_knowledge += f"解析：{analysis_text}\n\n"
            if knowledge_text:
                physics_knowledge += f"知识点：{knowledge_text}"
            else:
                # 兼容旧版：如果没有单独的knowledge字段，使用explanation
                old_explanation = correct_opt.get('explanation', '')
                old_explanation = self._remove_knowledge_prefix(old_explanation)
                physics_knowledge += f"知识点：{old_explanation}"
            
            nodes.append({
                "id": correct_result_id,
                "content": correct_opt.get("outcome_feedback", "") or correct_opt.get("result", "选择正确！"),
                "type": "ep_end",
                "metadata": {
                    "physics_knowledge": physics_knowledge,
                    "chapter_title": chapter_title,
                    "knowledge_point": knowledge_point,
                    "primary_knowledge_point": knowledge_point,
                    "supporting_knowledge_points": supporting_knowledge_points,
                    "knowledge_role_in_this_level": knowledge_role
                }
            })
            
            edges[correct_decision_id] = [correct_result_id]
            
            # 错误选项 -> branch_decision -> fatal
            branch_decision_ids = [correct_decision_id]
            
            for branch_idx, wrong_opt in enumerate(wrong_opts):
                # 错误决策节点
                branch_decision_id = f"{correct_decision_id}_branch_{branch_idx + 1}_decision"
                wrong_option_text = wrong_opt.get("option_text", "")
                # 优先使用 action_feedback 作为 decision 节点正文，兼容旧版 result
                wrong_transition = wrong_opt.get("action_feedback", "") or wrong_opt.get("result", wrong_option_text)

                nodes.append({
                    "id": branch_decision_id,
                    "content": wrong_transition,
                    "type": "decision",
                    "metadata": {
                        "choice_option": wrong_option_text
                    }
                })
                branch_decision_ids.append(branch_decision_id)
                
                # 错误结果节点 (fatal)
                branch_fatal_id = f"{correct_decision_id}_branch_{branch_idx + 1}_fatal"
                
                # 构建错误选项的物理知识说明
                wrong_explanation = wrong_opt.get('explanation', '')
                # 兼容旧版：如果有 analysis 和 knowledge 字段，合并使用
                if not wrong_explanation:
                    wrong_analysis = wrong_opt.get('analysis', '')
                    wrong_knowledge = wrong_opt.get('knowledge', '')
                    wrong_explanation = f"{wrong_analysis} {wrong_knowledge}".strip()
                
                # 正确答案的知识点文本（供LLM后处理参考）
                correct_knowledge_text = knowledge_text if knowledge_text else correct_opt.get('explanation', '')
                # 去掉知识点前缀，但保留"知识点："标签
                correct_knowledge_text = self._remove_knowledge_prefix(correct_knowledge_text)
                
                # 先用占位符构建，知识点部分将由LLM后处理生成适配版本
                wrong_header = knowledge_point or question.get("knowledge_point", "")
                if supporting_knowledge_points:
                    wrong_header += f"\n辅助知识点：{'、'.join(supporting_knowledge_points)}"

                wrong_physics = f"{wrong_header}\n\n"
                wrong_physics += f"错误原因：{wrong_explanation}\n\n"
                wrong_physics += f"知识点：{correct_knowledge_text}"
                
                nodes.append({
                    "id": branch_fatal_id,
                    "content": wrong_opt.get("outcome_feedback", "") or wrong_opt.get("result", "选择错误..."),
                    "type": "fatal",
                    "metadata": {
                        "explanation": wrong_opt.get("explanation", ""),
                        "physics_knowledge": wrong_physics,
                        "knowledge_point": knowledge_point,
                        "primary_knowledge_point": knowledge_point,
                        "supporting_knowledge_points": supporting_knowledge_points,
                        "knowledge_role_in_this_level": knowledge_role,
                        "wrong_explanation": wrong_explanation,
                        "correct_knowledge_ref": correct_knowledge_text
                    }
                })
                
                # 错误决策 -> fatal
                edges[branch_decision_id] = [branch_fatal_id]
                # fatal -> 返回决策点前的节点（与原项目一致）
                edges[branch_fatal_id] = [pre_decision_node_id]
            
            # 问题节点 -> 所有决策选项
            edges[question_node_id] = branch_decision_ids
            
            # 更新prev_node_id为正确结果节点
            prev_node_id = correct_result_id
        
        # 3. 故事结尾节点
        ending_id = self._next_node_id()
        nodes.append({
            "id": ending_id,
            "content": content.get("story_ending", "故事结束。"),
            "type": "ending",
            "metadata": {}
        })
        edges[prev_node_id] = [ending_id]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "start_node_id": "node_1",
            "ending_node_ids": [ending_id]
        }
    
    def convert_and_save(self, cot_result: Dict, output_path: str) -> str:
        """
        转换并保存为JSON文件
        
        Args:
            cot_result: COT生成器的输出
            output_path: 输出文件路径
            
        Returns:
            输出文件路径
        """
        story_graph = self.convert(cot_result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(story_graph, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def validate_conversion(self, story_graph: Dict) -> bool:
        """
        验证转换后的StoryGraph是否有效
        
        Args:
            story_graph: 转换后的StoryGraph字典
            
        Returns:
            是否有效
        """
        try:
            # 检查必要字段
            if not story_graph.get("nodes"):
                print("错误：没有节点")
                return False
            
            if not story_graph.get("start_node_id"):
                print("错误：没有起始节点")
                return False
            
            if not story_graph.get("ending_node_ids"):
                print("错误：没有结局节点")
                return False
            
            # 检查所有边的节点是否存在
            node_ids = {node["id"] for node in story_graph["nodes"]}
            
            for from_id, to_ids in story_graph.get("edges", {}).items():
                if from_id not in node_ids:
                    print(f"错误：边的源节点 {from_id} 不存在")
                    return False
                for to_id in to_ids:
                    if to_id not in node_ids:
                        print(f"错误：边的目标节点 {to_id} 不存在")
                        return False
            
            # 统计节点类型
            type_counts = {}
            for node in story_graph["nodes"]:
                node_type = node.get("type", "unknown")
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
            
            print(f"节点统计: {type_counts}")
            print(f"总节点数: {len(story_graph['nodes'])}")
            print(f"总边数: {sum(len(v) for v in story_graph.get('edges', {}).values())}")
            
            return True
            
        except Exception as e:
            print(f"验证失败: {e}")
            return False


def test_converter():
    """测试转换器"""
    # 模拟COT生成器的输出
    mock_cot_result = {
        "final_result": {
            "parsed": {
                "story_intro": "周末的早晨，你来到爷爷家帮忙修理老房子。",
                "questions": [
                    {
                        "question_id": 1,
                        "story_context": "你发现院子里有一块大石头挡住了去路。",
                        "question_text": "如何用最省力的方式移开这块石头？",
                        "knowledge_point": "杠杆原理",
                        "options": [
                            {
                                "option_id": "A",
                                "option_text": "将支点放在靠近石头的位置，使阻力臂短、动力臂长",
                                "is_correct": True,
                                "result": "你轻松地撬起了石头！",
                                "explanation": "根据杠杆原理，动力臂越长，所需的力越小。"
                            },
                            {
                                "option_id": "B",
                                "option_text": "将支点放在铁棍中间，使两边力臂相等",
                                "is_correct": False,
                                "result": "你费了很大力气才勉强撬动石头。",
                                "explanation": "力臂相等时没有省力效果。"
                            },
                            {
                                "option_id": "C",
                                "option_text": "将支点放在远离石头的位置，使阻力臂长",
                                "is_correct": False,
                                "result": "石头纹丝不动，你的力气不够。",
                                "explanation": "阻力臂长于动力臂时反而费力。"
                            }
                        ],
                        "transition_to_next": "石头移开后，你走进了老房子。"
                    }
                ],
                "story_ending": "在你的帮助下，爷爷的老房子焕然一新！"
            }
        }
    }
    
    converter = CoTToStoryGraphConverter()
    result = converter.convert(mock_cot_result)
    
    print("\n转换结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n验证结果:")
    converter.validate_conversion(result)


if __name__ == "__main__":
    test_converter()
