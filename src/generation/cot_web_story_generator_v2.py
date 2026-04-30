"""
CoT Web Story Generator V2 - 使用Gemini-2.5-flash
整合 RAG + 初步生成 + COT反思 + 搜索补充 + 改进生成

流程：
1. RAG检索 - 从FAISS知识库获取相关内容
2. 初步生成 - 生成初版题目/故事
3. COT反思 - 思考题目的缺点和不足
4. 搜索补充 - 针对每个缺点单独搜索补充内容
5. 改进生成 - 基于补充内容改进题目
"""

import os
import sys
import json
import re
import datetime
from typing import List, Dict, Optional
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

# 加载环境变量
load_dotenv()

# 添加src目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.faiss_retriever import FAISSRetriever
from utils.model_provider_config import get_text_model_config



class GeminiCoTGenerator:
    """基于Gemini-2.5-flash的CoT生成器"""

    def _make_json_serializable(self, data):
        if isinstance(data, dict):
            return {key: self._make_json_serializable(value) for key, value in data.items()}
        if isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        if isinstance(data, tuple):
            return [self._make_json_serializable(item) for item in data]
        if isinstance(data, np.generic):
            return data.item()
        if isinstance(data, np.ndarray):
            return data.tolist()
        return data

    def _load_json_if_exists(self, file_path: str) -> Optional[Dict]:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _save_json(self, file_path: str, data: Dict) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self._make_json_serializable(self._sanitize_llm_payload(data)), f, ensure_ascii=False, indent=2)

    def _truncate_text(self, text: str, limit: int = 220) -> str:
        if not text:
            return ""
        return text if len(text) <= limit else text[:limit] + "..."

    def _extract_level_knowledge_points(self, story_framework: Dict) -> List[str]:
        levels = story_framework.get("levels", [])
        return [
            lvl.get("primary_knowledge_point", "") or lvl.get("knowledge_point", "")
            for lvl in levels
            if lvl.get("primary_knowledge_point") or lvl.get("knowledge_point")
        ]

    def _build_soft_injection_guidance(self, choice_knowledge: Dict) -> str:
        raw = choice_knowledge.get("raw", {})
        misconceptions = raw.get("misconceptions", [])
        error_options = raw.get("error_options", [])

        guidance_parts = []
        if misconceptions:
            guidance_parts.append("优先可注入的误区槽位（能自然映射时再使用，若不贴合当前场景则不要强行套用）：")
            for idx, item in enumerate(misconceptions[:2], 1):
                text = self._truncate_text(item.get("text", ""), 220)
                guidance_parts.append(f"- 误区槽位{idx}：{text}")
        if error_options:
            guidance_parts.append("可参考的错误方案原型（仅在当前任务与其操作维度接近时借用其错误逻辑）：")
            for idx, item in enumerate(error_options[:2], 1):
                text = self._truncate_text(item.get("text", ""), 220)
                guidance_parts.append(f"- 错误原型{idx}：{text}")

        if not guidance_parts:
            return "当前没有可稳定注入的误区素材，请只基于场景任务自然设计两个真实学生会犯的错。"

        guidance_parts.append("如果这些误区与本题的操作骨架不一致，请保留其错误认知方向，但不要机械照抄表述。")
        return "\n".join(guidance_parts)

    def _strip_trailing_commas(self, json_str: str) -> str:
        return re.sub(r',\s*([}\]])', r'\1', json_str)

    def _sanitize_llm_text(self, text: str) -> str:
        if not isinstance(text, str):
            return text

        cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"!\[[^\]]*image[^\]]*\]\([^\)]*\)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[\s*image[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<image[^>]*>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*image\s*[:：].*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _sanitize_llm_payload(self, value):
        if isinstance(value, str):
            return self._sanitize_llm_text(value)
        if isinstance(value, list):
            return [self._sanitize_llm_payload(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_llm_payload(item) for item in value]
        if isinstance(value, dict):
            return {key: self._sanitize_llm_payload(item) for key, item in value.items()}
        return value

    def _deduplicate_preserve_order(self, items: List[str]) -> List[str]:
        deduplicated = []
        for item in items:
            if item and item not in deduplicated:
                deduplicated.append(item)
        return deduplicated

    def _normalize_knowledge_point_list(self, value) -> List[str]:
        if isinstance(value, str):
            normalized = value.replace("，", ",").replace("、", ",").replace("；", ",")
            parts = re.split(r"[,\n]+", normalized)
        elif isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, str):
                    parts.extend(re.split(r"[,\n]+", item.replace("，", ",").replace("、", ",").replace("；", ",")))
        else:
            parts = []

        cleaned_parts = []
        for part in parts:
            cleaned = self._sanitize_llm_text(str(part)).strip()
            cleaned = re.sub(r"^[-*\d\.\)\s]+", "", cleaned)
            if cleaned:
                cleaned_parts.append(cleaned)

        return self._deduplicate_preserve_order(cleaned_parts)

    def _normalize_story_framework(self,
                                   story_framework: Dict,
                                   candidate_knowledge_points: List[str],
                                   num_questions: int) -> Dict:
        levels = story_framework.get("levels", []) or []
        story_arc = story_framework.get("story_arc", {}) or {}
        arc_stages = story_arc.get("stages", []) or []
        fallback_candidates = [kp for kp in candidate_knowledge_points if kp]
        valid_candidate_set = set(fallback_candidates)
        fallback_default = fallback_candidates[0] if fallback_candidates else "未知"

        if num_questions <= 1:
            max_distinct_primary = 1
        elif num_questions <= 3:
            max_distinct_primary = max(2, num_questions - 1)
        else:
            max_distinct_primary = max(3, min(num_questions - 1, (num_questions * 3 + 3) // 4))

        explicit_selected = [
            kp for kp in self._normalize_knowledge_point_list(story_framework.get("selected_knowledge_points", []))
            if kp in valid_candidate_set
        ]
        derived_primary_candidates = []
        for level in levels:
            raw_primary = self._sanitize_llm_text(str(level.get("primary_knowledge_point") or level.get("knowledge_point") or "")).strip()
            if raw_primary in valid_candidate_set and raw_primary not in derived_primary_candidates:
                derived_primary_candidates.append(raw_primary)
            for support_kp in self._normalize_knowledge_point_list(
                level.get("supporting_knowledge_points", []) or level.get("secondary_knowledge_points", [])
            ):
                if support_kp in valid_candidate_set and support_kp not in derived_primary_candidates:
                    derived_primary_candidates.append(support_kp)

        primary_pool = explicit_selected or derived_primary_candidates or fallback_candidates[:]
        if max_distinct_primary and len(primary_pool) > max_distinct_primary:
            primary_pool = primary_pool[:max_distinct_primary]
        if not primary_pool and fallback_default != "未知":
            primary_pool = [fallback_default]

        raw_level_sources = []
        for i in range(num_questions):
            raw_level = levels[i] if i < len(levels) else {}
            arc_stage = arc_stages[i] if i < len(arc_stages) else {}
            merged_level = dict(arc_stage)
            merged_level.update(raw_level)
            raw_level_sources.append(merged_level)

        if not raw_level_sources and num_questions > 0:
            raw_level_sources = [{} for _ in range(num_questions)]

        planned_primary_sequence = []
        if primary_pool:
            for i in range(num_questions):
                mapped_index = min(int(i * len(primary_pool) / max(num_questions, 1)), len(primary_pool) - 1)
                planned_primary_sequence.append(primary_pool[mapped_index])

        normalized_levels = []
        primary_first_seen = {}
        for i, level in enumerate(raw_level_sources):
            raw_primary = self._sanitize_llm_text(
                str(level.get("primary_knowledge_point") or level.get("knowledge_point") or "")
            ).strip()
            if raw_primary not in valid_candidate_set:
                raw_primary = ""

            primary_kp = raw_primary or (planned_primary_sequence[i] if i < len(planned_primary_sequence) else fallback_default)
            if primary_kp not in valid_candidate_set and fallback_candidates:
                primary_kp = planned_primary_sequence[i] if i < len(planned_primary_sequence) else fallback_candidates[min(i, len(fallback_candidates) - 1)]
            if not primary_kp:
                primary_kp = fallback_default

            supporting_kps = []
            for support_kp in self._normalize_knowledge_point_list(
                level.get("supporting_knowledge_points", []) or level.get("secondary_knowledge_points", [])
            ):
                if support_kp in valid_candidate_set and support_kp != primary_kp and support_kp not in supporting_kps:
                    supporting_kps.append(support_kp)

            if i > 0:
                prev_primary = normalized_levels[-1].get("primary_knowledge_point", "")
                if prev_primary and prev_primary != primary_kp and prev_primary not in supporting_kps and len(supporting_kps) < 2:
                    supporting_kps.append(prev_primary)

            reused_from_level = primary_first_seen.get(primary_kp)
            if reused_from_level is None:
                primary_first_seen[primary_kp] = i + 1

            knowledge_role = level.get("knowledge_role_in_this_level", "") or level.get("knowledge_role", "")
            if not knowledge_role:
                if reused_from_level is not None:
                    knowledge_role = "主知识点复用"
                elif supporting_kps:
                    knowledge_role = "主知识点推进，辅知识点衔接"
                else:
                    knowledge_role = "主知识点推进"

            challenge_text = level.get("challenge", "") or level.get("core_task", "") or "解决当前最紧迫的实际问题。"
            core_task_text = level.get("core_task", "") or level.get("challenge", "") or challenge_text
            success_state = level.get("success_state", "") or level.get("exit_trigger", "") or "当前阶段问题被缓解，角色获得推进下一步所需的条件。"
            transition_reason = level.get("transition_reason", "") or level.get("bridge_reason", "") or "当前阶段的结果直接改变了局面，因此下一关顺势展开。"
            entry_trigger = level.get("entry_trigger", "") or (f"承接第{i}关的结果进入当前阶段。" if i > 0 else "故事主任务开始。")
            why_here = level.get("why_here", "")
            if not why_here:
                if reused_from_level is not None:
                    why_here = f"当前主线仍在推进与【{primary_kp}】相关的任务，因此继续复用这一主知识点。"
                else:
                    why_here = f"当前阶段最关键的判断依赖【{primary_kp}】，它最能支撑这一步任务推进。"

            normalized_levels.append({
                "level_number": i + 1,
                "knowledge_point": primary_kp,
                "primary_knowledge_point": primary_kp,
                "supporting_knowledge_points": supporting_kps[:2],
                "knowledge_role_in_this_level": knowledge_role,
                "knowledge_point_reused": reused_from_level is not None,
                "reuse_from_level": reused_from_level,
                "stage_label": level.get("stage_label", f"第{i + 1}阶段"),
                "state_before": level.get("state_before", "") or (arc_stages[i].get("state_before", "") if i < len(arc_stages) else ""),
                "scene_description": level.get("scene_description", "") or "请在当前场景中设计一个自然、具体的任务镜头。",
                "challenge": challenge_text,
                "core_task": core_task_text,
                "stakes": level.get("stakes", "") or "如果这一关失败，主线任务会明显受阻。",
                "micro_decision_focus": level.get("micro_decision_focus", "") or "让玩家在真实任务中做出关键判断。",
                "task_type": level.get("task_type", "") or "剧情推进中的关键判断",
                "variety_guard": level.get("variety_guard", "") or "避免与前一关重复同一种题目外壳。",
                "transition_hint": level.get("transition_hint", "") or transition_reason,
                "misconceptions": level.get("misconceptions", "") or f"围绕【{primary_kp}】在当前情境下的真实学生误判设计。",
                "why_here": why_here,
                "entry_trigger": entry_trigger,
                "exit_trigger": level.get("exit_trigger", "") or success_state,
                "success_state": success_state,
                "bridge_reason": level.get("bridge_reason", "") or transition_reason,
                "transition_reason": transition_reason,
                "continuity_focus": level.get("continuity_focus", "") or "让下一关来自当前结果状态，而不是另起炉灶。"
            })

        story_framework["levels"] = normalized_levels[:num_questions]
        primary_knowledge_points = self._deduplicate_preserve_order([
            lvl.get("primary_knowledge_point", "")
            for lvl in story_framework["levels"]
            if lvl.get("primary_knowledge_point")
        ])
        selected_knowledge_points = []
        for level in story_framework["levels"]:
            for kp in [level.get("primary_knowledge_point", "")] + level.get("supporting_knowledge_points", []):
                if kp and kp not in selected_knowledge_points:
                    selected_knowledge_points.append(kp)

        if not selected_knowledge_points and fallback_candidates:
            selected_knowledge_points = fallback_candidates[:min(max_distinct_primary, len(fallback_candidates))]
        if not primary_knowledge_points and selected_knowledge_points:
            primary_knowledge_points = selected_knowledge_points[:]

        story_framework["selected_knowledge_points"] = selected_knowledge_points
        story_framework["primary_knowledge_points"] = primary_knowledge_points
        story_framework["level_knowledge_sequence"] = [
            lvl.get("primary_knowledge_point", "")
            for lvl in story_framework["levels"]
            if lvl.get("primary_knowledge_point")
        ]
        story_framework["knowledge_point_to_level_count"] = dict(Counter(story_framework["level_knowledge_sequence"]))
        story_framework.setdefault("story_arc", story_arc)
        return story_framework

    def __init__(self,
                 faiss_index_path: str = "faiss_database/physics_knowledge.index",
                 faiss_docs_path: str = "faiss_database/physics_knowledge_docs.json",
                 gemini_api_key: Optional[str] = None,
                 search_result_dir: str = "res_from_search",
                 text_provider: Optional[str] = None):
        """
        初始化生成器
        
        Args:
            faiss_index_path: FAISS索引路径
            faiss_docs_path: FAISS文档路径
            gemini_api_key: Gemini API密钥
            search_result_dir: 搜索结果保存目录
        """
        # 初始化FAISS检索器
        self.retriever = FAISSRetriever(faiss_index_path, faiss_docs_path)
        
        # 初始化文本模型客户端（默认保持原 Gemini 代理通道）
        text_config = get_text_model_config(text_provider)
        
        # 检查API key是否为空，如果为空则回退到aigcbest
        if not text_config["api_key"] and text_config["provider"] != "aigcbest":
            print(f"⚠️ 警告: {text_config['provider']} 提供商的API key未配置，自动回退到 aigcbest")
            text_config = get_text_model_config("aigcbest")
        
        self.text_provider = text_config["provider"]
        self.gemini_api_key = gemini_api_key or text_config["api_key"]
        self.gemini_base_url = text_config["base_url"]
        self.model = text_config["model"]
        
        # 最终检查API key
        if not self.gemini_api_key:
            raise ValueError(f"❌ 错误: {self.text_provider} 提供商的API key未配置，请检查环境变量")
        
        self.client = OpenAI(
            api_key=self.gemini_api_key,
            base_url=self.gemini_base_url,
            timeout=300.0  # 5分钟超时
        )
        
        # 搜索结果保存目录
        self.search_result_dir = search_result_dir
        os.makedirs(self.search_result_dir, exist_ok=True)
        
        print(f"✓ GeminiCoTGenerator 初始化完成")
        print(f"  - 提供商: {self.text_provider}")
        print(f"  - 模型: {self.model}")
        print(f"  - 搜索结果目录: {self.search_result_dir}")
    
    def _call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 32000, max_retries: int = 3) -> str:
        """调用Gemini LLM，支持重试机制"""
        import time
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 递增等待时间：2秒、4秒
                    print(f"⚠️ LLM调用失败（第{attempt + 1}次尝试）: {e}")
                    print(f"⏳ {wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ LLM调用失败（已重试{max_retries}次）: {e}")
                    raise

    def select_knowledge_points_from_pool(self, 
                                          knowledge_pool: List[Dict], 
                                          scenario: str, 
                                          num_questions: int = 10,
                                          min_count: Optional[int] = None) -> List[str]:
        """
        从知识点池中根据场景自主选择合适的知识点
        
        Args:
            knowledge_pool: 知识点池，每个知识点包含id、name、core_concept等字段
            scenario: 故事场景描述
            num_questions: 需要选择的知识点数量
            
        Returns:
            选择的知识点名称列表
        """
        target_count = min(max(1, num_questions), len(knowledge_pool))
        if min_count is None:
            min_count = max(4, min(target_count, max(3, target_count - max(2, target_count // 3))))
        min_count = min(target_count, max(1, min_count))

        print(f"\n{'='*70}")
        print(f"从知识点池中自主选择知识点")
        print(f"场景: {scenario}")
        print(f"候选上限: {target_count} 个知识点")
        print(f"候选下限: {min_count} 个知识点")
        print(f"知识点池大小: {len(knowledge_pool)} 个")
        print(f"{'='*70}")
        
        # 构建知识点池摘要（避免token过多）
        knowledge_summary = []
        for kp in knowledge_pool:
            knowledge_summary.append(f"- {kp.get('name', kp.get('id', ''))}")
        
        # 如果知识点池太大，只取前200个用于展示
        if len(knowledge_summary) > 200:
            knowledge_summary = knowledge_summary[:200]
            knowledge_summary.append(f"... (还有 {len(knowledge_pool) - 200} 个知识点)")
        
        prompt = f"""你是一位中学教育专家，需要根据给定的故事场景，从知识点池（包含物理、生物、化学、地理等学科）中选择最合适的知识点。

**故事场景**
{scenario}

**知识点池**（共 {len(knowledge_pool)} 个）
{chr(10).join(knowledge_summary)}

**任务要求**
请从上面的知识点池中选择最符合该场景的 {min_count} 到 {target_count} 个候选知识点，作为后续故事规划的素材池。

**重要约束**
1. 必须从上面的知识点池列表中选择，不能创造新的知识点名称
2. 知识点名称必须完全匹配，不能缩写或修改
3. 请优先少而精，只输出真正可能在该主线里自然使用的候选知识点；不要为了凑数硬塞边缘知识点
4. 输出数量必须在 {min_count} 到 {target_count} 之间；如果场景很集中，宁可接近下限，也不要为了接近上限而降低相关性
5. 这里只做候选池选择，不要按剧情顺序排序，不要试图提前规划每一关

**选择标准**
1. 知识点要与场景自然融合，能够在该场景中实际应用
2. 候选知识点之间可以来自不同学科，但都要服务于同一场景主线
3. 根据场景特点，选择最相关的学科知识点（如野外场景可选择地理、生物、生存技能等）
4. 避免选择过于抽象或与场景完全不相关的知识点

**输出格式**
请只返回选中的知识点名称，用逗号分隔，不要添加任何其他文字或解释。

例如：
压强与浮力,水的组成与净化,皮肤与体温调节,地图阅读与方向判断

**开始选择**
"""

        try:
            print(f"\n🤖 调用LLM选择知识点...")
            selected_kp_names = self._call_llm(prompt, temperature=0.2, max_tokens=10000)
            
            # 解析选择结果
            selected_kp_list = self._normalize_knowledge_point_list(selected_kp_names)
            
            # 验证选择的知识点是否在池中
            valid_kp_names = [kp.get('name', kp.get('id', '')) for kp in knowledge_pool]
            validated_kp_list = []
            for kp_name in selected_kp_list:
                if kp_name in valid_kp_names:
                    if kp_name not in validated_kp_list:
                        validated_kp_list.append(kp_name)
                else:
                    print(f"⚠️ 知识点 '{kp_name}' 不在池中，已跳过")
            
            # 如果验证后数量过少，只补充到下限，避免为了凑满候选上限而降低相关性
            min_required = min(min_count, len(knowledge_pool))
            if len(validated_kp_list) < min_required:
                print(f"⚠️ LLM选择的知识点数量不足 ({len(validated_kp_list)}/{min_required})，从池中补充")
                remaining_pool = [kp for kp in knowledge_pool if kp.get('name', kp.get('id', '')) not in validated_kp_list]
                needed = min_required - len(validated_kp_list)
                for i in range(min(needed, len(remaining_pool))):
                    validated_kp_list.append(remaining_pool[i].get('name', remaining_pool[i].get('id', '')))
            
            print(f"✅ 最终选择的知识点: {', '.join(validated_kp_list)}")
            return validated_kp_list[:min(target_count, len(validated_kp_list))]
            
        except Exception as e:
            print(f"❌ LLM选择知识点失败: {e}")
            print(f"🔄 回退到随机选择 {target_count} 个候选知识点")
            # 回退策略：随机选择
            import random
            selected = random.sample(knowledge_pool, min(target_count, len(knowledge_pool)))
            return [kp.get('name', kp.get('id', '')) for kp in selected]
    
    def _save_search_result(self, query: str, results: Dict, category: str):
        """保存搜索结果到文件"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{category}_{timestamp}.json"
        filepath = os.path.join(self.search_result_dir, filename)
        
        data = {
            "query": query,
            "category": category,
            "timestamp": timestamp,
            "results": results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ 搜索结果已保存: {filename}")
        return filepath
    
    def _search_single_issue(self, issue: str, knowledge_point: str) -> Dict:
        """针对单个问题进行RAG补充检索（使用本地FAISS知识库，无需外部网络）"""
        query = f"{knowledge_point} {issue} 学生误区 易错点 教学建议"
        print(f"\n  🔍 RAG补充检索: {query[:80]}")

        raw_results = self.retriever.search(query, top_k=5)
        formatted = [
            {
                'title': r.get('knowledge_name', knowledge_point),
                'url': 'faiss_local',
                'content': r.get('text', ''),
                'score': float(r.get('score', 0))
            }
            for r in raw_results
        ]
        print(f"  ✓ 检索到 {len(formatted)} 条相关内容")
        return {'query': query, 'results': formatted}

    # ============================================================
    # 核心生成流程
    # ============================================================
    
    def generate_interactive_story(self,
                                   knowledge_points: List[str],
                                   scenario: str = "日常生活场景",
                                   num_questions: int = 4,
                                   checkpoint_dir: Optional[str] = None,
                                   enable_resume: bool = True) -> Dict:
        """
        生成交互式故事（包含多个题目）- 分步生成策略
        
        完整流程：
        1. RAG检索知识
        2. 生成故事背景和整体框架
        3. 逐关生成每个题目（确保每关质量）
        4. COT反思和改进
        
        Args:
            knowledge_points: 知识点列表
            scenario: 故事场景
            num_questions: 题目数量（默认4个）
            
        Returns:
            Dict: 包含故事、题目、思考过程等
        """
        print(f"\n{'='*70}")
        print(f"开始生成交互式故事（分步生成模式）")
        print(f"知识点候选: {', '.join(knowledge_points)}")
        print(f"场景: {scenario}")
        print(f"题目数量: {num_questions}")
        print(f"{'='*70}")
        
        checkpoints_dir = None
        if checkpoint_dir:
            checkpoints_dir = os.path.join(checkpoint_dir, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)

        rag_file = os.path.join(checkpoints_dir, "step1_rag_knowledge.json") if checkpoints_dir else None
        story_arc_file = os.path.join(checkpoints_dir, "step2_story_arc.json") if checkpoints_dir else None
        framework_file = os.path.join(checkpoints_dir, "step3_story_framework.json") if checkpoints_dir else None
        questions_file = os.path.join(checkpoints_dir, "step4_questions.json") if checkpoints_dir else None
        initial_file = os.path.join(checkpoints_dir, "step4_initial_result.json") if checkpoints_dir else None
        reflection_file = os.path.join(checkpoints_dir, "step5_reflection.json") if checkpoints_dir else None
        supplements_file = os.path.join(checkpoints_dir, "step6_search_supplements.json") if checkpoints_dir else None
        improved_file = os.path.join(checkpoints_dir, "step7_improved_result.json") if checkpoints_dir else None

        # ========== 步骤1: RAG检索 ==========
        print(f"\n【步骤1】RAG检索相关知识...")
        rag_knowledge = self._load_json_if_exists(rag_file) if enable_resume and rag_file else None
        if rag_knowledge is not None:
            print("  🚀 检测到已有RAG结果，直接复用")
        else:
            rag_knowledge = self._retrieve_knowledge(knowledge_points)
            if rag_file:
                self._save_json(rag_file, rag_knowledge)
        
        # ========== 步骤2: 先规划连续剧情任务链 ==========
        print(f"\n【步骤2】先规划连续剧情任务链...")
        story_arc = self._load_json_if_exists(story_arc_file) if enable_resume and story_arc_file else None
        if story_arc is not None:
            print("  🚀 检测到已有剧情任务链，直接复用")
        else:
            story_arc = self._generate_story_arc(
                scenario, num_questions, rag_knowledge
            )
            if story_arc_file:
                self._save_json(story_arc_file, story_arc)

        # ========== 步骤3: 为剧情节点匹配知识点并生成框架 ==========
        print(f"\n【步骤3】为剧情节点匹配知识点并生成整体框架...")
        story_framework = self._load_json_if_exists(framework_file) if enable_resume and framework_file else None
        if story_framework is not None:
            print("  🚀 检测到已有故事框架，直接复用")
            story_framework.setdefault("story_arc", story_arc)
        else:
            story_framework = self._generate_story_framework(
                knowledge_points, scenario, num_questions, rag_knowledge, story_arc
            )
        story_framework["story_arc"] = story_arc
        story_framework = self._normalize_story_framework(story_framework, knowledge_points, num_questions)
        if framework_file:
            self._save_json(framework_file, story_framework)

        level_knowledge_points = self._extract_level_knowledge_points(story_framework)
        selected_knowledge_points = story_framework.get("selected_knowledge_points", []) or self._deduplicate_preserve_order(level_knowledge_points)
        if level_knowledge_points:
            print(f"  ✓ 规划后的关卡主知识点顺序: {', '.join(level_knowledge_points)}")
        else:
            level_knowledge_points = knowledge_points[:num_questions]
        if selected_knowledge_points:
            print(f"  ✓ 最终采用的知识点集合: {', '.join(selected_knowledge_points)}")
        
        # ========== 步骤4: 逐关生成题目 ==========
        print(f"\n【步骤4】逐关生成题目...")
        questions = []
        existing_questions_payload = self._load_json_if_exists(questions_file) if enable_resume and questions_file else None
        if existing_questions_payload:
            questions = existing_questions_payload.get("questions", [])
            print(f"  🚀 检测到已有 {len(questions)} 关题目，将从断点继续")

        for i in range(len(questions), num_questions):
            print(f"\n  --- 生成第 {i+1}/{num_questions} 关 ---")
            question = self._generate_single_question(
                i, level_knowledge_points, story_framework, rag_knowledge, questions
            )
            questions.append(question)
            if questions_file:
                self._save_json(questions_file, {
                    "knowledge_points": selected_knowledge_points,
                    "level_knowledge_points": level_knowledge_points,
                    "candidate_knowledge_points": knowledge_points,
                    "scenario": scenario,
                    "num_questions": num_questions,
                    "questions": questions
                })
            print(f"  ✓ 第{i+1}关生成完成: {question.get('chapter_title', '未命名')}")
        
        # 组装初步结果
        initial_result = {
            "raw_response": "",
            "parsed": {
                "story_intro": story_framework.get("story_intro", ""),
                "questions": questions,
                "story_ending": story_framework.get("story_ending", "")
            }
        }
        if initial_file:
            self._save_json(initial_file, initial_result)
        
        # ========== 步骤5: COT反思 ==========
        print(f"\n【步骤5】COT反思 - 分析题目缺点...")
        reflection = self._load_json_if_exists(reflection_file) if enable_resume and reflection_file else None
        if reflection is not None:
            print("  🚀 检测到已有反思结果，直接复用")
        else:
            reflection = self._reflect_on_questions(initial_result, story_framework)
            if reflection_file:
                self._save_json(reflection_file, reflection)
        
        # ========== 步骤6: 搜索补充 ==========
        print(f"\n【步骤6】针对缺点进行搜索补充...")
        search_supplements = self._load_json_if_exists(supplements_file) if enable_resume and supplements_file else None
        if search_supplements is not None:
            print("  🚀 检测到已有搜索补充结果，直接复用")
        else:
            search_supplements = self._search_for_improvements(
                reflection, level_knowledge_points
            )
            if supplements_file:
                self._save_json(supplements_file, search_supplements)
        
        # ========== 步骤7: 改进生成 ==========
        print(f"\n【步骤7】基于补充内容改进生成...")
        improved_result = self._load_json_if_exists(improved_file) if enable_resume and improved_file else None
        if improved_result is not None:
            print("  🚀 检测到已有改进结果，直接复用")
        else:
            improved_result = self._generate_improved(
                initial_result, reflection, search_supplements, story_framework, level_knowledge_points
            )
            improved_parsed = improved_result.get("parsed", {}) if isinstance(improved_result, dict) else {}
            if not improved_parsed or not improved_parsed.get("questions"):
                print("  ⚠ 改进结果为空或缺少题目，自动回退到初始结果，避免生成损坏的 final_result")
                improved_result = {
                    "raw_response": improved_result.get("raw_response", "") if isinstance(improved_result, dict) else "",
                    "parsed": self._normalize_option_feedbacks(
                        json.loads(json.dumps(initial_result.get("parsed", {}), ensure_ascii=False))
                    )
                }
            if improved_file:
                self._save_json(improved_file, improved_result)
        
        print(f"\n{'='*70}")
        print(f"✓ 生成完成")
        print(f"{'='*70}")
        
        return {
            "knowledge_points": selected_knowledge_points,
            "level_knowledge_points": level_knowledge_points,
            "candidate_knowledge_points": knowledge_points,
            "scenario": scenario,
            "rag_knowledge": rag_knowledge,
            "story_arc": story_arc,
            "story_framework": story_framework,
            "initial_result": initial_result,
            "reflection": reflection,
            "search_supplements": search_supplements,
            "final_result": improved_result
        }

    def _normalize_story_arc(self,
                             story_arc: Dict,
                             scenario: str,
                             num_questions: int) -> Dict:
        stages = story_arc.get("stages", []) or []
        normalized_stages = []

        for i, stage in enumerate(stages[:num_questions]):
            normalized_stages.append({
                "level_number": i + 1,
                "stage_label": stage.get("stage_label", f"第{i + 1}阶段"),
                "state_before": stage.get("state_before", ""),
                "scene_description": stage.get("scene_description", ""),
                "core_task": stage.get("core_task", ""),
                "stakes": stage.get("stakes", ""),
                "success_state": stage.get("success_state", ""),
                "transition_reason": stage.get("transition_reason", ""),
                "continuity_focus": stage.get("continuity_focus", ""),
                "micro_decision_focus": stage.get("micro_decision_focus", ""),
                "variety_guard": stage.get("variety_guard", "")
            })

        while len(normalized_stages) < num_questions:
            prev_stage = normalized_stages[-1] if normalized_stages else {
                "success_state": "故事刚开始，角色还在摸索第一步。",
                "transition_reason": "开篇任务自然出现。"
            }
            idx = len(normalized_stages) + 1
            normalized_stages.append({
                "level_number": idx,
                "stage_label": f"第{idx}阶段",
                "state_before": prev_stage.get("success_state", "上一阶段刚结束，角色进入新的局面。"),
                "scene_description": "延续上一阶段结果，推进同一主线任务。",
                "core_task": "解决当前阶段最关键、最现实的阻塞问题。",
                "stakes": "如果这一关判断失误，主线任务会被拖慢或方向走偏。",
                "success_state": "当前阶段的问题被缓解，你获得进入下一阶段所需的条件。",
                "transition_reason": "这一关的结果直接改变了局面，因此顺势进入下一阶段。",
                "continuity_focus": "让下一关来自当前结果状态，而不是另起炉灶。",
                "micro_decision_focus": "设计一个必须调用知识判断的真实微决策。",
                "variety_guard": "避免与前一关重复同一种微决策外壳。"
            })

        story_arc["main_goal"] = story_arc.get("main_goal", scenario)
        story_arc["story_intro"] = story_arc.get("story_intro", f"你进入了这样的情境：{scenario}")
        story_arc["story_ending"] = story_arc.get("story_ending", "经历一连串判断与选择后，你完成了这段教育冒险。")
        story_arc["stages"] = normalized_stages[:num_questions]
        return story_arc

    def _generate_story_arc(self,
                            scenario: str,
                            num_questions: int,
                            rag_knowledge: Dict) -> Dict:
        prompt = f"""你是一位教育叙事设计专家。请先不要选择知识点，而是先为一个教育互动故事规划连续的剧情任务链。

**故事大方向**
{scenario}

**知识库参考**
{rag_knowledge['summary']}

**你的目标**
请先规划一个包含 {num_questions} 关的连续故事，让它像一个真正推进中的冒险任务，而不是若干知识题拼接在一起。

**核心原则**
1. 先规划主线任务与状态变化，再考虑每关会需要哪类知识
2. 每一关都必须来自上一关的结果状态，不能靠旁白硬切场景
3. 每一关都要有一个现实、具体、会阻碍主线推进的局部任务
4. 每一关都应孕育一个“真实微决策点”，但不要把它直接写成考试题
5. 为了避免重复感，不要让多关都套用同一种判断外壳；但如果剧情自然需要，也允许局部呼应

**输出格式**（严格JSON）
```json
{{
  "main_goal": "贯穿全篇的主线目标",
  "story_intro": "故事开头（100-180字）",
  "stages": [
    {{
      "level_number": 1,
      "stage_label": "这一阶段的短标题",
      "state_before": "进入本关前角色的状态、资源、限制",
      "scene_description": "这一关的场景镜头",
      "core_task": "这一关必须解决的现实任务",
      "stakes": "如果判断失误，会带来什么具体阻碍",
      "success_state": "这一关成功后局面会怎么改变",
      "transition_reason": "为什么这个成功状态自然引出下一关",
      "continuity_focus": "本关与下一关要保持哪种连续性",
      "micro_decision_focus": "这一关最适合孕育哪种微决策，不要写成题干",
      "variety_guard": "为了避免和前后关重复，这一关要避开什么题目外壳"
    }}
  ],
  "story_ending": "故事结尾（50-100字）"
}}
```

请只输出JSON：
"""

        response = self._call_llm(prompt, temperature=0.7)
        parsed = self._parse_json_response(response)
        parsed = self._normalize_story_arc(parsed, scenario, num_questions)
        print(f"  ✓ 剧情任务链规划完成，包含 {len(parsed.get('stages', []))} 关")
        return parsed

    def _build_story_framework_summary(self, story_framework: Dict) -> str:
        levels = story_framework.get("levels", [])
        if not levels:
            return "暂无故事框架摘要。"

        summary_parts = []
        for level in levels:
            primary_kp = level.get('primary_knowledge_point', '') or level.get('knowledge_point', '')
            supporting_kps = ", ".join(level.get('supporting_knowledge_points', []))
            knowledge_display = primary_kp
            if supporting_kps:
                knowledge_display += f"（辅：{supporting_kps}）"
            summary_parts.append(
                f"第{level.get('level_number', '?')}关｜阶段：{level.get('stage_label', '')}｜"
                f"进入前状态：{self._truncate_text(level.get('state_before', ''), 120)}｜"
                f"当前任务：{self._truncate_text(level.get('core_task', '') or level.get('challenge', ''), 120)}｜"
                f"知识点：{knowledge_display}｜"
                f"成功后：{self._truncate_text(level.get('success_state', '') or level.get('exit_trigger', ''), 120)}｜"
                f"过渡原因：{self._truncate_text(level.get('transition_reason', '') or level.get('bridge_reason', ''), 120)}"
            )
        return "\n".join(summary_parts)

    def _generate_story_framework(self,
                                   knowledge_points: List[str],
                                   scenario: str,
                                   num_questions: int,
                                   rag_knowledge: Dict,
                                   story_arc: Optional[Dict] = None) -> Dict:
        """先基于剧情任务链，再为各阶段匹配知识点与教学切入点。"""
        story_arc = story_arc or self._generate_story_arc(scenario, num_questions, rag_knowledge)
        bridge_summary_parts = []
        for bridge in rag_knowledge.get('bridges', []):
            bridge_summary_parts.append(
                f"- 从【{bridge['from_knowledge_point']}】到【{bridge['to_knowledge_point']}】的过渡参考"
            )
            for case in bridge.get('from_story_cases', [])[:1]:
                bridge_summary_parts.append(f"  前一关可参考案例: {case['text'][:800]}")
            for case in bridge.get('to_story_cases', [])[:1]:
                bridge_summary_parts.append(f"  后一关可参考案例: {case['text'][:800]}")
        bridge_summary = "\n".join(bridge_summary_parts) if bridge_summary_parts else "无单独桥接案例，请自行设计自然过渡。"

        stage_summary_parts = []
        for stage in story_arc.get("stages", []):
            stage_summary_parts.append(
                f"第{stage.get('level_number', '?')}关：{stage.get('stage_label', '')}\n"
                f"- 进入前状态：{stage.get('state_before', '')}\n"
                f"- 当前任务：{stage.get('core_task', '')}\n"
                f"- 场景：{stage.get('scene_description', '')}\n"
                f"- 风险：{stage.get('stakes', '')}\n"
                f"- 成功后：{stage.get('success_state', '')}\n"
                f"- 过渡原因：{stage.get('transition_reason', '')}\n"
                f"- 微决策焦点：{stage.get('micro_decision_focus', '')}\n"
                f"- 去重复提醒：{stage.get('variety_guard', '')}"
            )
        stage_summary = "\n\n".join(stage_summary_parts)

        prompt = f"""你是一位教育专家，需要在已经规划好的剧情任务链上，为每一关匹配最自然、最有教育意义的知识点，并形成完整故事框架。

**候选知识点列表**（这是素材池，不要求全部使用，可以只选最适合的一部分，也允许某个知识点在多关中复用）
{', '.join(knowledge_points)}

**故事大方向**
{scenario}

**已经确定的剧情任务链**
主线目标：{story_arc.get('main_goal', scenario)}
故事开头：{story_arc.get('story_intro', '')}

{stage_summary}

**知识库参考**
{rag_knowledge['summary']}

**相邻知识点桥接参考**
{bridge_summary}

**任务要求**
请设计一个包含 {num_questions} 关的故事框架：
1. 不要重新发明剧情主线，而是沿用上面的剧情任务链
2. 为每一关匹配最适合的知识点；如果更自然，可以复用同一知识点，而不是强行每关换一个新点
3. 最终采用的不同知识点数量以“自然与有效”为准，不需要为了覆盖率硬凑数量
4. 每一关都要有一个 `primary_knowledge_point` 作为主知识点，必要时可增加 `supporting_knowledge_points`
5. 如果上一关与这一关本质上仍在解决同一类问题，可以直接复用同一个主知识点，不要为了形式变化而硬换知识点
6. 每一关都要说明：为什么此处是这个知识点最合适，而不是别的知识点
7. 每一关都要保留当前阶段的微决策焦点，为后续出题做准备
8. 故事开头和结尾应与剧情任务链一致

**重要原则**
- 场景要自然连贯，前一关的结果自然引出下一关
- 每关的挑战要符合生活实际，让学生有代入感
- 知识点要自然融入情节，不要生硬插入
- 使用第二人称视角（"你"）
- 优先利用上面的桥接参考，让相邻知识点之间形成明确的因果链条或任务推进链条
- 如果不同知识点之间连接牵强，则宁可减少不同知识点数量，改为让更贴合的知识点在不同阶段重复应用
- `transition_hint` 必须具体写清楚：上一关发生了什么，为什么自然引出下一关
- `selected_knowledge_points` 必须是最终真正采用的知识点集合，不得机械复制全部候选项
- 如果同一知识点跨两关或多关复用，请在 `knowledge_role_in_this_level` 或 `reuse_from_level` 中明确体现其复用关系
- `task_type` 只是当前阶段微决策形态的简洁标签，可以自由命名，不要机械套模板
- `variety_guard` 必须显式指出：这一关为了避免和前后关重复，出题时要注意什么

**输出格式**（严格按照JSON格式）：
```json
{{
  "selected_knowledge_points": ["最终真正采用的知识点"],
  "selection_rationale": ["为什么选这些知识点、为什么这样排序"],
  "main_goal": "贯穿故事的主任务",
  "story_intro": "故事开头（100-200字，设置场景和背景）",
  "levels": [
    {{
      "level_number": 1,
      "stage_label": "沿用剧情任务链的阶段名",
      "knowledge_point": "兼容旧字段：与 primary_knowledge_point 保持一致",
      "primary_knowledge_point": "这一关真正负责决策的主知识点",
      "supporting_knowledge_points": ["如有需要，可列出1-2个辅助知识点"],
      "knowledge_role_in_this_level": "如：主知识点推进 / 主知识点复用 / 主知识点推进，辅知识点衔接",
      "reuse_from_level": 1,
      "why_here": "为什么这一步使用这个知识点，而不是别的知识点",
      "entry_trigger": "这一关被上一关如何自然引出",
      "state_before": "进入本关前角色的状态",
      "scene_description": "这一关的场景设定（50-100字）",
      "challenge": "主角面临的具体挑战",
      "core_task": "这一关必须解决的现实任务",
      "stakes": "如果判断失误会带来的阻碍",
      "task_type": "当前阶段微决策形态的标签，自由命名",
      "micro_decision_focus": "本关最适合转化成哪种微决策焦点",
      "variety_guard": "为避免与前后关重复，本关出题时要避开什么",
      "success_state": "这一关成功后留下了什么新状态",
      "exit_trigger": "兼容旧字段：与success_state语义一致",
      "bridge_reason": "这一关为什么会自然过渡到下一关",
      "transition_reason": "为什么这个结果会自然引出下一关",
      "continuity_focus": "这一关与下一关要保持哪种连续性",
      "transition_hint": "如何过渡到下一关",
      "misconceptions": "可以设计的易错选项，每个易错选项要对应一个易错点，易错点要符合教学实际，让学生有代入感"
    }}
  ],
  "story_ending": "故事结尾（50-100字）"
}}
```

请只输出JSON：
"""

        response = self._call_llm(prompt, temperature=0.7)
        parsed = self._parse_json_response(response)
        parsed["main_goal"] = parsed.get("main_goal", story_arc.get("main_goal", scenario))
        parsed["story_intro"] = parsed.get("story_intro", story_arc.get("story_intro", ""))
        parsed["story_ending"] = parsed.get("story_ending", story_arc.get("story_ending", ""))
        parsed["story_arc"] = story_arc
        parsed = self._normalize_story_framework(parsed, knowledge_points, num_questions)

        print(f"  ✓ 故事框架生成完成，包含 {len(parsed.get('levels', []))} 关")

        return parsed

    def _plan_question_option_blueprint(self,
                                        level_index: int,
                                        current_level: Dict,
                                        story_framework: Dict,
                                        previous_questions: List[Dict],
                                        choice_knowledge: Dict) -> Dict:
        knowledge_point = current_level.get('primary_knowledge_point', '') or current_level.get('knowledge_point', '未知')
        supporting_knowledge_points = current_level.get('supporting_knowledge_points', []) or []
        recent_pattern_summary = self._summarize_recent_task_patterns(previous_questions)
        soft_injection_guidance = self._build_soft_injection_guidance(choice_knowledge)

        prompt = f"""你是一位教育专家。请先不要直接写最终题目，而是先为第 {level_index + 1} 关设计“微决策题目骨架”。

**故事背景**
{story_framework.get('story_intro', '')}

**当前关卡设定**
- 阶段名：{current_level.get('stage_label', '')}
- 进入前状态：{current_level.get('state_before', '')}
- 主知识点：{knowledge_point}
- 辅助知识点：{', '.join(supporting_knowledge_points) if supporting_knowledge_points else '无'}
- 知识点角色：{current_level.get('knowledge_role_in_this_level', '')}
- 场景：{current_level.get('scene_description', '')}
- 当前任务：{current_level.get('core_task', '') or current_level.get('challenge', '')}
- 风险：{current_level.get('stakes', '')}
- 易错选项：{current_level.get('misconceptions', '')}
- 本关为什么使用这个知识点：{current_level.get('why_here', '')}
- 进入本关的过渡提示：{current_level.get('entry_trigger', '（第一关，无前置过渡）') if level_index == 0 else current_level.get('entry_trigger', '')}
- 本关成功后状态：{current_level.get('success_state', '') or current_level.get('exit_trigger', '')}
- 本关结束后的过渡提示：{current_level.get('transition_reason', '') or current_level.get('transition_hint', '')}
- 下一关入口参考：{current_level.get('entry_trigger', '') if level_index < len(story_framework.get('levels', [])) - 1 else '这是最后一关'}
- 本关的微决策焦点：{current_level.get('micro_decision_focus', '')}
- 去重复提醒：{current_level.get('variety_guard', '')}

**前面的关卡**（用于保持连贯性）
{previous_questions}

**最近几关已使用的任务外壳**
{recent_pattern_summary}

**知识库参考**
{choice_knowledge['summary']}

**本题骨架规划（必须遵守，不要偏离到别的操作维度）**
请设计一个“知识辨析型”的微决策题目骨架：故事任务可以来自当前场景，但题干最终必须收束成一个非常具体、可枚举的知识槽位，让学生主要依靠知识点本身来选择，而不是依靠地名、特殊地标或剧情细节来猜测。
三个选项必须描述同一件事、同一种操作或同一种判断，只在一个唯一知识槽位上形成差异；这个槽位要尽量具体，如“滤层材料顺序”“支点位置”“地图定向依据”“撤离方向”“概念归类结果”，不要停留在“怎么做”这种泛泛层面。
错项必须优先来自该槽位上的真实学生常见误区，具有教学辨析价值；如果某个错项虽然和场景有关，但不属于同一槽位上的典型误判、没有知识价值，就不要使用。
骨架必须让后续 `question_text` 直接询问这个唯一槽位，让 `option_text` 只呈现该槽位的不同取值；不要在选项里继续扩写剧情，不要叠加第二个知识维度，不要写完整因果链，不要堆地名、具体地标、专有名词和无关物件细节。

**输出格式**（严格JSON）
```json
{{
  "task_type": "当前关的微决策形态标签，自由命名，不要机械套模板",
  "real_world_goal": "这一关要解决的现实任务",
  "knowledge_mechanism": "该知识点在这道题中真正起作用的机制",
  "shared_action_template": "三个选项共享的同一操作/同一判断框架，必须固定为同一知识槽位，不是三个不同策略",
  "option_sentence_pattern": "三个选项共享的表达骨架，优先写成同一槽位的不同取值，如不同顺序/位置/方向/分类结果",
  "variable_dimension": "唯一变化的是哪个具体知识槽位（如滤层顺序、支点位置、撤离方向、概念归类结果）",
  "correct_principle": "正确项抓住的关键原理",
  "correct_option_appeal": "如果不看答案标签，学生为什么也会认真考虑正确项",
  "wrong_slot_b": {{
    "error_source": "优先填误区槽位1；若不适合则填 natural_error",
    "misconception_type": "这一错项属于哪类学生误判",
    "student_appeal": "学生为什么会被这个错项说服",
    "wrong_logic": "错项B的错误逻辑"
  }},
  "wrong_slot_c": {{
    "error_source": "优先填误区槽位2；若不适合则填 natural_error",
    "misconception_type": "这一错项属于哪类学生误判",
    "student_appeal": "学生为什么会被这个错项说服",
    "wrong_logic": "错项C的错误逻辑"
  }},
  "option_plausibility_guard": "如何保证三个选项都是有教学价值的知识辨析项，而不是靠场景细节硬凑出来的方案",
  "story_context_focus": "本关情节要突出什么，才能自然衔接",
  "transition_focus": "本关结束时留下什么状态，引出下一关",
  "diversity_note": "本关如何避免和前几关重复"
}}
```

请只输出JSON：
"""

        response = self._call_llm(prompt, temperature=0.4)
        return self._parse_json_response(response)

    def _generate_single_question(self,
                                  level_index: int,
                                  knowledge_points: List[str],
                                  story_framework: Dict,
                                  rag_knowledge: Dict,
                                  previous_questions: List[Dict]) -> Dict:
        """
        第二步：逐关生成每个题目
        基于故事框架和前面的题目，生成当前关卡的完整内容
        """
        levels = story_framework.get('levels', [])
        current_level = levels[level_index] if level_index < len(levels) else {}
        primary_knowledge_point = current_level.get('primary_knowledge_point', '') or current_level.get('knowledge_point', knowledge_points[level_index] if level_index < len(knowledge_points) else '未知')
        supporting_knowledge_points = current_level.get('supporting_knowledge_points', []) or []
        knowledge_role = current_level.get('knowledge_role_in_this_level', '')
        choice_knowledge = self._retrieve_misconceptions_and_errors(
            primary_knowledge_point,
            current_level.get('scene_description', '')
        )
        teaching_advice = self._retrieve_teaching_advice(primary_knowledge_point)
        option_blueprint = self._plan_question_option_blueprint(
            level_index,
            current_level,
            story_framework,
            previous_questions,
            choice_knowledge
        )

        previous_context = ""
        if previous_questions:
            previous_context = "\n".join([
                f"第{i+1}关：情节—{q.get('story_context', '')[:220]}；问题—{q.get('question_text', '')[:120]}；结尾过渡—{q.get('transition_to_next', '（无）')}"
                for i, q in enumerate(previous_questions)
            ])

        framework_overview = self._build_story_framework_summary(story_framework)
        recent_pattern_summary = self._summarize_recent_task_patterns(previous_questions)
        next_level = levels[level_index + 1] if level_index + 1 < len(levels) else {}

        prompt = f"""你是一位教育专家，需要为交互式故事生成第 {level_index + 1} 关的完整交互内容。

**故事背景**
{story_framework.get('story_intro', '')}

**整体关卡框架（全局概览，请据此保证本关与前后关的连贯性）**
{framework_overview}
故事结尾预告：{story_framework.get('story_ending', '')}

**当前关卡设定**
- 阶段名：{current_level.get('stage_label', '')}
- 进入前状态：{current_level.get('state_before', '')}
- 主知识点：{primary_knowledge_point}
- 辅助知识点：{', '.join(supporting_knowledge_points) if supporting_knowledge_points else '无'}
- 知识点角色：{knowledge_role}
- 场景：{current_level.get('scene_description', '')}
- 当前任务：{current_level.get('core_task', '') or current_level.get('challenge', '')}
- 风险：{current_level.get('stakes', '')}
- 易错选项：{current_level.get('misconceptions', '')}
- 本关为什么使用这个知识点：{current_level.get('why_here', '')}
- 进入本关的过渡提示：{levels[level_index - 1].get('transition_hint', '（第一关，无前置过渡）') if level_index > 0 else '（第一关，无前置过渡）'}
- 本关成功后状态：{current_level.get('success_state', '') or current_level.get('exit_trigger', '')}
- 本关结束后的过渡提示：{current_level.get('transition_reason', '') or current_level.get('transition_hint', '')}
- 下一关入口参考：{next_level.get('entry_trigger', '') if next_level else '这是最后一关'}
- 本关的微决策焦点：{current_level.get('micro_decision_focus', '')}
- 去重复提醒：{current_level.get('variety_guard', '')}

**前面的关卡**（用于保持连贯性）
{previous_context if previous_context else '这是第一关'}

**最近几关已使用的任务外壳**
{recent_pattern_summary}

**知识库参考**
{rag_knowledge['summary']}

**本关误区与错误选项参考**
{choice_knowledge['summary']}

**本关教学建议参考**
{teaching_advice['summary']}

**本题骨架规划（必须遵守，不要偏离到别的操作维度）**
{json.dumps(option_blueprint, ensure_ascii=False, indent=2)}

**任务要求**
生成这一关的完整内容，包括：
1. 章节标题（统一使用“主标题：副标题”格式，长度适中，有趣有创意，与知识点相关）
2. 故事情节（自然连接前一关，明确承接当前状态与任务）
3. 问题描述
4. 三个选项（一个正确，两个错误）

**选项设计核心原则**
- 三个选项必须围绕同一个知识判断维度展开，像“把同一个知识点放进当前场景后的三种判断”，而不是三个不同的剧情方案
- 三个选项必须描述同一件事、同一种操作或同一种判断，只在一个知识判断槽位上拉开差异，如深/浅/正中、南/北/均匀、化合物/混合物/单质、正确顺序/错误顺序
- 选项字数尽量控制在20-35字之间，三个选项字数尽量接近，优先写成简洁的知识判断句
- 选项保持同类表达即可，不要求逐字同句式；但必须让人一眼看出它们在比较同一个知识判断点
- 错误选项必须是“看似合理但关键判断有误”的竞争性方案，不能一眼看出是错的，必须理解知识点才能判断
- 错误选项要优先基于上面的学生常见误区与错误选项参考设计，每个错误选项对应不同误区来源
- 如果误区素材与本题骨架不匹配，可以只借用其错误认知方向，禁止生硬照抄
- 如果一个候选错项只是“和场景有关”但不是典型误区、没有明确教学价值，就不要用它来凑第三个选项
- 必须严格遵守上面 `option_blueprint` 里的 `shared_action_template`、`option_sentence_pattern`、`variable_dimension`、`option_plausibility_guard`
- 选项要自然、贴近学生思维，不要过于学术化
- 动作和物品描述要自然，不要过于技术化
- `story_context` 和 `question_text` 负责剧情承接与任务背景，`option_text` 只负责知识辨析；不要把剧情细节继续灌进选项
- 选项正文只写“方案本身 + 简短知识依据”，不要把完整原理、后果、科普讲解塞进 `option_text`
- 除非知识判断本身必须依赖该名词，否则不要在 `option_text` 里重复地名、地标、专有名词、具体树名或无关道具细节
- 正确选项的 `analysis` 和 `knowledge`，以及错误选项的 `explanation`，要吸收上面的教学建议，做到专业、生动、有引导性
- `question_text` 必须把任务目标说清楚，并把知识点自然放进场景；到了选项层就应主要比较知识判断，而不是继续扩写场景
- 错误选项的核心必须是一个清晰的知识误区；行动框架可以相同，但误区必须有代表性、像真实学生会犯的错
- 错误原因应来自真实学生误判，如反向理解、条件遗漏、概念混淆、顺序错误、经验主义误判，禁止荒谬型错项
- 禁止生成只凭语感或常识就能秒排除的选项；三个选项都应像是认真思考后的方案
- 禁止出现“明显送命”“明显摆烂”“明显瞎试”的错误项；错误项也必须像真实学生会采纳的近似方案
- 每个选项都必须包含与知识点相关的判断依据，但依据要简短，不能写成小论文
- 三个选项去掉 A/B/C 标签后，也都必须像学生认真权衡后会考虑的方案
- 如果一个选项看起来像标准答案，而另外两个只是为了凑数的场景替代方案，说明设计失败，需要重写
- 当前题目的微决策外壳必须服务于剧情任务，不要写成脱离故事状态的标准化题库题
- 如果最近几关已经用了相近外壳，本关必须在保证自然的前提下换一个新的决策角度

**动作反馈与结果反馈要求**
- 每个选项都要生成两段不同内容：`action_feedback` 和 `outcome_feedback`
- `action_feedback`：描述你选择该方案后立刻做了什么，承接选项但不能直接复述 `option_text`
- `outcome_feedback`：描述动作之后具体发生了什么后果，必须明确体现结果的好坏、推进或受阻情况，并让人能感受到你将进入的下一幕状态
- `action_feedback` 偏过程镜头，`outcome_feedback` 偏结果落点，二者禁止同义改写

**选项结构示例**
```json
{{
  "option_id": "A",
  "option_text": "正确选项内容（优先20-35字，最多不超过40字，只写唯一知识槽位的取值，不扩写原因链）",
  "is_correct": true,
  "action_feedback": "选择后你立刻做了什么（自然、清楚、有画面感，承接选项但不能复述选项原句）",
  "outcome_feedback": "动作后具体发生了什么结果（自然展开即可，必须能看出推进效果）",
  "result": "兼容旧字段：与 outcome_feedback 保持一致",
  "analysis": "在这个场景中应用了什么原理，起到了什么作用，得到了什么结果（50-80字）",
  "knowledge": "这个原理是什么，定义、公式、定律内容（50-80字）"
}},
{{
  "option_id": "B",
  "option_text": "错误选项内容（优先20-35字，最多不超过40字，与A、C必须是同一知识槽位上的误区项）",
  "is_correct": false,
  "action_feedback": "选择后你立刻做了什么（自然、清楚、有画面感，承接选项但不能复述原句）",
  "outcome_feedback": "动作后具体发生了什么（必须是确定的受阻/失败/偏离后果，禁止出现“幸好”“还好”“虚惊一场”等模糊表达）",
  "result": "兼容旧字段：与 outcome_feedback 保持一致",
  "explanation": "解释为什么错误，错误认知是什么，正确原理是什么（60-100字）"
}},
{{
  "option_id": "C",
  "option_text": "错误选项内容（优先20-35字，最多不超过40字，与A、B必须是同一知识槽位上的误区项）",
  "is_correct": false,
  "action_feedback": "选择后你立刻做了什么（自然、清楚、有画面感，承接选项但不能复述原句）",
  "outcome_feedback": "动作后具体发生了什么（必须是确定的失败后果，禁止出现“幸好”“还好”“虚惊一场”等模糊表达）",
  "result": "兼容旧字段：与 outcome_feedback 保持一致",
  "explanation": "解释为什么错误（60-100字）"
}}
注意：`question_text` 必须直接问一个唯一知识槽位，如“滤层从上到下如何排列”“支点应放在哪个位置”“地图应依据什么定向”；三个选项必须只是这个槽位的不同取值，而不是三个剧情方案。情节放在题干里，选项聚焦同槽位辨析，禁止在选项中同时引入动作步骤、原因解释、风险规避或第二知识维度。

**输出格式**（严格按照JSON格式）：
```json
{{
  "question_id": {level_index + 1},
  "chapter_title": "主标题：副标题",
  "story_context": "题目前的故事情节（40-90字，明确承接上一关结果与当前状态）",
  "question_text": "问题描述",
  "knowledge_point": "{primary_knowledge_point}",
  "options": [
    {{
      "option_id": "A",
      "option_text": "正确选项内容（优先20-35字，最多不超过40字，只写唯一知识槽位的取值）",
      "is_correct": true,
      "action_feedback": "选择后你立刻做了什么（自然、清楚、有画面感，承接选项但不能复述选项原句）",
      "outcome_feedback": "动作后具体发生了什么结果（自然展开即可，必须能看出推进效果）",
      "result": "兼容旧字段：与 outcome_feedback 保持一致",
      "analysis": "在这个场景中应用了什么原理，起到了什么作用，得到了什么结果（50-80字）",
      "knowledge": "这个原理是什么，定义、公式、定律内容（50-80字）"
    }},
    {{
      "option_id": "B",
      "option_text": "错误选项内容（优先20-35字，最多不超过40字，与A、C必须是同一知识槽位上的误区项）",
      "is_correct": false,
      "action_feedback": "选择后你立刻做了什么（自然、清楚、有画面感，承接选项但不能复述原句）",
      "outcome_feedback": "动作后具体发生了什么（必须是确定的受阻/失败/偏离后果，禁止出现“幸好”“还好”“虚惊一场”等模糊表达）",
      "result": "兼容旧字段：与 outcome_feedback 保持一致",
      "explanation": "解释为什么错误，错误认知是什么，正确原理是什么（60-100字）"
    }},
    {{
      "option_id": "C",
      "option_text": "错误选项内容（优先20-35字，最多不超过40字，与A、B必须是同一知识槽位上的误区项）",
      "is_correct": false,
      "action_feedback": "选择后你立刻做了什么（自然、清楚、有画面感，承接选项但不能复述原句）",
      "outcome_feedback": "动作后具体发生了什么（必须是确定的失败后果，禁止出现“幸好”“还好”“虚惊一场”等模糊表达）",
      "result": "兼容旧字段：与 outcome_feedback 保持一致",
      "explanation": "解释为什么错误（60-100字）"
    }}
  ],
  "transition_to_next": "过渡到下一题的情节（自然承接即可，不必刻意控字数)"
}}
```

请只输出JSON：
"""

        response = self._call_llm(prompt, temperature=0.7)
        parsed = self._parse_json_response(response)
        if not parsed:
            repair_prompt = f"""请把下面这段本应为 JSON 的内容修复成合法 JSON。

要求：
1. 只输出 JSON 对象本身，不要解释
2. 保留原有字段和值，不要改写语义
3. 删除尾逗号、补全括号、修正格式错误
4. 如果存在 option_blueprint、action_feedback、outcome_feedback、result 等字段，尽量原样保留

原始内容：
{response}
"""
            repaired_response = self._call_llm(repair_prompt, temperature=0.0, max_tokens=12000)
            parsed = self._parse_json_response(repaired_response)

        parsed = self._sanitize_llm_payload(parsed)
        parsed["knowledge_point"] = primary_knowledge_point
        parsed["primary_knowledge_point"] = primary_knowledge_point
        parsed["supporting_knowledge_points"] = supporting_knowledge_points
        parsed["knowledge_role_in_this_level"] = knowledge_role
        parsed["transition_to_next"] = parsed.get("transition_to_next", "") or current_level.get('transition_hint', '') or current_level.get('transition_reason', '')

        for option in parsed.get("options", []):
            outcome_feedback = option.get("outcome_feedback", "") or option.get("result", "")
            action_feedback = option.get("action_feedback", "")
            if not action_feedback:
                action_feedback = option.get("option_text", "")
            option["action_feedback"] = action_feedback
            option["outcome_feedback"] = outcome_feedback
            option["result"] = outcome_feedback

        parsed["option_blueprint"] = option_blueprint

        return parsed

    def _normalize_option_feedbacks(self, parsed: Dict) -> Dict:
        """统一补齐 action_feedback / outcome_feedback / result 字段。"""
        for question in parsed.get("questions", []):
            for option in question.get("options", []):
                raw_result = option.get("result", "")
                action_feedback = option.get("action_feedback", "")
                outcome_feedback = option.get("outcome_feedback", "")

                if not action_feedback and raw_result:
                    split_index = -1
                    for marker in ["。", "！", "？", ". ", "! ", "? "]:
                        idx = raw_result.find(marker)
                        if idx != -1:
                            split_index = idx + len(marker.strip())
                            if marker.endswith(" "):
                                split_index = idx + len(marker)
                            break

                    if split_index != -1 and split_index < len(raw_result):
                        action_feedback = raw_result[:split_index].strip()
                        outcome_feedback = outcome_feedback or raw_result[split_index:].strip()
                    else:
                        action_feedback = option.get("option_text", "")

                if not action_feedback:
                    action_feedback = option.get("option_text", "")
                if not outcome_feedback:
                    outcome_feedback = raw_result

                option["action_feedback"] = action_feedback
                option["outcome_feedback"] = outcome_feedback
                option["result"] = outcome_feedback

        return parsed

    def _summarize_recent_task_patterns(self,
                                        previous_questions: List[Dict],
                                        limit: int = 3) -> str:
        recent_questions = previous_questions[-limit:]
        if not recent_questions:
            return "暂无前置任务外壳，可以自由选择最自然的微决策形态。"

        parts = []
        for question in recent_questions:
            blueprint = question.get("option_blueprint", {})
            parts.append(
                f"- 第{question.get('question_id', '?')}关："
                f"任务形态={blueprint.get('task_type', '未标注')}；"
                f"变化维度={blueprint.get('variable_dimension', '未标注')}；"
                f"任务目标={self._truncate_text(blueprint.get('real_world_goal', question.get('question_text', '')), 80)}"
            )
        return "\n".join(parts)
    
    def _retrieve_knowledge(self, knowledge_points: List[str]) -> Dict:
        """从RAISS检索相关知识"""
        retrieved = self.retriever.retrieve_for_story_generation(
            knowledge_points, top_k=3
        )
        bridges = []
        for i in range(len(knowledge_points) - 1):
            bridges.append(
                self.retriever.retrieve_bridge_examples(
                    knowledge_points[i],
                    knowledge_points[i + 1],
                    top_k=2
                )
            )
        
        # 构建知识摘要
        summary_parts = []
        for kp, data in retrieved.items():
            summary_parts.append(f"\n【{kp}】")
            if data['core_concept']:
                summary_parts.append(f"核心概念: {data['core_concept']['text'][:450]}")
            if data['story_cases']:
                for i, case in enumerate(data['story_cases'][:3], 1):
                    summary_parts.append(f"案例{i}: {case['text'][:350]}")
        if bridges:
            summary_parts.append("\n【知识点桥接参考】")
            for bridge in bridges:
                summary_parts.append(f"从 {bridge['from_knowledge_point']} 过渡到 {bridge['to_knowledge_point']}")
                for case in bridge['from_story_cases'][:1]:
                    summary_parts.append(f"前一知识点案例: {case['text'][:150]}")
                for case in bridge['to_story_cases'][:1]:
                    summary_parts.append(f"后一知识点案例: {case['text'][:150]}")
        
        knowledge_summary = "\n".join(summary_parts)
        print(f"  ✓ 检索到 {len(knowledge_points)} 个知识点的相关内容")
        
        return {
            "raw": retrieved,
            "summary": knowledge_summary,
            "bridges": bridges
        }

    def _retrieve_misconceptions_and_errors(self, knowledge_point: str, scenario: str) -> Dict:
        """检索本关错误选项设计所需的误区与错误选项素材"""
        results = self.retriever.retrieve_for_choice_generation(knowledge_point, scenario, top_k=3)
        summary_parts = []

        if results.get('misconceptions'):
            summary_parts.append("- 学生常见误区：")
            for item in results['misconceptions'][:3]:
                summary_parts.append(f"  - {item['text'][:180]}")

        if results.get('error_options'):
            summary_parts.append("- 可参考的错误选项设计：")
            for item in results['error_options'][:2]:
                summary_parts.append(f"  - {item['text'][:180]}")

        return {
            "raw": results,
            "summary": "\n".join(summary_parts) if summary_parts else "无专门误区素材，请基于学生常见错误思维自行设计。",
            "soft_injection_guidance": self._build_soft_injection_guidance({"raw": results})
        }

    def _retrieve_teaching_advice(self, knowledge_point: str) -> Dict:
        """检索本关解答生成所需的教学建议"""
        advice_items = self.retriever.retrieve_teaching_advice(knowledge_point, top_k=2)
        summary_parts = []
        for item in advice_items[:2]:
            summary_parts.append(f"- {item['text'][:220]}")

        return {
            "raw": advice_items,
            "summary": "\n".join(summary_parts) if summary_parts else "无专门教学建议，请采用教师引导式讲解。"
        }
    
    def _generate_initial(self,
                          knowledge_points: List[str],
                          scenario: str,
                          num_questions: int,
                          rag_knowledge: Dict) -> Dict:
        """初步生成题目 - 使用COT思考过程提高质量"""
        
        prompt = f"""你是一位教育专家，需要创作一个融合中学阶段各种知识点的交互式故事。

**候选知识点池**（从中选择 {num_questions} 个最适合串联成连贯故事的知识点）
{', '.join(knowledge_points)}

**知识库参考内容**
{rag_knowledge['summary']}

**故事大方向**
{scenario}
（注意：这只是大方向，具体的每一关场景和情节由你自由设计，确保剧情自然流畅）

**任务要求**
请使用Chain of Thought（思维链）方法，逐步推理并生成包含 {num_questions} 个选择题的交互式故事。
使用第二人称视角（"你"）。

---

**第一步：从候选池中选择知识点并规划顺序**
从上面的候选知识点池中，选择 {num_questions} 个最容易串联成连贯故事的知识点：
- 优先选择在同一场景下能自然衔接的知识点组合
- 确定它们的出场顺序，使剧情发展有逻辑递进感
- 说明为什么选择这些知识点、为什么按这个顺序排列

**第二步：设计具体场景和剧情**
基于你选择的知识点和故事大方向，设计具体的场景：
- 每个知识点对应一个具体的挑战场景，场景要符合故事大方向的设定
- 场景之间的过渡要自然，前一关的结果自然引出下一关的场景
- 避免生硬的知识点堆砌，让知识点融入剧情而非强行插入

**第三步：设计迷惑性选项（极其重要！）**

**核心原则：三个选项必须在同一维度上，只有关键参数不同！**

例如，杠杆原理的题目：
- 三个选项都应该是"使用杠杆"，只是支点位置、力臂长度等参数不同
- 不要出现"用绳子拉"、"找人帮忙"、"浇水"这种完全不同维度的选项

**正确的选项设计示例**（杠杆原理）：
- ✓ A: "将支点放在靠近重物的位置，使阻力臂短、动力臂长，用较小的力撬起重物"
- ✗ B: "将支点放在铁棍中间位置，使两边力臂相等，这样受力更均匀"
- ✗ C: "将支点放在远离重物的位置，使阻力臂长、动力臂短，因为这样更稳定"

**正确的选项设计示例**（摩擦力）：
- ✓ A: "在箱子底部垫上圆木棍，将滑动摩擦变为滚动摩擦，大幅减小阻力"
- ✗ B: "在箱子底部垫上方形木块，增大接触面积来减小摩擦力"
- ✗ C: "在箱子底部垫上粗糙的砂纸，增加摩擦系数使推动更稳定"

注意：所有选项都是"在底部垫东西"，只是垫的材料和物理理由不同，这样才有真正的迷惑性！

---

**输出格式**（严格按照JSON格式）：
```json
{{
  "story_intro": "故事开头（50-100字）",
  "questions": [
    {{
      "question_id": 1,
      "chapter_title": "与当前场景相关的章节名（6字左右，有趣有创意，例如：撬石板之谜、柜子搬运记、书架稳固术）",
      "story_context": "题目前的故事情节（30-50字）",
      "question_text": "问题描述",
      "knowledge_point": "涉及的知识点",
      "options": [
        {{
          "option_id": "A",
          "option_text": "选项内容（包含物理理由，句式与其他选项一致）",
          "is_correct": true,
          "result": "选择后发生的结果（描述选择正确后场景中具体发生了什么）",
          "analysis": "解析：在这个场景中你应用了什么原理，这个原理起到了什么作用，最终得到了什么结果（结合场景说明原理的应用过程和效果，50-80字）",
          "knowledge": "知识点：这个原理是什么？讲解原理的定义、公式、定律内容，以及更深层本质的物理知识（纯粹的物理知识讲解，不涉及具体场景，50-80字）"
        }},
        {{
          "option_id": "B",
          "option_text": "选项内容（包含错误的物理理由/误区）",
          "is_correct": false,
          "result": "选择后发生的错误结果（描述选择错误后场景中具体发生了什么）",
          "explanation": "解释为什么这个选择是错误的：错误的物理认知是什么，正确的原理是什么，在这个场景中为什么行不通（80-120字，把错误原因解释清楚即可）"
        }},
        {{
          "option_id": "C",
          "option_text": "选项内容（包含错误的物理理由/误区）",
          "is_correct": false,
          "result": "选择后发生的错误结果",
          "explanation": "解释为什么这个选择是错误的（80-120字）"
        }}
      ],
      "transition_to_next": "过渡到下一题的情节"
    }}
  ],
  "story_ending": "故事结尾（30-50字）"
}}
```

请开始思考和生成（只输出JSON）：
"""
        
        response = self._call_llm(prompt, temperature=0.7)
        parsed = self._parse_json_response(response)
        
        print(f"  ✓ 初步生成完成，包含 {len(parsed.get('questions', []))} 个题目")
        
        return {
            "raw_response": response,
            "parsed": parsed
        }
    
    def _reflect_on_questions(self, initial_result: Dict, story_framework: Optional[Dict] = None) -> Dict:
        """COT反思 - 分析题目的缺点"""
        
        questions_json = json.dumps(initial_result['parsed'], ensure_ascii=False, indent=2)
        framework_summary = self._build_story_framework_summary(story_framework or {})
        
        prompt = f"""你是一位资深的教育专家和题目审核员。请仔细分析以下交互式故事题目，找出其中的问题和不足。

**剧情任务链摘要**
{framework_summary}

**待审核的题目**
{questions_json}

**请逐步分析以下方面**

**第一步：分析故事连贯性**
- 题目之间的过渡是否自然，前一关的结果自然引出下一关？
- 情节发展是否合理，知识点是否被自然融入？
- 每一关是否真的继承了对应阶段的状态与任务，还是只是套了一个情境壳？
- 有哪些生硬或突兀的地方？

**第二步：分析任务外壳与选项质量**
- 不同关卡之间的微决策外壳是否过度重复？如果只是换了物品或场景皮肤，也要指出
- 正确选项是否准确体现知识点？如果选项中没有明显的知识点，则需要重新设计选项。
- 错误选项是否有足够的迷惑性？错误选项要优先基于上面的学生常见误区与错误选项参考设计。
- 选项之间是否有明显的区分度？如果选项内容大部分都是重复的，只有一部分细节不一致，这种选项需要重新设计。
- 有没有选项过于明显或过于离谱？如果有，需要重新设计选项。
- 是否存在“只靠生活常识/风险直觉就能秒选正确答案”的情况？如果存在，必须判定为高优先级问题。
- 是否存在“错误选项只是看起来更危险、更麻烦、更极端”，而不是在同一任务框架下只错一个关键知识判断点？如果存在，必须指出。
- 三个选项是否都像认真思考后的可执行方案，还是其中某些选项明显像凑数、摆烂、胡来？如果是后者，必须指出。
- 判断正确答案时，是否必须真正调用知识点原理；如果不需要调用知识，只凭语感或常识就能做题，必须指出。
- 对每一关都要优先检查：错误项是否属于“竞争性干扰项”。如果不是，请明确说明是哪里不具竞争性。

**第二步补充判定标准（非常重要）**
- 如果一个错误选项因为“明显更危险”“明显更费力”“明显更夸张”而容易被排除，这不算高质量干扰项。
- 如果一个错误选项没有共享与正确项相同的任务维度/行动框架，也判定为不合格。
- 如果一个错误选项的问题不在知识判断，而只是态度冲动、行为失控、故意乱来，也应判定为不合格，除非本题知识点本身就是情绪/行为管理。
- 只要发现某一关存在“一眼错”“常识秒选”“非竞争性干扰项”，就在 improvement_points 中加入对应问题，且 priority 设为 high。
- 如果连续多关使用了几乎同一种任务外壳，导致玩家体验重复，也应在 improvement_points 中指出。

**第三步：总结需要改进的具体问题**
请列出尽可能多的最需要改进的具体问题，每个问题要具体明确，便于后续搜索补充。
- 不要因为整体故事流畅就忽略选项层面的缺陷。
- 只要某一关的错误选项不够“竞争性”、过于明显、或无法逼迫学生调用知识点思考，就必须列入问题。
- improvement_points 里至少优先覆盖“选项设计”问题，不能只关注字段冗余或文档一致性。
- 如果某关没有真正承接前一关留下的状态，也必须列入问题。

**输出格式**（严格按照JSON格式）：
```json
{{
  "coherence_analysis": {{
    "issues": ["问题1", "问题2"],
    "suggestions": ["建议1", "建议2"]
  }},
  "options_analysis": {{
    "issues": ["问题1", "问题2"],
    "suggestions": ["建议1", "建议2"]
  }},
  "knowledge_analysis": {{
    "issues": ["问题1", "问题2"],
    "suggestions": ["建议1", "建议2"]
  }},
  "improvement_points": [
    {{
      "issue": "具体问题描述",
      "search_query": "用于搜索补充的查询词",
      "priority": "high/medium/low"
    }}
  ]
}}
```

请开始分析：
"""
        
        response = self._call_llm(prompt, temperature=0.3)
        parsed = self._parse_json_response(response)
        
        improvement_points = parsed.get('improvement_points', [])
        print(f"  ✓ 反思完成，发现 {len(improvement_points)} 个需要改进的问题")
        
        for i, point in enumerate(improvement_points, 1):
            print(f"    {i}. [{point.get('priority', 'medium')}] {point.get('issue', '')[:50]}...")
        
        return {
            "raw_response": response,
            "parsed": parsed
        }
    
    def _search_for_improvements(self,
                                  reflection: Dict,
                                  knowledge_points: List[str]) -> List[Dict]:
        """针对每个问题单独搜索补充内容 - 使用反思中的具体问题"""
        
        improvement_points = reflection['parsed'].get('improvement_points', [])
        search_results = []
        
        # 只处理高优先级和中优先级的问题
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_points = sorted(
            improvement_points,
            key=lambda x: priority_order.get(x.get('priority', 'medium'), 1)
        )
        
        # 已搜索的知识点，避免重复搜索
        searched_kps = set()
        
        # 最多搜索3个问题（减少重复）
        for point in sorted_points[:3]:
            issue = point.get('issue', '')
            
            # 从问题中提取相关知识点
            related_kp = self._extract_knowledge_point_from_issue(issue, knowledge_points)
            
            # 避免重复搜索同一知识点
            if related_kp in searched_kps:
                continue
            searched_kps.add(related_kp)
            
            # 单独搜索这个问题
            result = self._search_single_issue(issue, related_kp)
            
            search_results.append({
                "issue": issue,
                "knowledge_point": related_kp,
                "result": result
            })
        
        print(f"  ✓ 完成 {len(search_results)} 个问题的搜索补充")
        
        return search_results
    
    def _extract_knowledge_point_from_issue(self, issue: str, knowledge_points: List[str]) -> str:
        """从问题描述中提取相关知识点"""
        question_match = re.search(r'第\s*(\d+)\s*关', issue)
        if question_match:
            question_index = int(question_match.group(1)) - 1
            if 0 <= question_index < len(knowledge_points):
                return knowledge_points[question_index]

        for kp in knowledge_points:
            if kp in issue:
                return kp
        # 如果没找到，返回第一个知识点
        return knowledge_points[0] if knowledge_points else "物理"
    
    def _generate_improved(self,
                           initial_result: Dict,
                           reflection: Dict,
                           search_supplements: List[Dict],
                           story_framework: Dict,
                           knowledge_points: List[str]) -> Dict:
        """基于补充内容改进生成"""
        
        # 构建补充知识摘要
        supplement_summary = []
        for item in search_supplements:
            # 处理search_and_extract返回的results列表
            results = item['result'].get('results', [])
            if results:
                supplement_summary.append(f"【关于：{item['issue'][:100]}】")
                for r in results[:2]:
                    if r.get('content'):
                        supplement_summary.append(f"- {r['content'][:600]}")
        
        supplement_text = "\n".join(supplement_summary) if supplement_summary else "无额外补充"
        
        # 构建改进建议摘要
        improvement_suggestions = []
        for key in ['coherence_analysis', 'options_analysis', 'knowledge_analysis']:
            analysis = reflection['parsed'].get(key, {})
            suggestions = analysis.get('suggestions', [])
            improvement_suggestions.extend(suggestions)

        rag_augmentation = []
        for question in initial_result['parsed'].get('questions', []):
            kp = question.get('knowledge_point', '')
            scenario_text = question.get('story_context', '')
            choice_knowledge = self._retrieve_misconceptions_and_errors(kp, scenario_text)
            teaching_advice = self._retrieve_teaching_advice(kp)
            rag_augmentation.append(f"【{kp} 的误区与错误选项参考】\n{choice_knowledge['summary']}")
            rag_augmentation.append(f"【{kp} 的教学建议】\n{teaching_advice['summary']}")
        
        suggestions_text = "\n".join([f"- {s}" for s in improvement_suggestions[:10]])
        augmentation_text = "\n\n".join(rag_augmentation) if rag_augmentation else "无额外的误区或教学建议补兑。"

        # Task 4: 构建按题目序号的定点修复指令，避免 LLM 笼统改写而不解决具体问题
        improvement_points_list = reflection['parsed'].get('improvement_points', [])
        questions_list = initial_result['parsed'].get('questions', [])
        targeted_fixes = []
        for point in improvement_points_list:
            issue = point.get('issue', '')
            matched_q = None
            for q in questions_list:
                kp = q.get('knowledge_point', '')
                qid = str(q.get('question_id', ''))
                if kp and kp in issue:
                    matched_q = q
                    break
                if qid and (f"第{qid}关" in issue or f"question{qid}" in issue):
                    matched_q = q
                    break
            if matched_q:
                targeted_fixes.append(
                    f"- 第{matched_q.get('question_id', '?')}关（{matched_q.get('knowledge_point', '')}）：{issue}"
                )
            else:
                targeted_fixes.append(f"- （全局）：{issue}")
        targeted_fixes_text = "\n".join(targeted_fixes) if targeted_fixes else suggestions_text
        framework_summary = self._build_story_framework_summary(story_framework)

        stage_reference_parts = []
        levels = story_framework.get('levels', [])
        for level in levels:
            stage_reference_parts.append(
                f"第{level.get('level_number', '?')}关规划\n"
                f"- 阶段名：{level.get('stage_label', '')}\n"
                f"- 进入前状态：{level.get('state_before', '')}\n"
                f"- 当前任务：{level.get('core_task', '') or level.get('challenge', '')}\n"
                f"- 风险：{level.get('stakes', '')}\n"
                f"- 知识点：{level.get('knowledge_point', '')}\n"
                f"- 微决策焦点：{level.get('micro_decision_focus', '')}\n"
                f"- 去重复提醒：{level.get('variety_guard', '')}\n"
                f"- 成功后状态：{level.get('success_state', '') or level.get('exit_trigger', '')}\n"
                f"- 过渡原因：{level.get('transition_reason', '') or level.get('transition_hint', '')}"
            )
        stage_reference_text = "\n\n".join(stage_reference_parts) if stage_reference_parts else "无额外关卡规划摘要。"

        prompt = f"""你是一位教育专家。请根据以下反馈和补充资料，改进交互式故事题目。

**既定剧情任务链（必须保持）**
{framework_summary}

**原始题目**
{json.dumps(initial_result['parsed'], ensure_ascii=False, indent=2)}

**逐关规划参考**
{stage_reference_text}

**需要改进的问题（请按题目序号定点修复，不要整体改写）**
{targeted_fixes_text}

**搜索补充的知识**
{supplement_text}

**知识库深度补充（误区 / 错误选项 / 教学建议）**
{augmentation_text}

**改进要求**
1. 保持整体故事场景、主线任务和已采用的知识点集合基本不变，但允许按关重构题干、选项、局部场景和过渡
2. 每一关都必须继续承接“进入前状态 -> 当前任务 -> 成功后状态”这条链路，不能把关卡改成脱离主线的小练习
3. 如果某一关原本的题目骨架不合理，可以重写该关的 `question_text`、`story_context`、`transition_to_next`，必要时重构这一关的任务表达
4. 改进题目之间的过渡，使其更加自然流畅，让下一幕明显来自上一幕结果
5. **重点改进选项设计（极其重要！）**：
   - **三个选项必须在同一维度上，只有关键参数不同**
   - **每个选项字数控制在20-35字，三个选项字数要接近**
   - **三个选项的句式结构要相似**（如都是"将...放在...，因为..."的格式）
   - 例如杠杆题：三个选项都是"使用杠杆"，只是支点位置不同
   - 例如摩擦力题：三个选项都是"在底部垫东西"，只是垫的材料不同
   - 不要出现"用绳子拉"、"找人帮忙"这种完全不同维度的选项
   - 错误选项要优先映射到补充材料中的学生误区；如果映射会让题目变得生硬，则保留误区方向但重新写成符合场景的自然错误
   - 错误选项必须是“看似合理但关键判断有误”的竞争性方案，禁止一眼错、荒谬错、搞笑错
   - 每个选项都必须包含与知识点有关的判断依据，不能只有动作没有理由
   - 错误选项只能在一个关键认知点上出错，行动框架本身仍要成立
   - **禁止把“更危险、更累、更夸张”直接当作错误项设计手段**；错误项必须像学生认真推理后可能会选的方案
   - **禁止生成只靠生活常识就能秒排除的错误项**；学生必须比较知识依据才能区分正误
   - 如果某关现有错误选项不够竞争性，允许你直接重写该关全部三个选项，而不是小修小补
   - 优先把错误项设计成：概念混淆、条件遗漏、方向映射错误、顺序错误、局部经验误用、单一线索过度泛化
   - 错误项的失败后果要具体，但不能在选项文本里预先暴露“这很危险所以别选”的信号
   - 如果多关的微决策外壳过于相似，允许你在不破坏剧情任务链的前提下，换一个更自然的决策角度，避免重复感
6. 确保知识点准确无误
7. **所有选项都必须包含两段式反馈**：
   - `action_feedback`：描述选择该方案后立刻做了什么，承接选项但不能直接复述 `option_text`
   - `outcome_feedback`：描述动作之后具体发生了什么后果，必须清楚体现推进 / 受阻 / 偏离 / 失败，并让读者感受到下一幕状态
   - `action_feedback` 偏过程镜头，`outcome_feedback` 偏结果落点，二者禁止同义改写
   - `result` 为兼容旧字段，内容必须与 `outcome_feedback` 保持一致
8. **正确选项**要有result、analysis和knowledge三个字段：
   - result：与 outcome_feedback 保持一致
   - analysis：在场景中应用了什么原理，起到了什么作用，得到了什么结果（60-100字）
   - knowledge：这个原理是什么，定义、公式、定律、更深层本质（纯粹知识点知识，不涉及场景，60-100字）
9. **错误选项**要有result和explanation两个字段：
   - result：与 outcome_feedback 保持一致
   - explanation：解释为什么错误，错误认知是什么，正确原理是什么（80-120字，把错误原因解释清楚即可），并尽量采用教师引导式讲解风格
10. 每个question要有chapter_title字段（统一使用“主标题：副标题”格式，与场景相关，有趣有创意）
11. 如有必要，可以减少某些关卡中知识点的“定义背诵味”，让它更多体现在决策动作本身

**本轮改进的优先级要求**
- 相比字段冗余、轻微文案不一致等问题，优先修复“错误项质量不足”“一眼错”“不需要知识推理就能做对”的问题
- 如果反思意见与原题质量冲突，请优先执行“提升竞争性干扰项质量”这一目标
- 每一关都至少自检一次：学生是否必须真正比较三个方案中的知识依据，才能选出正确项；如果不是，请继续重写该关选项
- 如果某关虽然题目能做，但没有真正承接剧情任务链，也请优先修复
- 如果连续几关只是换皮重复，也请优先打散外壳

**正确的选项设计示例**（杠杆原理，注意字数和结构一致）：
- ✓ A: "将支点放在靠近石板的位置，这样阻力臂短、动力臂长，可以用较小的力撇动重物。"(38字)
- ✗ B: "将支点放在铁棍正中间的位置，这样两边力臂相等，更容易保持平衡稳定。"(37字)
- ✗ C: "将支点放在远离石板的位置，这样撇棍移动范围大，更容易找到发力点。"(36字)

**你必须避免的弱错误项示例**
- 弱错误项1："冲进沼泽里乱跑，因为这样更快"（问题：太明显，像故意犯错）
- 弱错误项2："什么都不做，坐着等"（问题：若知识点不是情绪管理/风险管理，则通常不是竞争性方案）
- 弱错误项3："选最陡最危险的路，因为刺激"（问题：只靠常识就能排除）

**你应该追求的错误项特征**
- 看起来有理由
- 与正确项做的是同一类任务
- 只在关键知识判断上偏了一点
- 学生需要比较原理、条件和后果，不能只凭常识排除

**输出格式**（严格JSON，不要有其他内容）：
```json
{{
  "story_intro": "改进后的故事开头",
  "questions": [
    {{
      "question_id": 1,
      "chapter_title": "与场景相关的章节名（6字左右，例如：撬石板之谜）",
      "story_context": "改进后的情节",
      "question_text": "改进后的问题",
      "knowledge_point": "知识点",
      "options": [
        {{
          "option_id": "A",
          "option_text": "选项内容（20-35字，与B、C结构相同）",
          "is_correct": true,
          "action_feedback": "选择后你立刻做了什么（35-50字，承接选项但不能复述原句）",
          "outcome_feedback": "动作后具体发生了什么结果（40-60字，必须能看出推进效果）",
          "result": "与 outcome_feedback 保持一致",
          "analysis": "在场景中应用了什么原理，起到了什么作用，得到了什么结果",
          "knowledge": "这个原理是什么，定义、公式、定律、更深层本质"
        }},
        {{
          "option_id": "B",
          "option_text": "选项内容（20-35字，与A、C结构相同）",
          "is_correct": false,
          "action_feedback": "选择后你立刻做了什么（35-50字，承接选项但不能复述原句）",
          "outcome_feedback": "动作后具体发生了什么（40-60字，必须是确定的失败后果，禁止“幸好”“还好”等模糊表达）",
          "result": "与 outcome_feedback 保持一致",
          "explanation": "解释为什么错误，错误认知是什么，正确原理是什么"
        }},
        {{
          "option_id": "C",
          "option_text": "选项内容（20-35字，与A、B结构相同）",
          "is_correct": false,
          "action_feedback": "选择后你立刻做了什么（35-50字，承接选项但不能复述原句）",
          "outcome_feedback": "动作后具体发生了什么（40-60字，必须是确定的失败后果，禁止“幸好”“还好”等模糊表达）",
          "result": "与 outcome_feedback 保持一致",
          "explanation": "解释为什么错误"
        }}
      ],
      "transition_to_next": "改进后的过渡"
    }}
  ],
  "story_ending": "改进后的结尾"
}}
```

请只输出JSON：
"""
        
        response = self._call_llm(prompt, temperature=0.5)
        parsed = self._parse_json_response(response)
        if not parsed or not parsed.get("questions"):
            repair_prompt = f"""请把下面这段内容整理成合法 JSON，并且只输出 JSON 对象本身。

要求：
1. 删除 `<think>...</think>`、解释文字、标题、代码围栏等非 JSON 内容
2. 保留原有 JSON 字段和值，不要改写语义
3. 若存在 `action_feedback` / `outcome_feedback` / `result`，尽量原样保留
4. 如果 `result` 与 `outcome_feedback` 重复，只保留合法 JSON，不要额外解释

原始内容：
{response}
"""
            repaired_response = self._call_llm(repair_prompt, temperature=0.0, max_tokens=16000)
            parsed = self._parse_json_response(repaired_response)
        parsed = self._normalize_option_feedbacks(parsed)

        if not parsed or not parsed.get("questions"):
            print("  ⚠ 改进阶段解析后仍无有效题目，稍后将回退到初始结果")
        
        print(f"  ✓ 改进生成完成")
        
        return {
            "raw_response": response,
            "parsed": parsed
        }
    
    def _parse_json_response(self, response: str) -> Dict:
        """解析JSON响应 - 增强容错处理"""
        json_str = ""
        try:
            json_str = self._extract_json_candidate(response)
            json_str = self._strip_trailing_commas(json_str).strip()
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON解析失败: {e}")
            # 尝试修复常见的JSON截断问题
            try:
                # 尝试补全被截断的JSON
                fixed_json = self._try_fix_truncated_json(self._strip_trailing_commas(json_str))
                if fixed_json:
                    return json.loads(fixed_json)
            except:
                pass
            
            # 保存原始响应用于调试
            debug_file = f"output/debug_response_{datetime.datetime.now().strftime('%H%M%S')}.txt"
            os.makedirs("output", exist_ok=True)
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"  ⚠ 原始响应已保存到: {debug_file}")
            return {}

    def _extract_json_candidate(self, response: str) -> str:
        """从模型响应中尽量提取最可能的 JSON 主体。"""
        if not response:
            return ""

        text = response.strip()

        while text.lower().startswith("<think>") and "</think>" in text.lower():
            lower_text = text.lower()
            think_end = lower_text.find("</think>")
            text = text[think_end + len("</think>"):].strip()

        if "```json" in text:
            json_start = text.find("```json") + 7
            json_end = text.find("```", json_start)
            if json_end == -1:
                return text[json_start:].strip()
            return text[json_start:json_end].strip()

        if "```" in text:
            first_fence = text.find("```") + 3
            fence_body = text[first_fence:]
            if fence_body.lstrip().startswith(("{", "[")):
                json_end = text.find("```", first_fence)
                if json_end == -1:
                    return text[first_fence:].strip()
                return text[first_fence:json_end].strip()

        balanced = self._find_balanced_json_block(text)
        if balanced:
            return balanced

        first_brace = text.find("{")
        first_bracket = text.find("[")
        candidates = [idx for idx in [first_brace, first_bracket] if idx != -1]
        if candidates:
            return text[min(candidates):].strip()

        return text

    def _find_balanced_json_block(self, text: str) -> Optional[str]:
        """寻找首个大致平衡的 JSON 对象或数组，忽略字符串内部括号。"""
        start = -1
        opener = ""
        for idx, ch in enumerate(text):
            if ch in "[{":
                start = idx
                opener = ch
                break

        if start == -1:
            return None

        closer = "}" if opener == "{" else "]"
        stack = []
        in_string = False
        escape = False

        for idx in range(start, len(text)):
            ch = text[idx]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if not stack:
                    continue
                expected = "}" if stack[-1] == "{" else "]"
                if ch == expected:
                    stack.pop()
                    if not stack:
                        return text[start:idx + 1].strip()

        return None
    
    def _try_fix_truncated_json(self, json_str: str) -> Optional[str]:
        """尝试修复被截断的JSON"""
        json_str = self._strip_trailing_commas(json_str)
        # 计算未闭合的括号
        open_braces = json_str.count('{') - json_str.count('}')
        open_brackets = json_str.count('[') - json_str.count(']')
        
        # 如果有未闭合的字符串，先尝试闭合
        if json_str.count('"') % 2 == 1:
            json_str += '"'
        
        # 补全括号
        json_str += ']' * open_brackets
        json_str += '}' * open_braces
        
        return json_str


def test_gemini_cot_generator():
    """测试Gemini CoT生成器"""
    print("\n" + "="*70)
    print("测试 Gemini CoT 生成器")
    print("="*70)
    
    # 初始化生成器
    generator = GeminiCoTGenerator()
    
    # 测试用例 - 8个知识点，贴近生活的场景
    result = generator.generate_interactive_story(
        knowledge_points=["光的反射", "光的折射", "凸透镜成像", "声音的传播", "电路连接", "欧姆定律", "电功率", "磁场"],
        scenario="周末帮爷爷修理老房子里的各种电器和设备",
        num_questions=8
    )
    
    # 保存结果
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"output/gemini_cot_result_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 只保存可序列化的部分
        save_data = {
            "knowledge_points": result["knowledge_points"],
            "scenario": result["scenario"],
            "rag_knowledge_summary": result["rag_knowledge"]["summary"],
            "initial_result": result["initial_result"]["parsed"],
            "reflection": result["reflection"]["parsed"],
            "final_result": result["final_result"]["parsed"]
        }
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 结果已保存到: {output_file}")
    
    # 打印最终结果摘要
    print("\n" + "="*70)
    print("【最终生成结果摘要】")
    print("="*70)
    
    final = result["final_result"]["parsed"]
    if final:
        print(f"\n故事开头: {final.get('story_intro', 'N/A')[:100]}...")
        
        questions = final.get('questions', [])
        for q in questions:
            print(f"\n题目 {q.get('question_id', '?')}: {q.get('question_text', 'N/A')[:50]}...")
            print(f"  知识点: {q.get('knowledge_point', 'N/A')}")
            options = q.get('options', [])
            for opt in options:
                mark = "✓" if opt.get('is_correct') else "✗"
                print(f"  {mark} {opt.get('option_id', '?')}: {opt.get('text', 'N/A')[:40]}...")
    
    return result


if __name__ == "__main__":
    test_gemini_cot_generator()
