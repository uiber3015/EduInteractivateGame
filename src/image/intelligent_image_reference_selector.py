#!/usr/bin/env python3
"""
智能参考图片选择器（基于scene_transition的两阶段生成策略）

策略：
1. 阶段A（新场景）：scene_transition=1 的节点，只参考主角图片，可全部并行生成
2. 阶段B（同场景）：scene_transition=0 的节点，参考主角图片 + 同场景中最近已生成图片的URL
   阶段B必须在阶段A完成后执行，以确保能获取到同场景的参考图片
"""

import json
from typing import Dict, List, Optional, Tuple
from collections import deque


class IntelligentImageReferenceSelector:
    """智能参考图片选择器（基于scene_transition字段）"""
    
    def __init__(self, story_graph: Dict, character_reference_url: str, recurring_character_refs: Optional[Dict[str, str]] = None, enable_scene_reference: bool = True, prompt_mode: str = "composed", max_reference_images: int = 14):
        """
        初始化选择器
        
        Args:
            story_graph: StoryGraph格式的字典
            character_reference_url: 主角参考图片URL
        """
        self.story_graph = story_graph
        self.character_reference_url = character_reference_url
        self.recurring_character_refs = recurring_character_refs or {}
        self.enable_scene_reference = enable_scene_reference
        self.prompt_mode = (prompt_mode or 'composed').strip().lower()
        self.max_reference_images = max(1, max_reference_images)
        self.nodes = {node['id']: node for node in story_graph.get('nodes', [])}
        self.edges = story_graph.get('edges', {})
        self.reverse_edges = self._build_reverse_edges()
        
        # 缓存：节点ID -> 生成的图片URL（阶段A完成后填充）
        self.node_to_image_url = {}
    
    def _get_node_reference_urls(self, node_id: str, scene_ref: Optional[str] = None) -> List[str]:
        node = self.nodes.get(node_id, {})
        metadata = node.get('metadata', {}) if isinstance(node.get('metadata'), dict) else {}
        recurring_characters = metadata.get('recurring_characters', [])
        role_reference_urls = metadata.get('role_reference_urls', [])

        reference_urls: List[str] = []
        if self.character_reference_url:
            reference_urls.append(self.character_reference_url)

        if isinstance(role_reference_urls, list):
            for ref_url in role_reference_urls:
                if ref_url and ref_url not in reference_urls:
                    reference_urls.append(ref_url)

        for character_name in recurring_characters:
            ref_url = self.recurring_character_refs.get(character_name)
            if ref_url and ref_url not in reference_urls:
                reference_urls.append(ref_url)

        if scene_ref:
            if scene_ref in reference_urls:
                reference_urls.remove(scene_ref)
            reference_urls.append(scene_ref)

        if len(reference_urls) <= self.max_reference_images:
            return reference_urls

        if scene_ref and scene_ref in reference_urls:
            return reference_urls[: self.max_reference_images - 1] + [scene_ref]
        return reference_urls[:self.max_reference_images]
    
    def _get_node_prompt(self, node: Dict) -> str:
        metadata = node.get('metadata', {}) if isinstance(node.get('metadata'), dict) else {}
        composed_prompt = metadata.get('composed_image_prompt', '')
        image_prompt = metadata.get('image_prompt', '')
        manual_override = metadata.get('manual_image_prompt_override', '')

        if self.prompt_mode == 'source':
            return image_prompt or composed_prompt or manual_override or ''
        if self.prompt_mode == 'manual':
            return manual_override or composed_prompt or image_prompt or ''
        return composed_prompt or manual_override or image_prompt or ''
    
    def _build_reverse_edges(self) -> Dict[str, List[str]]:
        """构建反向边（父节点映射）"""
        reverse = {}
        for node_id in self.nodes:
            reverse[node_id] = []
        
        for from_node, to_nodes in self.edges.items():
            for to_node in to_nodes:
                if to_node not in reverse:
                    reverse[to_node] = []
                reverse[to_node].append(from_node)
        
        return reverse
    
    def _is_new_scene(self, node_id: str) -> bool:
        """
        判断节点是否为新场景（scene_transition=1）
        
        Args:
            node_id: 节点ID
            
        Returns:
            True表示新场景，False表示同场景延续
        """
        node = self.nodes.get(node_id)
        if not node:
            return True  # 安全默认值
        
        scene_transition = node.get('metadata', {}).get('scene_transition', 1)
        return scene_transition == 1
    
    def _find_same_scene_reference(self, node_id: str) -> Optional[str]:
        """
        为同场景节点（scene_transition=0）找到最近的已生成图片URL
        
        向父节点方向回溯，找到同场景中最近一个已有图片的节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            参考图片URL，找不到则返回None
        """
        visited = set()
        queue = deque()
        
        # 从父节点开始搜索
        parents = self.reverse_edges.get(node_id, [])
        for parent in parents:
            queue.append(parent)
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            # 如果这个节点已有生成的图片URL，使用它
            if current in self.node_to_image_url:
                return self.node_to_image_url[current]
            
            # 继续向父节点回溯
            parents = self.reverse_edges.get(current, [])
            for parent in parents:
                if parent not in visited:
                    queue.append(parent)
        
        return None
    
    def _is_image_already_generated(self, node_id: str) -> bool:
        """判断节点是否已经生成过图片（支持续跑场景）"""
        node = self.nodes.get(node_id, {})
        metadata = node.get('metadata', {}) if isinstance(node.get('metadata'), dict) else {}
        return bool(metadata.get('image_url') or metadata.get('image_path'))
    
    def set_node_image_url(self, node_id: str, image_url: str) -> None:
        """
        记录节点的生成图片URL（阶段A完成后调用）
        
        Args:
            node_id: 节点ID
            image_url: 生成的图片URL
        """
        self.node_to_image_url[node_id] = image_url
    
    def prepare_phase_a_tasks(self) -> List[Dict]:
        """
        准备阶段A任务：所有 scene_transition=1 的节点
        只参考主角图片，可全部并行生成
        
        Returns:
            任务列表
        """
        tasks = []
        
        for node_id, node in self.nodes.items():
            metadata = node.get('metadata', {}) if isinstance(node.get('metadata'), dict) else {}
            image_prompt = self._get_node_prompt(node)
            if not image_prompt:
                continue

            if self._is_image_already_generated(node_id):
                continue
            
            if self._is_new_scene(node_id):
                tasks.append({
                    'node_id': node_id,
                    'image_prompt': image_prompt,
                    'reference_image_url': self._get_node_reference_urls(node_id),
                    'is_scene_start': True,
                    'phase': 'A'
                })
        
        return tasks
    
    def prepare_phase_b_tasks(self) -> List[Dict]:
        """
        准备阶段B任务：所有 scene_transition=0 的节点
        参考主角图片 + 同场景中最近已生成图片的URL（双参考图）
        
        必须在阶段A完成并调用 set_node_image_url 后才能调用
        
        Returns:
            任务列表
        """
        tasks = []
        
        for node_id, node in self.nodes.items():
            metadata = node.get('metadata', {}) if isinstance(node.get('metadata'), dict) else {}
            image_prompt = self._get_node_prompt(node)
            if not image_prompt:
                continue

            if self._is_image_already_generated(node_id):
                continue
            
            if not self._is_new_scene(node_id):
                # 找同场景的参考图片
                scene_ref = self._find_same_scene_reference(node_id) if self.enable_scene_reference else None
                reference_urls = self._get_node_reference_urls(node_id, scene_ref)
                
                tasks.append({
                    'node_id': node_id,
                    'image_prompt': image_prompt,
                    'reference_image_url': reference_urls,
                    'is_scene_start': False,
                    'phase': 'B'
                })
        
        return tasks
    
    def prepare_generation_tasks(self) -> Tuple[List[Dict], List[Dict]]:
        """
        准备两阶段生成任务
        
        Returns:
            (阶段A任务列表, 阶段B任务列表)
        """
        phase_a = self.prepare_phase_a_tasks()
        # 注意：阶段B任务需要在阶段A完成后才能准备（需要已生成的图片URL）
        # 这里先返回空列表，由调用方在阶段A完成后调用 prepare_phase_b_tasks()
        return phase_a, []


def analyze_story_graph(story_graph_path: str, character_reference_url: str, recurring_character_refs: Optional[Dict[str, str]] = None, prompt_mode: str = "composed") -> IntelligentImageReferenceSelector:
    """
    分析故事图并创建参考图片选择器
    
    Args:
        story_graph_path: 故事图JSON文件路径
        character_reference_url: 主角参考图片URL
        
    Returns:
        IntelligentImageReferenceSelector实例
    """
    with open(story_graph_path, 'r', encoding='utf-8') as f:
        story_graph = json.load(f)
    
    return IntelligentImageReferenceSelector(story_graph, character_reference_url, recurring_character_refs, prompt_mode=prompt_mode)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        graph_path = sys.argv[1]
        character_url = sys.argv[2] if len(sys.argv) > 2 else "https://pub-141831e61e69445289222976a15b6fb3.r2.dev/Image_to_url_V2/--_----imagetourl.cloud-1770284168770-zpdnnd.png"
        
        selector = analyze_story_graph(graph_path, character_url)
        phase_a = selector.prepare_phase_a_tasks()
        
        print(f"\n📊 生成任务分析:")
        print(f"阶段A（新场景）: {len(phase_a)} 个节点")
        
        # 模拟阶段A完成
        for task in phase_a:
            selector.set_node_image_url(task['node_id'], f"https://example.com/{task['node_id']}.png")
        
        phase_b = selector.prepare_phase_b_tasks()
        print(f"阶段B（同场景）: {len(phase_b)} 个节点")
        print(f"总任务数: {len(phase_a) + len(phase_b)}")
        
        print(f"\n🎬 阶段A（新场景，只参考主角）:")
        for i, task in enumerate(phase_a, 1):
            print(f"  {i}. {task['node_id']}")
        
        print(f"\n🎬 阶段B（同场景，参考主角+前图）:")
        for i, task in enumerate(phase_b, 1):
            ref_count = len(task['reference_image_url']) if isinstance(task['reference_image_url'], list) else 1
            print(f"  {i}. {task['node_id']} (参考图数: {ref_count})")
