#!/usr/bin/env python3
"""
故事图数据结构
用于存储故事节点和它们之间的关系
"""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum


class NodeType(Enum):
    """节点类型枚举"""
    NORMAL = "normal"  # 普通故事节点
    DECISION = "decision"  # 决策点节点
    ENDING = "ending"  # 结局节点
    EP_START = "ep_start"  # 章节起始节点
    EP_END = "ep_end"  # 章节结束节点
    FATAL = "fatal"  # 致命结局节点


@dataclass
class StoryNode:
    """故事节点类"""
    id: str  # 节点唯一标识
    content: str  # 节点内容
    node_type: NodeType = NodeType.NORMAL  # 节点类型
    choices: List[str] = field(default_factory=list)  # 决策选项（仅对决策点节点有效）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def __post_init__(self):
        """后初始化处理"""
        if not self.id:
            raise ValueError("节点ID不能为空")
        if not self.content:
            raise ValueError("节点内容不能为空")
    
    def add_choice(self, choice: str) -> None:
        """添加决策选项"""
        if self.node_type == NodeType.DECISION and choice:
            self.choices.append(choice)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据"""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)


class StoryGraph:
    """故事图类"""
    
    def __init__(self):
        """初始化故事图"""
        self.nodes: Dict[str, StoryNode] = {}  # 节点ID到节点的映射
        self.edges: Dict[str, List[str]] = {}  # 节点ID到其后继节点ID列表的映射
        self.reverse_edges: Dict[str, List[str]] = {}  # 节点ID到其前驱节点ID列表的映射
        self.start_node_id: Optional[str] = None  # 起始节点ID
        self.ending_node_ids: Set[str] = set()  # 结局节点ID集合
    
    def add_node(self, node: StoryNode) -> None:
        """添加节点"""
        if node.id in self.nodes:
            raise ValueError(f"节点ID '{node.id}' 已存在")
        
        self.nodes[node.id] = node
        self.edges[node.id] = []
        self.reverse_edges[node.id] = []
        
        # 如果是第一个节点，设置为起始节点
        if self.start_node_id is None:
            self.start_node_id = node.id
        
        # 如果是结局节点，添加到结局节点集合
        if node.node_type == NodeType.ENDING:
            self.ending_node_ids.add(node.id)
    
    def add_edge(self, from_node_id: str, to_node_id: str) -> None:
        """添加边"""
        if from_node_id not in self.nodes:
            raise ValueError(f"源节点ID '{from_node_id}' 不存在")
        if to_node_id not in self.nodes:
            raise ValueError(f"目标节点ID '{to_node_id}' 不存在")
        if to_node_id in self.edges[from_node_id]:
            return  # 边已存在，不重复添加
        
        self.edges[from_node_id].append(to_node_id)
        self.reverse_edges[to_node_id].append(from_node_id)
    
    def get_node(self, node_id: str) -> Optional[StoryNode]:
        """获取节点"""
        return self.nodes.get(node_id)
    
    def get_successors(self, node_id: str) -> List[str]:
        """获取后继节点ID列表"""
        return self.edges.get(node_id, [])
    
    def get_predecessors(self, node_id: str) -> List[str]:
        """获取前驱节点ID列表"""
        return self.reverse_edges.get(node_id, [])
    
    def get_start_node(self) -> Optional[StoryNode]:
        """获取起始节点"""
        if self.start_node_id:
            return self.nodes.get(self.start_node_id)
        return None
    
    def get_ending_nodes(self) -> List[StoryNode]:
        """获取所有结局节点"""
        return [self.nodes[node_id] for node_id in self.ending_node_ids if node_id in self.nodes]
    
    def get_all_nodes(self) -> List[StoryNode]:
        """获取所有节点"""
        return list(self.nodes.values())
    
    def get_all_edges(self) -> Dict[str, List[str]]:
        """获取所有边"""
        return {node_id: successors for node_id, successors in self.edges.items() if successors}
    
    def set_start_node(self, node_id: str) -> None:
        """设置起始节点"""
        if node_id not in self.nodes:
            raise ValueError(f"节点ID '{node_id}' 不存在")
        self.start_node_id = node_id
    
    def mark_as_ending(self, node_id: str) -> None:
        """标记节点为结局节点"""
        if node_id not in self.nodes:
            raise ValueError(f"节点ID '{node_id}' 不存在")
        self.nodes[node_id].node_type = NodeType.ENDING
        self.ending_node_ids.add(node_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """将图转换为字典格式"""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "content": node.content,
                    "type": node.node_type.value,
                    "metadata": node.metadata
                }
                for node in self.nodes.values()
            ],
            "edges": self.get_all_edges(),
            "start_node_id": self.start_node_id,
            "ending_node_ids": list(self.ending_node_ids)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoryGraph':
        """从字典创建图"""
        graph = cls()
        
        # 添加节点
        for node_data in data.get("nodes", []):
            node = StoryNode(
                id=node_data["id"],
                content=node_data["content"],
                node_type=NodeType(node_data["type"]),
                metadata=node_data.get("metadata", {})
            )
            graph.add_node(node)
        
        # 添加边
        for from_node_id, to_node_ids in data.get("edges", {}).items():
            for to_node_id in to_node_ids:
                graph.add_edge(from_node_id, to_node_id)
        
        # 设置起始节点
        start_node_id = data.get("start_node_id")
        if start_node_id:
            graph.set_start_node(start_node_id)
        
        # 标记结局节点
        for node_id in data.get("ending_node_ids", []):
            graph.mark_as_ending(node_id)
        
        return graph
    
    def validate(self) -> bool:
        """验证图的有效性"""
        # 检查是否有节点
        if not self.nodes:
            return False
        
        # 检查是否有起始节点
        if not self.start_node_id or self.start_node_id not in self.nodes:
            return False
        
        # 检查所有边是否有效
        for from_node_id, to_node_ids in self.edges.items():
            if from_node_id not in self.nodes:
                return False
            for to_node_id in to_node_ids:
                if to_node_id not in self.nodes:
                    return False
        
        # 检查所有反向边是否有效
        for to_node_id, from_node_ids in self.reverse_edges.items():
            if to_node_id not in self.nodes:
                return False
            for from_node_id in from_node_ids:
                if from_node_id not in self.nodes:
                    return False
        
        # 检查结局节点是否有效
        for node_id in self.ending_node_ids:
            if node_id not in self.nodes:
                return False
        
        return True
    
    def __str__(self) -> str:
        """返回图的字符串表示"""
        node_count = len(self.nodes)
        edge_count = sum(len(successors) for successors in self.edges.values())
        return f"StoryGraph(nodes={node_count}, edges={edge_count}, start_node={self.start_node_id}, endings={len(self.ending_node_ids)})"
    
    def __repr__(self) -> str:
        """返回图的详细字符串表示"""
        return self.__str__()