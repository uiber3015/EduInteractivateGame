"""
FAISS检索器
用于从FAISS数据库中检索相关知识
"""

import os
import json
import numpy as np
import faiss
from openai import OpenAI
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class FAISSRetriever:
    """FAISS检索器类"""
    
    def __init__(self, index_path: str, documents_path: str, embedding_model: str = "text-embedding-3-large"):
        """
        初始化检索器
        
        Args:
            index_path: FAISS索引文件路径
            documents_path: 文档数据文件路径
            embedding_model: OpenAI embedding模型名称
        """
        # 配置自定义API
        api_key = os.getenv("OPENAI_API_KEY", "sk-XPostJRTy2UO1Lajdof2alXLnmpgVJnmlVHhRFDyxT1l6QlH")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api2.aigcbest.top/v1")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.embedding_model = embedding_model
        
        # 加载FAISS索引
        self.index = faiss.read_index(index_path)
        print(f"✓ FAISS索引已加载: {self.index.ntotal} 个向量")
        
        # 加载文档数据
        with open(documents_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        print(f"✓ 文档数据已加载: {len(self.documents)} 个文档")

        # 构建倒排索引：(knowledge_name, section_type) -> [doc_idx]
        self._meta_index = {}
        for i, doc in enumerate(self.documents):
            kn = doc['metadata'].get('knowledge_name', '')
            st = doc['metadata'].get('section_type', '')
            self._meta_index.setdefault((kn, st), []).append(i)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """获取文本的embedding向量"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding, dtype='float32')
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               section_type: Optional[str] = None,
               knowledge_name: Optional[str] = None) -> List[Dict]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回top-k个结果
            section_type: 过滤特定类型的文档（可选）
                - 'core_concept': 核心概念
                - 'story_case': 故事案例
                - 'misconception': 学生误区
                - 'error_option': 错误选项
                - 'teaching_advice': 教学建议
            knowledge_name: 过滤特定知识点（可选）
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        # 获取查询的embedding
        query_embedding = self.get_embedding(query)
        query_embedding = np.array([query_embedding])
        
        # 当指定了 knowledge_name 时，走倒排索引精确查找（避免 FAISS 语义候选遗漏）
        if knowledge_name:
            if section_type:
                candidate_indices = self._meta_index.get((knowledge_name, section_type), [])
            else:
                candidate_indices = []
                for (kn, st), idx_list in self._meta_index.items():
                    if kn == knowledge_name:
                        candidate_indices.extend(idx_list)

            if candidate_indices:
                # 取前 top_k 条直接返回（数量少，无需重排）
                results = []
                for idx in candidate_indices[:top_k]:
                    doc = self.documents[idx]
                    results.append({
                        'distance': 0.0,
                        'similarity': 1.0,
                        'text': doc['text'],
                        'metadata': doc['metadata']
                    })
                return results
            # 倒排索引无结果则走 fallback（调用方负责降级逻辑）
            return []

        # 无 knowledge_name 过滤时，用 FAISS 做语义检索
        total_docs = self.index.ntotal
        search_k = min(max(top_k * 20, 100), total_docs) if section_type else top_k
        distances, indices = self.index.search(query_embedding, search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                if section_type and doc['metadata']['section_type'] != section_type:
                    continue
                results.append({
                    'distance': float(dist),
                    'similarity': 1 / (1 + dist),
                    'text': doc['text'],
                    'metadata': doc['metadata']
                })
                if len(results) >= top_k:
                    break

        return results
    
    def retrieve_for_story_generation(self, knowledge_points: List[str], top_k: int = 3) -> Dict[str, List[Dict]]:
        """
        为故事生成检索相关知识
        
        Args:
            knowledge_points: 知识点列表
            top_k: 每个知识点返回top-k个结果
            
        Returns:
            Dict: 每个知识点的检索结果
        """
        results = {}
        for kp in knowledge_points:
            query = f"关于{kp}的教学故事案例和应用场景"

            # 检索核心概念（先精确匹配，空则降级语义检索）
            core_concepts = self.search(query, top_k=1, section_type='core_concept', knowledge_name=kp)
            if not core_concepts:
                core_concepts = self.search(query, top_k=1, section_type='core_concept')

            # 检索故事案例（先精确匹配，空则降级语义检索）
            story_cases = self.search(query, top_k=top_k, section_type='story_case', knowledge_name=kp)
            if not story_cases:
                story_cases = self.search(query, top_k=top_k, section_type='story_case')

            results[kp] = {
                'core_concept': core_concepts[0] if core_concepts else None,
                'story_cases': story_cases
            }
        
        return results
    
    def retrieve_for_choice_generation(self, knowledge_point: str, scenario: str, top_k: int = 3) -> Dict:
        """
        为选项生成检索相关知识
        
        Args:
            knowledge_point: 知识点名称
            scenario: 场景描述
            top_k: 返回top-k个结果
            
        Returns:
            Dict: 检索结果
        """
        query = f"{knowledge_point}：{scenario}，学生常见的错误理解和误区"
        
        # 检索学生误区（先精确匹配，空则降级语义检索）
        misconceptions = self.search(query, top_k=top_k, section_type='misconception', knowledge_name=knowledge_point)
        if not misconceptions:
            misconceptions = self.search(query, top_k=top_k, section_type='misconception')

        # 检索错误选项设计（先精确匹配，空则降级语义检索）
        error_options = self.search(query, top_k=top_k, section_type='error_option', knowledge_name=knowledge_point)
        if not error_options:
            error_options = self.search(query, top_k=top_k, section_type='error_option')

        return {
            'misconceptions': misconceptions,
            'error_options': error_options,
            'has_specific_data': len(misconceptions) > 0 or len(error_options) > 0,
            'scenario': scenario,
            'knowledge_point': knowledge_point
        }

    def retrieve_teaching_advice(self, knowledge_point: str, top_k: int = 2) -> List[Dict]:
        """
        为解答生成检索教学建议
        
        Args:
            knowledge_point: 知识点名称
            top_k: 返回top-k个结果
            
        Returns:
            List[Dict]: 教学建议列表
        """
        query = f"{knowledge_point} 教学建议 课堂讲解 引导学生理解"
        results = self.search(query, top_k=top_k, section_type='teaching_advice', knowledge_name=knowledge_point)
        if not results:
            results = self.search(query, top_k=top_k, section_type='teaching_advice')
        return results

    def retrieve_bridge_examples(self, knowledge_point_a: str, knowledge_point_b: str, top_k: int = 2) -> Dict:
        """
        为相邻知识点检索桥接案例，用于增强剧情过渡的自然性
        
        Args:
            knowledge_point_a: 前一知识点
            knowledge_point_b: 后一知识点
            top_k: 每个知识点检索的案例数
            
        Returns:
            Dict: 桥接检索结果
        """
        query_a = f"{knowledge_point_a} 生活场景 应用 故事案例"
        query_b = f"{knowledge_point_b} 生活场景 应用 故事案例"

        stories_a = self.search(query_a, top_k=top_k, section_type='story_case', knowledge_name=knowledge_point_a)
        stories_b = self.search(query_b, top_k=top_k, section_type='story_case', knowledge_name=knowledge_point_b)

        return {
            'from_knowledge_point': knowledge_point_a,
            'to_knowledge_point': knowledge_point_b,
            'from_story_cases': stories_a,
            'to_story_cases': stories_b
        }
    
    def retrieve_for_coherence_check(self, knowledge_point: str) -> Dict:
        """
        为连贯性检查检索核心概念
        
        Args:
            knowledge_point: 知识点名称
            
        Returns:
            Dict: 核心概念
        """
        query = f"{knowledge_point}的核心概念和定义"
        results = self.search(query, top_k=1, section_type='core_concept', knowledge_name=knowledge_point)
        
        return results[0] if results else None
    
    def get_all_knowledge_points(self) -> List[str]:
        """
        获取所有知识点名称
        
        Returns:
            List[str]: 知识点名称列表
        """
        knowledge_points = set()
        for doc in self.documents:
            knowledge_points.add(doc['metadata']['knowledge_name'])
        
        return sorted(list(knowledge_points))
    
    def get_knowledge_statistics(self) -> Dict:
        """
        获取知识库统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = {
            'total_documents': len(self.documents),
            'total_vectors': self.index.ntotal,
            'knowledge_points': {},
            'section_types': {}
        }
        
        for doc in self.documents:
            kp = doc['metadata']['knowledge_name']
            st = doc['metadata']['section_type']
            
            if kp not in stats['knowledge_points']:
                stats['knowledge_points'][kp] = 0
            stats['knowledge_points'][kp] += 1
            
            if st not in stats['section_types']:
                stats['section_types'][st] = 0
            stats['section_types'][st] += 1
        
        return stats


def test_retriever():
    """测试检索器功能"""
    print("=" * 60)
    print("测试FAISS检索器")
    print("=" * 60)
    
    # 初始化检索器
    retriever = FAISSRetriever(
        index_path="faiss_database/physics_knowledge.index",
        documents_path="faiss_database/physics_knowledge_docs.json"
    )
    
    # 1. 测试基本搜索
    print("\n【测试1：基本搜索】")
    query = "如何利用杠杆原理省力？"
    print(f"查询: {query}")
    results = retriever.search(query, top_k=3)
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i} (相似度: {result['similarity']:.4f})")
        print(f"知识点: {result['metadata']['knowledge_name']}")
        print(f"类型: {result['metadata']['section_type']}")
        print(f"内容: {result['text'][:150]}...")
    
    # 2. 测试故事生成检索
    print("\n\n【测试2：故事生成检索】")
    knowledge_points = ["杠杆原理", "摩擦力"]
    results = retriever.retrieve_for_story_generation(knowledge_points, top_k=2)
    for kp, data in results.items():
        print(f"\n知识点: {kp}")
        if data['core_concept']:
            print(f"  核心概念: {data['core_concept']['text'][:100]}...")
        print(f"  故事案例数: {len(data['story_cases'])}")
    
    # 3. 测试选项生成检索
    print("\n\n【测试3：选项生成检索】")
    results = retriever.retrieve_for_choice_generation(
        knowledge_point="杠杆原理",
        scenario="需要撬起一块大石头",
        top_k=2
    )
    print(f"误区数: {len(results['misconceptions'])}")
    print(f"错误选项数: {len(results['error_options'])}")
    
    # 4. 获取统计信息
    print("\n\n【测试4：统计信息】")
    stats = retriever.get_knowledge_statistics()
    print(f"总文档数: {stats['total_documents']}")
    print(f"总向量数: {stats['total_vectors']}")
    print(f"知识点数: {len(stats['knowledge_points'])}")
    print(f"\n各类型文档数:")
    for section_type, count in stats['section_types'].items():
        print(f"  {section_type}: {count}")


if __name__ == "__main__":
    test_retriever()
