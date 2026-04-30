"""
构建FAISS向量数据库
将physics_knowledge_database.md中的知识点向量化并存储到FAISS数据库中
"""

import os
import json
import re
from typing import List, Dict
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class FAISSKnowledgeBase:
    def __init__(self, embedding_model="text-embedding-3-large"):
        """
        初始化FAISS知识库
        
        Args:
            embedding_model: OpenAI的embedding模型名称
        """
        # 配置多个API key用于轮换
        self.api_keys = [
            "sk-XPostJRTy2UO1Lajdof2alXLnmpgVJnmlVHhRFDyxT1l6QlH",
            "sk-O4jVyzh5v1Vee8V8QvKft0yA8Y5yc1rNfpNfCUPQSmT6mqjY"
        ]
        self.current_api_index = 0
        base_url = os.getenv("OPENAI_BASE_URL", "https://api2.aigcbest.top/v1")
        
        self.client = OpenAI(
            api_key=self.api_keys[self.current_api_index],
            base_url=base_url
        )
        self.base_url = base_url
        self.embedding_model = embedding_model
        # text-embedding-3-large的维度是3072
        self.dimension = 3072 if embedding_model == "text-embedding-3-large" else 1536
        self.index = None
        self.documents = []  # 存储文档内容和元数据
        self.api_call_count = 0  # 记录API调用次数
        
    def parse_knowledge_database(self, md_file_path: str) -> List[Dict]:
        """
        解析physics_knowledge_database.md文件
        将每个知识点的不同部分分块存储
        
        Returns:
            List[Dict]: 包含文本和元数据的文档列表
        """
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 只保留知识点部分，去掉数据库使用说明等
        # 找到"数据库使用说明"的位置，只处理之前的内容
        usage_section_pos = content.find('## 数据库使用说明')
        if usage_section_pos != -1:
            content = content[:usage_section_pos]
            print("✓ 已过滤掉数据库使用说明部分")
        
        documents = []
        
        # 使用正则表达式分割知识点
        # 匹配 "## 数字. 知识点名称"
        knowledge_pattern = r'## (\d+)\. (.+?)\n'
        knowledge_sections = re.split(knowledge_pattern, content)
        
        # knowledge_sections[0]是文件开头的目录部分，跳过
        # 之后每3个元素为一组：[编号, 标题, 内容]
        for i in range(1, len(knowledge_sections), 3):
            if i + 2 >= len(knowledge_sections):
                break
                
            knowledge_id = knowledge_sections[i]
            knowledge_name = knowledge_sections[i + 1].strip()
            knowledge_content = knowledge_sections[i + 2]
            
            # 跳过非知识点部分
            if not knowledge_id.isdigit():
                continue
            
            # 跳过重复的知识点（如果编号已经处理过）
            if any(doc['metadata']['knowledge_id'] == int(knowledge_id) and 
                   doc['metadata']['knowledge_name'] == knowledge_name 
                   for doc in documents):
                print(f"⚠ 跳过重复知识点: {knowledge_id}. {knowledge_name}")
                continue
            
            print(f"正在处理: {knowledge_id}. {knowledge_name}")
            
            # 1. 提取核心概念
            core_concept_match = re.search(
                r'### 核心概念\n(.*?)(?=\n###[^#]|\Z)', 
                knowledge_content, 
                re.DOTALL
            )
            if core_concept_match:
                core_concept = core_concept_match.group(1).strip()
                documents.append({
                    'text': f"知识点：{knowledge_name}\n\n核心概念：\n{core_concept}",
                    'metadata': {
                        'knowledge_id': int(knowledge_id),
                        'knowledge_name': knowledge_name,
                        'section_type': 'core_concept',
                        'section_name': '核心概念'
                    }
                })
            
            # 2. 提取教学故事案例
            story_section_match = re.search(
                r'### 教学故事案例\s*\n(.*?)(?=\n###[^#]|\Z)', 
                knowledge_content, 
                re.DOTALL
            )
            if story_section_match:
                story_section = story_section_match.group(1)
                # 分割每个案例 - 修复正则表达式
                story_cases = re.findall(
                    r'#### (案例\d+[：:].+?)\n(.*?)(?=\n####|\n###[^#]|\Z)', 
                    story_section, 
                    re.DOTALL
                )
                if story_cases:
                    for case_title, case_content in story_cases:
                        documents.append({
                            'text': f"知识点：{knowledge_name}\n\n{case_title}\n{case_content.strip()}",
                            'metadata': {
                                'knowledge_id': int(knowledge_id),
                                'knowledge_name': knowledge_name,
                                'section_type': 'story_case',
                                'section_name': case_title
                            }
                        })
                else:
                    print(f"  ⚠ {knowledge_name}: 未找到故事案例")
            
            # 3. 提取学生常见误区
            misconception_section_match = re.search(
                r'### 学生常见误区\s*\n(.*?)(?=\n###[^#]|\Z)', 
                knowledge_content, 
                re.DOTALL
            )
            if misconception_section_match:
                misconception_section = misconception_section_match.group(1)
                # 分割每个误区 - 修复正则表达式
                misconceptions = re.findall(
                    r'#### (误区\d+[：:].+?)\n(.*?)(?=\n####|\n###[^#]|\Z)', 
                    misconception_section, 
                    re.DOTALL
                )
                if misconceptions:
                    for misc_title, misc_content in misconceptions:
                        documents.append({
                            'text': f"知识点：{knowledge_name}\n\n{misc_title}\n{misc_content.strip()}",
                            'metadata': {
                                'knowledge_id': int(knowledge_id),
                                'knowledge_name': knowledge_name,
                                'section_type': 'misconception',
                                'section_name': misc_title
                            }
                        })
                else:
                    print(f"  ⚠ {knowledge_name}: 未找到误区")
            
            # 4. 提取典型错误选项设计
            error_option_section_match = re.search(
                r'### 典型错误选项设计\s*\n(.*?)(?=\n###[^#]|\Z)', 
                knowledge_content, 
                re.DOTALL
            )
            if error_option_section_match:
                error_option_section = error_option_section_match.group(1)
                # 提取场景和选项
                scenario_match = re.search(r'\*\*场景\*\*[：:]\s*(.+?)\n', error_option_section)
                correct_option_match = re.search(r'\*\*正确选项\*\*[：:]\s*(.+?)\n', error_option_section)
                
                # 分割每个错误选项 - 修复正则表达式
                error_options = re.findall(
                    r'\*\*错误选项(\d+)\*\*[（(]基于误区\d+[）)][：:]\s*(.+?)\n(.*?)(?=\*\*错误选项|\*\*场景|\n###[^#]|\Z)', 
                    error_option_section, 
                    re.DOTALL
                )
                
                scenario = scenario_match.group(1).strip() if scenario_match else ""
                correct_option = correct_option_match.group(1).strip() if correct_option_match else ""
                
                if error_options:
                    for option_num, option_text, option_analysis in error_options:
                        documents.append({
                            'text': f"知识点：{knowledge_name}\n\n场景：{scenario}\n正确选项：{correct_option}\n\n错误选项{option_num}：{option_text}\n{option_analysis.strip()}",
                            'metadata': {
                                'knowledge_id': int(knowledge_id),
                                'knowledge_name': knowledge_name,
                                'section_type': 'error_option',
                                'section_name': f'错误选项{option_num}'
                            }
                        })
                else:
                    print(f"  ⚠ {knowledge_name}: 未找到错误选项")
            
            # 5. 提取教学建议
            teaching_advice_match = re.search(
                r'### 教学建议\s*\n(.*?)(?=\n###[^#]|---|\Z)', 
                knowledge_content, 
                re.DOTALL
            )
            if teaching_advice_match:
                teaching_advice = teaching_advice_match.group(1).strip()
                if teaching_advice:
                    documents.append({
                        'text': f"知识点：{knowledge_name}\n\n教学建议：\n{teaching_advice}",
                        'metadata': {
                            'knowledge_id': int(knowledge_id),
                            'knowledge_name': knowledge_name,
                            'section_type': 'teaching_advice',
                            'section_name': '教学建议'
                        }
                    })
        
        print(f"\n总共解析出 {len(documents)} 个文档块")
        return documents
    
    def switch_api_key(self):
        """切换到下一个API key"""
        self.current_api_index = (self.current_api_index + 1) % len(self.api_keys)
        self.client = OpenAI(
            api_key=self.api_keys[self.current_api_index],
            base_url=self.base_url
        )
        print(f"\n  → 已切换到备用API key #{self.current_api_index + 1}")
    
    def get_embedding(self, text: str, max_retries: int = 5) -> np.ndarray:
        """
        获取文本的embedding向量（带重试机制和API轮换）
        
        Args:
            text: 输入文本
            max_retries: 最大重试次数
            
        Returns:
            np.ndarray: embedding向量
        """
        import time
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text,
                    timeout=30.0  # 30秒超时
                )
                self.api_call_count += 1
                
                # 每调用20次API，等待更长时间避免频率限制
                if self.api_call_count % 20 == 0:
                    print(f"\n  ⏸ 已调用{self.api_call_count}次API，休息5秒...")
                    time.sleep(5)
                
                return np.array(response.data[0].embedding, dtype='float32')
                
            except Exception as e:
                error_msg = str(e)
                print(f"\n  ⚠ API调用失败 (尝试 {attempt + 1}/{max_retries})")
                print(f"  错误: {error_msg[:150]}")
                
                if attempt < max_retries - 1:
                    # 如果是连接错误或频率限制，尝试切换API key
                    if "Connection error" in error_msg or "rate" in error_msg.lower() or "limit" in error_msg.lower():
                        if attempt == 0:  # 第一次失败就切换
                            self.switch_api_key()
                            wait_time = 3
                        else:
                            wait_time = (attempt + 1) * 5  # 递增等待时间：5秒、10秒、15秒...
                    else:
                        wait_time = (attempt + 1) * 3
                    
                    print(f"  ⏳ {wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ API调用失败，已达最大重试次数")
                    # 最后一次尝试切换API key
                    if len(self.api_keys) > 1:
                        print(f"  → 尝试切换到另一个API key...")
                        self.switch_api_key()
                        time.sleep(5)
                        try:
                            response = self.client.embeddings.create(
                                model=self.embedding_model,
                                input=text,
                                timeout=30.0
                            )
                            return np.array(response.data[0].embedding, dtype='float32')
                        except:
                            pass
                    raise
    
    def build_index(self, documents: List[Dict]):
        """
        构建FAISS索引
        
        Args:
            documents: 文档列表
        """
        import time
        
        print("\n开始构建FAISS索引...")
        self.documents = documents
        
        # 创建FAISS索引（使用L2距离）
        self.index = faiss.IndexFlatL2(self.dimension)
        
        embeddings = []
        total = len(documents)
        start_time = time.time()
        
        for i, doc in enumerate(documents):
            # 显示详细进度
            print(f"\r正在向量化: {i}/{total} ({i/total*100:.1f}%) - {doc['metadata']['knowledge_name'][:20]}...", end='', flush=True)
            
            try:
                embedding = self.get_embedding(doc['text'])
                embeddings.append(embedding)
                
                # 每10个显示一次详细信息
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    remaining = avg_time * (total - i - 1)
                    print(f"\n  ✓ 已完成 {i+1}/{total}, 预计剩余时间: {remaining:.1f}秒")
                    
            except Exception as e:
                print(f"\n  ❌ 文档 {i} 向量化失败: {e}")
                print(f"  文档内容: {doc['text'][:100]}...")
                raise
        
        print(f"\n\n✓ 所有文档向量化完成！")
        
        # 转换为numpy数组并添加到索引
        print("正在构建FAISS索引...")
        embeddings_array = np.array(embeddings)
        self.index.add(embeddings_array)
        
        print(f"✓ FAISS索引构建完成！共 {self.index.ntotal} 个向量")
        print(f"总耗时: {time.time() - start_time:.1f}秒")
    
    def save(self, index_path: str, documents_path: str):
        """
        保存FAISS索引和文档数据
        
        Args:
            index_path: FAISS索引保存路径
            documents_path: 文档数据保存路径
        """
        # 保存FAISS索引
        faiss.write_index(self.index, index_path)
        print(f"FAISS索引已保存到: {index_path}")
        
        # 保存文档数据
        with open(documents_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        print(f"文档数据已保存到: {documents_path}")
    
    def load(self, index_path: str, documents_path: str):
        """
        加载FAISS索引和文档数据
        
        Args:
            index_path: FAISS索引路径
            documents_path: 文档数据路径
        """
        # 加载FAISS索引
        self.index = faiss.read_index(index_path)
        print(f"FAISS索引已加载: {self.index.ntotal} 个向量")
        
        # 加载文档数据
        with open(documents_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        print(f"文档数据已加载: {len(self.documents)} 个文档")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回top-k个结果
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        # 获取查询的embedding
        query_embedding = self.get_embedding(query)
        query_embedding = np.array([query_embedding])
        
        # 搜索
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 构建结果
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                result = {
                    'rank': i + 1,
                    'distance': float(dist),
                    'document': self.documents[idx]
                }
                results.append(result)
        
        return results


def main():
    """
    主函数：构建FAISS知识库
    """
    print("=" * 60)
    print("开始构建FAISS物理知识库")
    print("=" * 60)
    
    # 1. 初始化知识库
    kb = FAISSKnowledgeBase()
    
    # 2. 解析知识库文件
    md_file_path = "physics_knowledge_database.md"
    if not os.path.exists(md_file_path):
        print(f"错误：找不到文件 {md_file_path}")
        return
    
    documents = kb.parse_knowledge_database(md_file_path)
    
    # 3. 构建FAISS索引
    try:
        kb.build_index(documents)
    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断")
        return
    except Exception as e:
        print(f"\n\n❌ 构建失败: {e}")
        return
    
    # 4. 保存索引和文档
    output_dir = "faiss_database"
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = os.path.join(output_dir, "physics_knowledge.index")
    documents_path = os.path.join(output_dir, "physics_knowledge_docs.json")
    
    kb.save(index_path, documents_path)
    
    print("\n" + "=" * 60)
    print("FAISS知识库构建完成！")
    print("=" * 60)
    print(f"\n索引文件: {index_path}")
    print(f"文档文件: {documents_path}")
    print(f"\n总文档数: {len(documents)}")
    print(f"向量维度: {kb.dimension}")
    print(f"API调用总次数: {kb.api_call_count}")
    
    # 5. 测试搜索功能
    print("\n" + "=" * 60)
    print("测试搜索功能")
    print("=" * 60)
    
    test_queries = [
        "如何利用杠杆原理省力？",
        "学生对电路串联并联有什么误解？",
        "光的折射在生活中的应用"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        results = kb.search(query, top_k=3)
        for result in results:
            print(f"\n  排名 {result['rank']} (距离: {result['distance']:.4f})")
            print(f"  知识点: {result['document']['metadata']['knowledge_name']}")
            print(f"  类型: {result['document']['metadata']['section_type']}")
            print(f"  内容预览: {result['document']['text'][:100]}...")


if __name__ == "__main__":
    main()
