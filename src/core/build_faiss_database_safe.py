"""
安全构建FAISS向量数据库（带断点续传）
如果构建过程中断，可以从上次的位置继续
"""

import os
import json
import pickle
from build_faiss_database import FAISSKnowledgeBase

def build_with_checkpoint():
    """带检查点的构建过程"""
    
    checkpoint_file = "faiss_database/build_checkpoint.pkl"
    embeddings_file = "faiss_database/embeddings_cache.pkl"
    
    print("=" * 60)
    print("安全构建FAISS物理知识库（带断点续传）")
    print("=" * 60)
    
    # 1. 初始化知识库
    kb = FAISSKnowledgeBase()
    
    # 2. 解析知识库文件
    md_file_path = "physics_knowledge_database.md"
    if not os.path.exists(md_file_path):
        print(f"错误：找不到文件 {md_file_path}")
        return
    
    documents = kb.parse_knowledge_database(md_file_path)
    print(f"\n总共解析出 {len(documents)} 个文档块")
    
    # 3. 检查是否有缓存的embeddings
    embeddings_cache = {}
    start_index = 0
    
    if os.path.exists(embeddings_file):
        response = input("\n发现缓存的embeddings，是否继续上次的进度？(y/n): ")
        if response.lower() == 'y':
            with open(embeddings_file, 'rb') as f:
                embeddings_cache = pickle.load(f)
            start_index = len(embeddings_cache)
            print(f"✓ 已加载 {start_index} 个缓存的embeddings")
    
    # 4. 构建embeddings（带缓存）
    print(f"\n开始从第 {start_index} 个文档构建...")
    
    import time
    import numpy as np
    
    for i in range(start_index, len(documents)):
        doc = documents[i]
        
        # 显示进度
        progress = (i / len(documents)) * 100
        print(f"\r进度: {i}/{len(documents)} ({progress:.1f}%) - {doc['metadata']['knowledge_name'][:30]}...", 
              end='', flush=True)
        
        try:
            # 获取embedding
            embedding = kb.get_embedding(doc['text'])
            embeddings_cache[i] = embedding
            
            # 每10个保存一次缓存
            if (i + 1) % 10 == 0:
                os.makedirs("faiss_database", exist_ok=True)
                with open(embeddings_file, 'wb') as f:
                    pickle.dump(embeddings_cache, f)
                print(f"\n  ✓ 已保存进度: {i+1}/{len(documents)}")
            
            # 避免请求过快
            time.sleep(0.5)
            
        except KeyboardInterrupt:
            print("\n\n用户中断，保存当前进度...")
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings_cache, f)
            print(f"✓ 进度已保存，下次运行将从第 {i} 个文档继续")
            return
        except Exception as e:
            print(f"\n❌ 文档 {i} 处理失败: {e}")
            print(f"文档内容: {doc['text'][:100]}...")
            # 保存进度
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings_cache, f)
            print(f"✓ 进度已保存")
            raise
    
    print("\n\n✓ 所有embeddings获取完成！")
    
    # 5. 构建FAISS索引
    print("\n构建FAISS索引...")
    kb.documents = documents
    
    import faiss
    kb.index = faiss.IndexFlatL2(kb.dimension)
    
    # 转换embeddings为数组
    embeddings_list = [embeddings_cache[i] for i in range(len(documents))]
    embeddings_array = np.array(embeddings_list)
    kb.index.add(embeddings_array)
    
    print(f"✓ FAISS索引构建完成！共 {kb.index.ntotal} 个向量")
    
    # 6. 保存索引和文档
    output_dir = "faiss_database"
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = os.path.join(output_dir, "physics_knowledge.index")
    documents_path = os.path.join(output_dir, "physics_knowledge_docs.json")
    
    kb.save(index_path, documents_path)
    
    # 7. 清理缓存文件
    if os.path.exists(embeddings_file):
        os.remove(embeddings_file)
        print("✓ 已清理缓存文件")
    
    print("\n" + "=" * 60)
    print("FAISS知识库构建完成！")
    print("=" * 60)
    print(f"\n索引文件: {index_path}")
    print(f"文档文件: {documents_path}")
    print(f"\n总文档数: {len(documents)}")
    print(f"向量维度: {kb.dimension}")


if __name__ == "__main__":
    try:
        build_with_checkpoint()
    except KeyboardInterrupt:
        print("\n\n构建已中断")
    except Exception as e:
        print(f"\n\n构建失败: {e}")
        import traceback
        traceback.print_exc()
