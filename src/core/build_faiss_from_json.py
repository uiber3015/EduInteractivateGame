"""
从 JSON 格式知识点文件构建多学科 FAISS 向量数据库

支持物理、生物、化学、地理/历史/通识等学科，合并为一个统一数据库。
带断点续传功能，中断后可从上次位置继续。

用法（从项目根目录运行）：
    python src/core/build_faiss_from_json.py
"""

import os
import sys
import json
import glob
import pickle
import time

import numpy as np
import faiss

# 把 src/core 加入路径以复用 FAISSKnowledgeBase
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_faiss_database import FAISSKnowledgeBase


# ──────────────────────────────────────────────
# 配置：4 个学科的 JSON 目录（相对项目根目录）
# ──────────────────────────────────────────────
SUBJECT_DIRS = [
    "data/rag_data/physics_knowledge_points",
    "data/rag_data/biology_knowledge_points",
    "data/rag_data/chemistry_knowledge_points",
    "data/rag_data/general_knowledge_points",
]

# 输出路径（覆盖原有数据库）
OUTPUT_INDEX   = "faiss_database/physics_knowledge.index"
OUTPUT_DOCS    = "faiss_database/physics_knowledge_docs.json"
CHECKPOINT_PKL = "faiss_database/build_json_checkpoint.pkl"


# ──────────────────────────────────────────────
# JSON → FAISS 文档块解析
# ──────────────────────────────────────────────
def parse_json_knowledge_point(filepath: str) -> list:
    """将单个 JSON 知识点文件解析为多个 FAISS 文档块"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    kp_name = data.get("name", os.path.splitext(os.path.basename(filepath))[0])
    docs = []

    # 1. 核心概念
    core_concept = data.get("core_concept", "")
    if core_concept:
        docs.append({
            "text": f"知识点：{kp_name}\n\n核心概念：\n{core_concept}",
            "metadata": {
                "knowledge_name": kp_name,
                "section_type": "core_concept",
                "section_name": "核心概念",
            },
        })

    # 2. 故事案例
    for case in data.get("story_cases", []):
        title   = case.get("title", "")
        content = case.get("content", "")
        application = case.get("application", "")
        text = f"知识点：{kp_name}\n\n{title}\n{content}"
        if application:
            text += f"\n\n应用说明：{application}"
        docs.append({
            "text": text,
            "metadata": {
                "knowledge_name": kp_name,
                "section_type": "story_case",
                "section_name": title,
            },
        })

    # 3. 学生误区
    for misc in data.get("misconceptions", []):
        title   = misc.get("title", "")
        wrong   = misc.get("wrong_understanding", "")
        correct = misc.get("correct_understanding", "")
        typical = misc.get("typical_error", "")
        exam    = misc.get("exam_example", "")
        text = f"知识点：{kp_name}\n\n{title}\n错误理解：{wrong}\n正确理解：{correct}"
        if typical:
            text += f"\n典型错误：{typical}"
        if exam:
            text += f"\n考试示例：{exam}"
        docs.append({
            "text": text,
            "metadata": {
                "knowledge_name": kp_name,
                "section_type": "misconception",
                "section_name": title,
            },
        })

    # 4. 错误选项设计
    for opt in data.get("error_options", []):
        scenario       = opt.get("scenario", "")
        correct_opt    = opt.get("correct_option", "")
        option_num     = opt.get("option_number", "")
        option_text    = opt.get("option_text", "")
        deception      = opt.get("deception", "")
        error_reason   = opt.get("error_reason", "")
        failure_analysis = opt.get("failure_analysis", "")
        text = (
            f"知识点：{kp_name}\n\n"
            f"场景：{scenario}\n"
            f"正确选项：{correct_opt}\n\n"
            f"错误选项{option_num}：{option_text}\n"
            f"迷惑性：{deception}\n"
            f"错误原因：{error_reason}\n"
            f"{failure_analysis}"
        )
        docs.append({
            "text": text,
            "metadata": {
                "knowledge_name": kp_name,
                "section_type": "error_option",
                "section_name": f"错误选项{option_num}",
            },
        })

    # 5. 教学建议
    teaching_advice = data.get("teaching_advice", "")
    if teaching_advice:
        docs.append({
            "text": f"知识点：{kp_name}\n\n教学建议：\n{teaching_advice}",
            "metadata": {
                "knowledge_name": kp_name,
                "section_type": "teaching_advice",
                "section_name": "教学建议",
            },
        })

    return docs


# ──────────────────────────────────────────────
# 主构建流程
# ──────────────────────────────────────────────
def build_from_json(subject_dirs=None):
    if subject_dirs is None:
        subject_dirs = SUBJECT_DIRS

    print("=" * 60)
    print("构建多学科 FAISS 知识库（从 JSON 文件，带断点续传）")
    print("=" * 60)

    # 收集所有 JSON 文件（排除 00_index.json 等汇总索引文件）
    all_files = []
    for d in subject_dirs:
        if not os.path.isdir(d):
            print(f"  ⚠ 目录不存在，跳过: {d}")
            continue
        files = sorted(glob.glob(os.path.join(d, "*.json")))
        files = [f for f in files if not os.path.basename(f).startswith("00_")]
        print(f"  {d}: {len(files)} 个文件")
        all_files.extend(files)

    if not all_files:
        print("❌ 未找到任何 JSON 文件，退出。")
        return

    print(f"\n总计 {len(all_files)} 个知识点文件")

    # 解析所有文档块
    print("\n解析文档中...")
    documents = []
    for filepath in all_files:
        try:
            docs = parse_json_knowledge_point(filepath)
            documents.extend(docs)
        except Exception as e:
            print(f"  ⚠ 解析失败 {filepath}: {e}")

    print(f"总计解析出 {len(documents)} 个文档块")

    # 加载断点缓存
    embeddings_cache = {}
    start_index = 0
    if os.path.exists(CHECKPOINT_PKL):
        resp = input("\n发现断点缓存，是否继续上次进度？(y/n): ")
        if resp.strip().lower() == "y":
            with open(CHECKPOINT_PKL, "rb") as f:
                embeddings_cache = pickle.load(f)
            start_index = len(embeddings_cache)
            print(f"✓ 已加载 {start_index} 个缓存的 embeddings")

    # 初始化 FAISSKnowledgeBase（提供 get_embedding + save）
    kb = FAISSKnowledgeBase()

    # 计算 embeddings（带自动保存缓存）
    print(f"\n开始从第 {start_index} 个文档计算 embedding...")
    for i in range(start_index, len(documents)):
        doc = documents[i]
        pct = i / len(documents) * 100
        print(
            f"\r进度: {i}/{len(documents)} ({pct:.1f}%) "
            f"- {doc['metadata']['knowledge_name'][:25]}...",
            end="", flush=True,
        )
        try:
            emb = kb.get_embedding(doc["text"])
            embeddings_cache[i] = emb

            # 每 20 个文档保存一次缓存
            if (i + 1) % 20 == 0:
                os.makedirs("faiss_database", exist_ok=True)
                with open(CHECKPOINT_PKL, "wb") as f:
                    pickle.dump(embeddings_cache, f)
                print(f"\n  ✓ 进度已保存: {i + 1}/{len(documents)}")

        except KeyboardInterrupt:
            print("\n\n中断，保存进度...")
            os.makedirs("faiss_database", exist_ok=True)
            with open(CHECKPOINT_PKL, "wb") as f:
                pickle.dump(embeddings_cache, f)
            print(f"✓ 已保存到第 {i} 个文档，下次运行可继续")
            return
        except Exception as e:
            print(f"\n❌ 文档 {i} 处理失败: {e}")
            os.makedirs("faiss_database", exist_ok=True)
            with open(CHECKPOINT_PKL, "wb") as f:
                pickle.dump(embeddings_cache, f)
            raise

    print(f"\n\n✓ 全部 {len(documents)} 个 embeddings 获取完成！")

    # 构建 FAISS 索引
    print("\n构建 FAISS 索引...")
    kb.documents = documents
    kb.index = faiss.IndexFlatL2(kb.dimension)
    embeddings_list = [embeddings_cache[i] for i in range(len(documents))]
    embeddings_array = np.array(embeddings_list, dtype="float32")
    kb.index.add(embeddings_array)
    print(f"✓ FAISS 索引构建完成，共 {kb.index.ntotal} 个向量")

    # 保存
    os.makedirs("faiss_database", exist_ok=True)
    kb.save(OUTPUT_INDEX, OUTPUT_DOCS)

    # 清理断点缓存
    if os.path.exists(CHECKPOINT_PKL):
        os.remove(CHECKPOINT_PKL)
        print("✓ 已清理断点缓存文件")

    print("\n" + "=" * 60)
    print("多学科 FAISS 知识库构建完成！")
    print("=" * 60)
    print(f"  索引文件: {OUTPUT_INDEX}")
    print(f"  文档文件: {OUTPUT_DOCS}")
    print(f"  总文档数: {len(documents)}")
    print(f"  向量维度: {kb.dimension}")


if __name__ == "__main__":
    try:
        build_from_json()
    except KeyboardInterrupt:
        print("\n\n构建已中断")
    except Exception as e:
        print(f"\n\n构建失败: {e}")
        import traceback
        traceback.print_exc()
