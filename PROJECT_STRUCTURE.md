# EduInteractiveGame 项目结构

## 目录结构

```
EduInteractivateGame/
├── src/                                  # 核心源代码
│   ├── __init__.py
│   ├── one_click_start.py                # 主入口：一键启动脚本
│   ├── cot_full_pipeline.py              # 核心 Pipeline
│   ├── core/                             # 故事图与检索核心模块
│   │   ├── StoryGraph.py
│   │   ├── faiss_retriever.py
│   │   ├── build_faiss_database.py
│   │   └── build_faiss_database_safe.py
│   ├── generation/                       # CoT 故事生成与转换模块
│   │   ├── cot_web_story_generator_v2.py
│   │   └── cot_to_storygraph_converter.py
│   ├── image/                            # 图像提示词与图像生成模块
│   ├── image_consistency/                # 角色/场景一致性处理模块
│   ├── visualization/                    # 前端可视化与 Flask 服务
│   │   ├── story_visualizer.py
│   │   └── templates/
│   │       ├── index.html
│   │       ├── interactive_story.html
│   │       └── story_graph_visualization.html
│   └── utils/                            # 配置与辅助工具
├── data/                                 # 核心数据目录
│   ├── knowledge.txt                     # 知识点列表
│   ├── prompts.txt                       # 提示词模板
│   └── rag_data/                         # RAG 知识库原始数据
├── output/                               # 预留输出目录
├── .env                                  # 环境变量配置
├── requirements.txt                      # Python 运行依赖
└── PROJECT_STRUCTURE.md                  # 当前打包版结构说明
```

## 使用方法

### 1. 快速开始（推荐）
```bash
python src/one_click_start.py
```

### 2. 使用核心Pipeline
```python
from src.cot_full_pipeline import cot_pipeline

result = cot_pipeline(
    knowledge_points=["杠杆原理", "摩擦力"],
    scenario="小明帮助爷爷修理老房子",
    num_questions=3,
    generate_images=True
)
```

### 3. 单独启动可视化
```python
from src.visualization.story_visualizer import run_story_visualizer

run_story_visualizer(
    custom_story_graph_path="path/to/story_graph.json",
    port=5000
)
```

## 打包内容说明

1. **保留核心代码**：包含执行代码、前端模板、图像生成与可视化逻辑
2. **保留核心数据**：包含 `data/rag_data`、`knowledge.txt`、`prompts.txt`
3. **预留输出目录**：根目录下已创建空的 `output/` 用于放置后续生成结果
4. **去除非核心内容**：未包含旧归档、历史输出、缓存和其他冗余目录

## 注意事项

- 所有代码现在位于 `src/` 目录下
- 本打包版不包含 `archive/`、历史 `output/`、`docs/` 等非核心目录
- 运行脚本时需要从项目根目录执行
- 环境变量配置位于根目录 `.env`
- 如果需要运行检索流程，请确认本地已准备对应的向量索引文件和模型 API Key
