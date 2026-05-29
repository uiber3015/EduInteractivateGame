# EduInteractiveGame 项目结构

## 在线 Demo

| 类型 | 链接 |
|------|------|
| 源码仓库 | [https://github.com/uiber3015/EducationalSeriousGame_Demo](https://github.com/uiber3015/EducationalSeriousGame_Demo) |
| 可游玩 Demo（Vercel） | [https://educational-serious-game-demo.vercel.app/](https://educational-serious-game-demo.vercel.app/) |

- **GitHub**：双语教育游戏静态演示包的完整代码与资源（含 `index.html`、`interactive-story.html`、叙事地图页及 `static/story_graph/` 数据）。
- **在线 Demo**：部署在 Vercel 上的可访问版本，无需本地启动 Flask，浏览器打开即可进入总入口并体验四个互动故事。

---

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

> **说明**：当前对外展示的静态 Demo 由独立仓库 [EducationalSeriousGame_Demo](https://github.com/uiber3015/EducationalSeriousGame_Demo) 维护，由本项目的生成结果打包部署；本地开发仍以 `src/` 下的 Pipeline 与可视化模块为主。

---

## 使用方法

### 0. 在线体验（无需安装）

直接打开可游玩 Demo：[https://educational-serious-game-demo.vercel.app/](https://educational-serious-game-demo.vercel.app/)

- 总入口：`/` 或 `/zh`、`/en`
- 互动故事：`interactive-story.html?lang=zh&game=tech_education`（可替换游戏与语言参数）
- 叙事地图：`narrative-map-classic.html` / `narrative-map-elk.html`

源码与静态资源见：[https://github.com/uiber3015/EducationalSeriousGame_Demo](https://github.com/uiber3015/EducationalSeriousGame_Demo)

### 1. 快速开始（推荐）

```bash
python src/one_click_start.py
```

### 2. 使用核心 Pipeline

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

---

## 打包内容说明

1. **保留核心代码**：包含执行代码、前端模板、图像生成与可视化逻辑
2. **保留核心数据**：包含 `data/rag_data`、`knowledge.txt`、`prompts.txt`
3. **预留输出目录**：根目录下已创建空的 `output/` 用于放置后续生成结果
4. **去除非核心内容**：未包含旧归档、历史输出、缓存和其他冗余目录
5. **对外 Demo**：生成内容可同步至 [EducationalSeriousGame_Demo](https://github.com/uiber3015/EducationalSeriousGame_Demo)，并通过 Vercel 发布为 [在线可玩版本](https://educational-serious-game-demo.vercel.app/)

---

## 注意事项

- 所有代码现在位于 `src/` 目录下
- 本打包版不包含 `archive/`、历史 `output/`、`docs/` 等非核心目录
- 运行脚本时需要从项目根目录执行
- 环境变量配置位于根目录 `.env`
- 如果需要运行检索流程，请确认本地已准备对应的向量索引文件和模型 API Key
- 在线 Demo 为静态站点，不依赖本仓库内的 Flask 服务；本地调试可视化仍可使用 `src/visualization/story_visualizer.py`
