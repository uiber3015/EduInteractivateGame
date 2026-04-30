"""
一键启动脚本

自动完成以下流程：
1. 生成交互式物理故事（COT Pipeline）
2. 自动启动可视化服务器查看结果

使用方法：
    python one_click_start.py
"""

import os
import sys
import glob
import webbrowser
import threading
import time

# 添加src目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def find_latest_story_graph(output_dir):
    """在输出目录中查找最终的故事图JSON文件"""
    # 优先查找带图片的完整版本
    patterns = [
        os.path.join(output_dir, "output", "*_with_images.json"),
        os.path.join(output_dir, "output", "*_with_scene_transitions.json"),
        os.path.join(output_dir, "output", "*_with_image_prompts.json"),
        os.path.join(output_dir, "output", "enhanced_story_graph.json"),
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            # 返回最新的文件
            return max(matches, key=os.path.getmtime)
    
    return None


def open_browser_delayed(url, delay=2):
    """延迟打开浏览器"""
    time.sleep(delay)
    webbrowser.open(url)


def main():
    print("\n" + "=" * 70)
    print("  🚀 一键启动：交互式物理教育故事生成 + 可视化")
    print("=" * 70)
    
    # ========================================
    # 阶段1：运行COT Pipeline生成故事
    # ========================================
    print("\n📌 阶段1：生成交互式物理故事")
    print("-" * 50)
    
    from cot_full_pipeline import cot_pipeline
    
    # 默认配置
    default_knowledge_points = [
        "杠杆原理", "摩擦力", "重心与稳定性"
    ]
    default_scenario = "小明帮助爷爷修理老房子：需要撬起重物、搬运家具、调整倾斜的书架，在这个过程中学习物理知识"
    default_num_questions = 3
    
    print(f"\n候选知识点池: {', '.join(default_knowledge_points)}")
    print(f"故事大方向: {default_scenario}")
    print(f"题目数量: {default_num_questions}")
    
    # 询问是否使用默认配置
    use_default = input("\n是否使用默认配置? (y/n, 默认y): ").strip().lower()
    
    if use_default == 'n':
        # 询问是否自动选择知识点
        auto_select_input = input("\n是否让大模型从400个知识点池中自动选择知识点? (y/n, 默认n): ").strip().lower()
        auto_select_knowledge = auto_select_input == 'y'
        
        if auto_select_knowledge:
            knowledge_points = None
            scenario = input("请输入故事场景: ").strip()
            if not scenario:
                scenario = default_scenario
            
            num_input = input(f"请输入题目数量（默认{default_num_questions}）: ").strip()
            num_questions = int(num_input) if num_input.isdigit() else default_num_questions
        else:
            kp_input = input("请输入知识点（用逗号分隔）: ").strip()
            knowledge_points = [kp.strip() for kp in kp_input.split(",") if kp.strip()]
            if not knowledge_points:
                knowledge_points = default_knowledge_points
            
            scenario = input("请输入故事场景: ").strip()
            if not scenario:
                scenario = default_scenario
            
            num_input = input(f"请输入题目数量（默认{default_num_questions}）: ").strip()
            num_questions = int(num_input) if num_input.isdigit() else default_num_questions
    else:
        auto_select_knowledge = False
        knowledge_points = default_knowledge_points
        scenario = default_scenario
        num_questions = default_num_questions
    
    # 询问是否生成图片
    gen_images_input = input("\n是否生成图片? (y/n, 默认y): ").strip().lower()
    generate_images = gen_images_input != 'n'

    # 选择文本提供商
    print("\n可用的文本提供商:")
    print("  1. aigcbest (默认，当前旧通道)")
    print("  2. yunwu")
    text_provider_choice = input("请选择文本提供商 (1/2, 默认1): ").strip()
    text_provider = "yunwu" if text_provider_choice == "2" else "aigcbest"
    print(f"✅ 已选择文本提供商: {text_provider}")

    image_provider = "legacy"
    
    # 选择模型
    model = "gpt-image-2"
    if generate_images:
        print("\n可用的图像提供商:")
        print("  1. legacy (默认，当前旧通道)")
        print("  2. yunwu")
        image_provider_choice = input("请选择图像提供商 (1/2, 默认1): ").strip()
        image_provider = "yunwu" if image_provider_choice == "2" else "legacy"
        print(f"✅ 已选择图像提供商: {image_provider}")

        print("\n可用的图像生成模型:")
        print("  1. gpt-image-1")
        print("  2. gpt-image-2 (默认，推荐)")
        print("  3. gpt-image-1.5")
        model_choice = input("请选择模型 (1/2/3, 默认2): ").strip()
        if model_choice == "1":
            model = "gpt-image-1"
        elif model_choice == "3":
            model = "gpt-image-1.5"
        else:
            model = "gpt-image-2"
        print(f"✅ 已选择模型: {model}")
    
    # 运行Pipeline
    result = cot_pipeline(
        knowledge_points=knowledge_points,
        scenario=scenario,
        num_questions=num_questions,
        generate_images=generate_images,
        model=model,
        text_provider=text_provider,
        image_provider=image_provider,
        auto_select_knowledge=auto_select_knowledge
    )
    
    output_dir = result["output_dir"]
    print(f"\n✅ 故事生成完成！输出目录: {output_dir}")
    
    # ========================================
    # 阶段2：启动可视化服务器
    # ========================================
    print("\n" + "=" * 70)
    print("📌 阶段2：启动可视化服务器")
    print("-" * 50)
    
    # 查找最终的故事图文件
    story_graph_path = find_latest_story_graph(output_dir)
    
    if not story_graph_path:
        print("⚠️ 未找到故事图文件，请手动运行 story_visualizer.py")
        return
    
    print(f"📁 故事图文件: {story_graph_path}")
    
    # 启动可视化
    from visualization.story_visualizer import run_story_visualizer
    
    port = 5000
    url = f"http://localhost:{port}"
    
    print(f"\n🌐 启动可视化服务器: {url}")
    print(f"   交互式故事: {url}/interactive-story")
    print(f"   故事图谱: {url}/story-graph-visualization")
    print("\n按 Ctrl+C 停止服务器\n")
    
    # 延迟自动打开浏览器
    browser_thread = threading.Thread(
        target=open_browser_delayed, 
        args=(f"{url}/interactive-story", 2),
        daemon=True
    )
    browser_thread.start()
    
    # 启动Flask服务器（阻塞）
    run_story_visualizer(
        custom_story_graph_path=story_graph_path,
        debug=False,
        host='0.0.0.0',
        port=port
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 服务器已停止")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
