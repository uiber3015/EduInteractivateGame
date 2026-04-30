import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / 'output'
VISUALIZER_SCRIPT = PROJECT_ROOT / 'src' / 'visualization' / 'story_visualizer.py'
DEFAULT_GRAPH_FILENAME = 'story_graph_with_image_prompts_with_scene_transitions_with_images.json'


def get_available_projects():
    if not OUTPUT_ROOT.exists():
        return []

    projects = []
    for item in sorted(OUTPUT_ROOT.iterdir()):
        if not item.is_dir():
            continue
        graph_path = item / 'output' / DEFAULT_GRAPH_FILENAME
        if graph_path.exists():
            projects.append({
                'name': item.name,
                'graph_path': graph_path,
            })
    return projects


def find_free_port(start_port=5000, max_tries=50):
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(('127.0.0.1', port)) != 0:
                return port
    raise RuntimeError('未找到可用端口，请关闭占用 5000-5049 的程序后重试。')


def choose_project(projects, requested_name=None):
    if not projects:
        raise RuntimeError('output 目录下没有找到可启动的作品。')

    if requested_name:
        for project in projects:
            if project['name'] == requested_name:
                return project
        available = ', '.join(project['name'] for project in projects)
        raise RuntimeError(f'未找到指定作品：{requested_name}。可用作品：{available}')

    return projects[-1]


def start_server(graph_path, port):
    command = [
        sys.executable,
        str(VISUALIZER_SCRIPT),
        '--story-graph',
        str(graph_path),
        '--host',
        '127.0.0.1',
        '--port',
        str(port),
    ]

    creationflags = 0
    if os.name == 'nt':
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    return subprocess.Popen(command, cwd=str(PROJECT_ROOT), creationflags=creationflags)


def wait_for_server(port, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                return True
        time.sleep(0.25)
    return False


def main():
    parser = argparse.ArgumentParser(description='启动本地作品展示')
    parser.add_argument('--project', dest='project_name', help='指定要启动的作品目录名')
    parser.add_argument('--port', type=int, default=5000, help='本地服务端口')
    parser.add_argument('--no-browser', action='store_true', help='启动后不自动打开浏览器')
    args = parser.parse_args()

    projects = get_available_projects()
    project = choose_project(projects, args.project_name)
    port = find_free_port(args.port)

    print(f'启动作品：{project["name"]}')
    print(f'故事图文件：{project["graph_path"]}')
    print(f'本地地址：http://127.0.0.1:{port}')

    process = start_server(project['graph_path'], port)
    if process.poll() is not None:
        raise RuntimeError('展示服务启动失败，请检查 Python 环境和依赖是否安装完整。')

    if not wait_for_server(port):
        process.terminate()
        raise RuntimeError('展示服务启动超时，请稍后重试。')

    if not args.no_browser:
        webbrowser.open(f'http://127.0.0.1:{port}/')

    print('浏览器已打开。关闭此窗口将不会自动停止服务，如需结束请关闭对应的 Python 进程。')
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()


if __name__ == '__main__':
    main()
