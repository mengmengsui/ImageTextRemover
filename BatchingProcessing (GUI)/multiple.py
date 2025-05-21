from flask import Flask, render_template, jsonify, request
import threading
import uuid
import subprocess
import sys
from multiple_core import process_folder

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

tasks = {}

def select_folder():
    """通过独立进程调用Tkinter选择文件夹（修复编码和空值）"""
    try:
        # 启动独立进程并指定UTF-8编码
        result = subprocess.run(
            [sys.executable, "folder_selector.py"],
            capture_output=True,
            text=True,          # 启用文本模式
            encoding='utf-8',   # 显式指定UTF-8编码
            timeout=30
        )
        # 处理可能的空输出（用户取消选择）
        return result.stdout.strip() if result.stdout else ""
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        print(f"选择文件夹异常: {str(e)}")
        return ""

def background_process(task_id, folder_path):
    """后台处理任务（保持不变）"""
    try:
        tasks[task_id] = {'progress': [], 'is_complete': False}
        process_folder(
            folder_path=folder_path,
            update_progress=lambda filename, status, msg:
                tasks[task_id]['progress'].append({
                    'filename': filename,
                    'status': status,
                    'message': msg
                })
        )
    except Exception as e:
        tasks[task_id]['progress'].append({
            'filename': '系统错误',
            'status': 'error',
            'message': str(e)
        })
    finally:
        tasks[task_id]['is_complete'] = True


@app.route('/')
def index():
    return render_template('multiple_index.html')


@app.route('/select-folder')
def get_folder():
    folder = select_folder()
    return jsonify({'folderPath': folder if folder else ''})


@app.route('/start-process', methods=['POST'])
def start_process():
    data = request.get_json()
    task_id = str(uuid.uuid4())

    # 启动后台线程处理
    threading.Thread(
        target=background_process,
        args=(task_id, data['folder']),
        daemon=True
    ).start()

    return jsonify({'taskId': task_id})


@app.route('/process-status/<task_id>')
def get_process_status(task_id):
    task = tasks.get(task_id, {
        'progress': [],
        'is_complete': False
    })
    return jsonify({
        'progress': task['progress'],
        'is_complete': task['is_complete']
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
