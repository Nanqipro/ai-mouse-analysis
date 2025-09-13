#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
神经元数据分析Web工具

主应用程序，提供Web界面用于选择算法模块、上传数据和调节参数
"""

import os
import sys
import json
import uuid
import subprocess
import shutil
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd

# 添加配置路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'adapters'))
from modules_config import MODULES_CONFIG, get_module_config, get_all_modules, validate_parameters, UPLOADS_DIR, RESULTS_DIR
from module_adapter import AdapterFactory

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# 确保必要的目录存在
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(app.static_folder or 'static'), exist_ok=True)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'xlsx', 'csv', 'txt'}

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_task_id():
    """生成唯一的任务ID"""
    return str(uuid.uuid4())

def get_task_dir(task_id):
    """获取任务目录路径"""
    return os.path.join(RESULTS_DIR, task_id)

def save_task_info(task_id, info):
    """保存任务信息"""
    task_dir = get_task_dir(task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    info_file = os.path.join(task_dir, 'task_info.json')
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

def load_task_info(task_id):
    """加载任务信息"""
    info_file = os.path.join(get_task_dir(task_id), 'task_info.json')
    if os.path.exists(info_file):
        with open(info_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def validate_uploaded_file(file_path, module_name):
    """验证上传的文件格式"""
    try:
        adapter = AdapterFactory.get_adapter(module_name)
        return adapter.validate_input(file_path)
    except Exception as e:
        return False, f"验证失败: {str(e)}"

@app.route('/')
def index():
    """主页"""
    modules = {name: config for name, config in MODULES_CONFIG.items()}
    return render_template('index.html', modules=modules)

@app.route('/api/modules')
def api_modules():
    """获取所有模块信息"""
    return jsonify(MODULES_CONFIG)

@app.route('/api/module/<module_name>')
def api_module_detail(module_name):
    """获取特定模块的详细信息"""
    config = get_module_config(module_name)
    if config:
        return jsonify(config)
    else:
        return jsonify({'error': '模块不存在'}), 404

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """文件上传接口"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        file = request.files['file']
        module_name = request.form.get('module')
        
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not module_name:
            return jsonify({'error': '没有指定模块'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件格式'}), 400
        
        # 生成任务ID
        task_id = generate_task_id()
        task_dir = get_task_dir(task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        # 保存文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(task_dir, filename)
        file.save(file_path)
        
        # 验证文件
        is_valid, message = validate_uploaded_file(file_path, module_name)
        if not is_valid:
            shutil.rmtree(task_dir)  # 删除任务目录
            return jsonify({'error': f'文件验证失败: {message}'}), 400
        
        # 保存任务信息
        task_info = {
            'task_id': task_id,
            'module': module_name,
            'filename': filename,
            'file_path': file_path,
            'upload_time': datetime.now().isoformat(),
            'status': 'uploaded',
            'validation_message': message
        }
        save_task_info(task_id, task_info)
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': message
        })
    
    except Exception as e:
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

@app.route('/api/upload_additional', methods=['POST'])
def api_upload_additional():
    """上传额外文件（如位置文件）"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        file = request.files['file']
        task_id = request.form.get('task_id')
        file_type = request.form.get('file_type')  # 如 'position_file'
        
        if not task_id or not file_type:
            return jsonify({'error': '缺少必要参数'}), 400
        
        task_info = load_task_info(task_id)
        if not task_info:
            return jsonify({'error': '任务不存在'}), 404
        
        # 保存额外文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(get_task_dir(task_id), f"{file_type}_{filename}")
        file.save(file_path)
        
        # 更新任务信息
        if 'additional_files' not in task_info:
            task_info['additional_files'] = {}
        task_info['additional_files'][file_type] = {
            'filename': filename,
            'file_path': file_path
        }
        save_task_info(task_id, task_info)
        
        return jsonify({
            'success': True,
            'message': f'{file_type} 文件上传成功'
        })
    
    except Exception as e:
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """开始分析"""
    try:
        data = request.get_json()
        task_id = data.get('task_id')
        parameters = data.get('parameters', {})
        
        if not task_id:
            return jsonify({'error': '缺少任务ID'}), 400
        
        task_info = load_task_info(task_id)
        if not task_info:
            return jsonify({'error': '任务不存在'}), 404
        
        module_name = task_info['module']
        
        # 验证参数
        is_valid, errors = validate_parameters(module_name, parameters)
        if not is_valid:
            return jsonify({'error': f'参数验证失败: {"; ".join(errors)}'}), 400
        
        # 更新任务状态
        task_info['parameters'] = parameters
        task_info['status'] = 'running'
        task_info['start_time'] = datetime.now().isoformat()
        save_task_info(task_id, task_info)
        
        # 准备额外文件
        additional_files = {}
        if 'additional_files' in task_info:
            for file_type, file_info in task_info['additional_files'].items():
                additional_files[file_type] = file_info['file_path']
        
        # 异步执行分析
        def run_analysis_async():
            try:
                adapter = AdapterFactory.get_adapter(module_name)
                output_dir = get_task_dir(task_id)
                
                result = adapter.run_analysis(
                    file_path=task_info['file_path'],
                    output_dir=output_dir,
                    parameters=parameters,
                    additional_files=additional_files
                )
                
                # 更新任务状态
                task_info = load_task_info(task_id)
                if result['success']:
                    task_info['status'] = 'completed'
                    task_info['output_files'] = [f['filename'] for f in result['output_files']]
                else:
                    task_info['status'] = 'error'
                    task_info['error'] = result['error']
                
                task_info['end_time'] = datetime.now().isoformat()
                task_info['analysis_result'] = result
                save_task_info(task_id, task_info)
                
            except Exception as e:
                task_info = load_task_info(task_id)
                task_info['status'] = 'error'
                task_info['error'] = str(e)
                task_info['end_time'] = datetime.now().isoformat()
                save_task_info(task_id, task_info)
        
        # 在后台线程中运行分析
        analysis_thread = threading.Thread(target=run_analysis_async)
        analysis_thread.daemon = True
        analysis_thread.start()
            
        return jsonify({
            'success': True,
            'message': '分析已开始',
            'task_id': task_id
        })
        
    except Exception as e:
        task_info['status'] = 'error'
        task_info['error'] = str(e)
        save_task_info(task_id, task_info)
        return jsonify({'error': f'启动分析失败: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'分析请求处理失败: {str(e)}'}), 500

@app.route('/api/status/<task_id>')
def api_status(task_id):
    """获取任务状态"""
    task_info = load_task_info(task_id)
    if not task_info:
        return jsonify({'error': '任务不存在'}), 404
    
    # 任务状态已经由分析线程更新，这里不需要额外检查
    
    return jsonify(task_info)

@app.route('/api/results/<task_id>')
def api_results(task_id):
    """获取分析结果"""
    task_info = load_task_info(task_id)
    if not task_info:
        return jsonify({'error': '任务不存在'}), 404
    
    if task_info.get('status') != 'completed':
        return jsonify({'error': '分析尚未完成'}), 400
    
    task_dir = get_task_dir(task_id)
    results = []
    
    # 扫描输出文件
    for filename in os.listdir(task_dir):
        if filename.startswith('task_info.') or filename.startswith('temp_'):
            continue
        
        file_path = os.path.join(task_dir, filename)
        if os.path.isfile(file_path):
            file_ext = filename.split('.')[-1].lower()
            file_type = 'unknown'
            
            if file_ext in ['png', 'jpg', 'jpeg', 'gif']:
                file_type = 'image'
            elif file_ext in ['csv', 'xlsx', 'txt']:
                file_type = 'data'
            
            results.append({
                'filename': filename,
                'type': file_type,
                'size': os.path.getsize(file_path),
                'download_url': url_for('download_file', task_id=task_id, filename=filename)
            })
    
    return jsonify({
        'task_id': task_id,
        'results': results
    })

@app.route('/download/<task_id>/<filename>')
def download_file(task_id, filename):
    """下载结果文件"""
    task_info = load_task_info(task_id)
    if not task_info:
        return "任务不存在", 404
    
    file_path = os.path.join(get_task_dir(task_id), filename)
    if not os.path.exists(file_path):
        return "文件不存在", 404
    
    return send_file(file_path, as_attachment=True)

@app.route('/view/<task_id>/<filename>')
def view_file(task_id, filename):
    """查看结果文件（主要用于图片）"""
    task_info = load_task_info(task_id)
    if not task_info:
        return "任务不存在", 404
    
    file_path = os.path.join(get_task_dir(task_id), filename)
    if not os.path.exists(file_path):
        return "文件不存在", 404
    
    return send_file(file_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)