#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
神经元数据分析Web工具启动脚本
"""

import os
import sys
import webbrowser
import threading
import time

# 确保必要的目录存在
base_dir = os.path.dirname(os.path.abspath(__file__))
uploads_dir = os.path.join(base_dir, 'uploads')
results_dir = os.path.join(base_dir, 'results')
static_dir = os.path.join(base_dir, 'static')
templates_dir = os.path.join(base_dir, 'templates')

for directory in [uploads_dir, results_dir, static_dir, templates_dir]:
    os.makedirs(directory, exist_ok=True)

# 创建静态文件目录结构
os.makedirs(os.path.join(static_dir, 'js'), exist_ok=True)
os.makedirs(os.path.join(static_dir, 'css'), exist_ok=True)
os.makedirs(os.path.join(static_dir, 'images'), exist_ok=True)

def open_browser():
    """延迟打开浏览器"""
    time.sleep(1.5)  # 等待服务器启动
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    from app import app
    
    print("="*60)
    print("神经元数据分析Web工具")
    print("="*60)
    print(f"上传目录: {uploads_dir}")
    print(f"结果目录: {results_dir}")
    print("")
    print("启动Web服务器...")
    print("访问地址: http://localhost:5000")
    print("正在自动打开浏览器...")
    print("按 Ctrl+C 停止服务器")
    print("="*60)
    
    # 在后台线程中打开浏览器
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n服务器已停止")