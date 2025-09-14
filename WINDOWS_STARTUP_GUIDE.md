# Windows 一键启动指南

本项目提供了两种Windows一键启动方式，可以同时启动后端API服务和前端界面，并自动打开浏览器。

## 🚀 快速启动

### 方式一：批处理文件（推荐）
1. 双击 `start_project.bat` 文件
2. 等待服务启动完成
3. 浏览器会自动打开前端页面

### 方式二：PowerShell脚本
1. 右键点击 `start_project.ps1` 文件
2. 选择 "使用PowerShell运行"
3. 如果出现执行策略错误，请先运行：`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## 📋 前置要求

在使用启动脚本之前，请确保已安装以下软件：

### Python 环境
- **Python 3.8+**
- 下载地址：https://www.python.org/downloads/
- 安装时请勾选 "Add Python to PATH"

### Node.js 环境
- **Node.js 16+**
- 下载地址：https://nodejs.org/
- 推荐下载 LTS 版本

## 🌐 服务地址

启动成功后，可以通过以下地址访问：

- **前端界面**: http://localhost:5173
- **后端API**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **API交互文档**: http://localhost:8000/redoc

## 🔧 手动启动（备选方案）

如果自动启动脚本遇到问题，可以手动启动：

### 启动后端
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 启动前端
```bash
cd frontend
npm install
npm run dev
```

## ❗ 常见问题

### 1. Python 未找到
- 确保已安装 Python 3.8+
- 检查 Python 是否已添加到系统 PATH
- 在命令行中运行 `python --version` 验证

### 2. Node.js 未找到
- 确保已安装 Node.js 16+
- 检查 Node.js 是否已添加到系统 PATH
- 在命令行中运行 `node --version` 验证

### 3. 端口被占用
- 后端默认端口：8000
- 前端默认端口：5173
- 如果端口被占用，请关闭占用端口的程序或修改配置

### 4. PowerShell 执行策略错误
以管理员身份运行 PowerShell，执行：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 5. 依赖安装失败
- 确保网络连接正常
- 可以尝试使用国内镜像源：
  - Python: `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/`
  - Node.js: `npm install --registry https://registry.npmmirror.com`

## 🛑 停止服务

要停止服务，请：
1. 关闭后端服务窗口（按 Ctrl+C 或直接关闭窗口）
2. 关闭前端服务窗口（按 Ctrl+C 或直接关闭窗口）
3. 关闭浏览器标签页

## 📝 项目结构

```
ai-mouse-analysis/
├── backend/                 # 后端API服务
│   ├── main.py             # FastAPI主程序
│   ├── requirements.txt    # Python依赖
│   └── src/                # 业务逻辑模块
├── frontend/               # 前端Vue应用
│   ├── package.json        # Node.js依赖
│   ├── src/                # Vue源码
│   └── index.html          # 入口页面
├── start_project.bat       # Windows批处理启动脚本
├── start_project.ps1       # PowerShell启动脚本
└── WINDOWS_STARTUP_GUIDE.md # 本说明文档
```

## 🎯 功能特性

- ✅ 自动检查Python和Node.js环境
- ✅ 自动安装项目依赖
- ✅ 同时启动后端和前端服务
- ✅ 自动打开浏览器访问前端页面
- ✅ 彩色输出和进度提示
- ✅ 错误处理和友好提示

---

如有问题，请检查控制台输出的错误信息，或参考上述常见问题解决方案。