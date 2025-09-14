#!/bin/bash
# 钙信号分析平台开发环境启动脚本

echo "🚀 启动钙信号分析平台开发环境..."

# 设置环境变量以解决HTTP 431错误
export NODE_OPTIONS="--max-http-header-size=65536 --max-old-space-size=4096"
export UV_THREADPOOL_SIZE=128

# 检查Node.js版本
echo "📋 检查Node.js版本..."
node --version
npm --version

# 检查Python版本
echo "📋 检查Python版本..."
python --version
python -c "import sys; print(f'Python可执行文件: {sys.executable}')"

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p backend/uploads
mkdir -p backend/results
mkdir -p backend/temp

# 启动后端服务
echo "🔧 启动后端服务..."
cd backend
echo "当前目录: $(pwd)"

# 检查后端依赖
if [ ! -f "requirements.txt" ]; then
    echo "❌ 未找到requirements.txt"
    exit 1
fi

# 启动FastAPI服务器（后台运行）
echo "启动FastAPI服务器..."
nohup python main.py > ../backend.log 2>&1 &
BACKEND_PID=$!
echo "后端服务PID: $BACKEND_PID"

# 等待后端启动
echo "⏳ 等待后端服务启动..."
sleep 5

# 检查后端是否正常启动
echo "🔍 检查后端服务状态..."
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✅ 后端服务启动成功"
else
    echo "❌ 后端服务启动失败，请检查backend.log"
    cat ../backend.log
    exit 1
fi

# 启动前端服务
echo "🎨 启动前端服务..."
cd ../frontend

# 检查前端依赖
if [ ! -d "node_modules" ]; then
    echo "📦 安装前端依赖..."
    npm install
fi

# 启动Vite开发服务器
echo "启动Vite开发服务器..."
npm run dev

echo "🎉 开发环境启动完成！"
echo "前端地址: http://localhost:5173"
echo "后端地址: http://localhost:8000"
echo "后端API文档: http://localhost:8000/docs"
