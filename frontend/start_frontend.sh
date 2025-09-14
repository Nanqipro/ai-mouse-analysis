#!/bin/bash
# 前端服务启动脚本 - 解决HTTP 431错误

echo "🎨 启动前端服务（解决HTTP 431错误）..."

# 设置Node.js环境变量
export NODE_OPTIONS="--max-http-header-size=65536 --max-old-space-size=4096"

# 显示当前配置
echo "📋 当前Node.js配置:"
echo "NODE_OPTIONS: $NODE_OPTIONS"
echo "Node.js版本: $(node --version)"
echo "npm版本: $(npm --version)"

# 清理可能的缓存问题
echo "🧹 清理缓存..."
rm -rf node_modules/.vite
rm -rf dist

# 检查依赖
if [ ! -d "node_modules" ]; then
    echo "📦 安装依赖..."
    npm install
fi

# 启动开发服务器
echo "🚀 启动Vite开发服务器..."
echo "配置说明:"
echo "- 请求头大小限制: 65536 bytes (64KB)"
echo "- 内存限制: 4096MB"
echo "- 监听地址: 0.0.0.0:5173"
echo "- HMR端口: 5174"

# 启动服务
npm run dev
