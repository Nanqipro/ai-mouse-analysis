# HTTP 431 错误故障排除指南

## 问题描述
在钙波检测功能模块中出现 "Request Header Fields Too Large" (HTTP 431) 错误，导致预览图生成失败。

## 解决方案

### 🚨 立即行动步骤

#### 第一步：完全停止所有服务
```bash
# 停止前端服务（在前端终端按 Ctrl+C）
# 停止后端服务（在后端终端按 Ctrl+C）

# 确保端口被完全释放
lsof -ti:5173 | xargs kill -9
lsof -ti:8000 | xargs kill -9
```

#### 第二步：设置环境变量
在终端中执行：
```bash
export NODE_OPTIONS="--max-http-header-size=65536 --max-old-space-size=4096"
export UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE=65536
```

#### 第三步：使用新的启动脚本重启服务

**方式1：使用自动化脚本（推荐）**
```bash
cd /Users/nanpipro/Documents/gitlocal/ai-mouse-analysis
./start_dev.sh
```

**方式2：分别手动启动**

后端：
```bash
cd /Users/nanpipro/Documents/gitlocal/ai-mouse-analysis/backend
export UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE=65536
python main.py
```

前端（新终端）：
```bash
cd /Users/nanpipro/Documents/gitlocal/ai-mouse-analysis/frontend
./start_frontend.sh
```

### 🔧 已应用的修复

#### 前端修复 (vite.config.js)
- ✅ 增加 `maxHttpHeaderSize` 到 64KB
- ✅ 配置代理缓冲和超时
- ✅ 移除问题头部信息
- ✅ 增加HMR端口配置

#### 后端修复 (main.py)
- ✅ 设置 `h11_max_incomplete_event_size` 为 64KB
- ✅ 增加文件上传限制到 200MB
- ✅ 优化并发和超时配置
- ✅ 改进中间件错误处理

#### 启动配置 (package.json)
- ✅ 更新Node.js启动参数
- ✅ 增加内存限制配置

### 🔍 验证步骤

1. **检查服务状态**
   ```bash
   curl http://localhost:8000/  # 后端健康检查
   curl http://localhost:5173/  # 前端访问检查
   ```

2. **查看服务日志**
   - 后端：检查控制台输出，应该显示配置信息
   - 前端：检查是否有431错误信息

3. **测试文件上传**
   - 访问 http://localhost:5173
   - 选择钙波检测功能
   - 上传小文件（< 50MB）进行测试

### 🚨 如果问题仍然存在

#### 方案A：减少文件大小
- 使用小于50MB的Excel文件
- 分批处理大型数据集
- 删除不必要的数据列

#### 方案B：系统级配置
如果上述方案无效，可能需要系统级配置：

**macOS:**
```bash
# 临时增加系统限制
sudo sysctl -w kern.ipc.maxsockbuf=16777216
```

**检查Node.js版本:**
```bash
node --version  # 推荐 v18.0.0 或更高
npm --version
```

#### 方案C：替代上传方式
如果问题持续，可以：
1. 将文件分割成多个小文件
2. 使用命令行工具直接处理
3. 考虑使用不同的数据格式（CSV而非Excel）

### 📋 环境要求
- Node.js: >= 18.0.0
- Python: >= 3.8
- 可用内存: >= 4GB
- 系统支持的最大请求头: >= 64KB

### 📞 支持信息
如果问题仍然存在，请提供：
1. 使用的文件大小
2. 浏览器控制台的完整错误信息
3. 后端日志输出
4. 系统信息（Node.js版本、操作系统等）
