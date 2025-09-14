@echo off
chcp 65001
echo ========================================
echo    AI Mouse Analysis Project Launcher
echo ========================================
echo.

echo [1/4] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python环境，请先安装Python 3.8+
    pause
    exit /b 1
)

echo [2/4] 检查Node.js环境...
node --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Node.js环境，请先安装Node.js
    pause
    exit /b 1
)

echo [3/4] 启动后端服务...
cd /d "%~dp0backend"
start "Backend Server" cmd /k "echo 正在启动后端服务... && pip install -r requirements.txt && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo [4/4] 启动前端服务...
cd /d "%~dp0frontend"
start "Frontend Server" cmd /k "echo 正在启动前端服务... && npm install && npm run dev"

echo.
echo 等待服务启动...
timeout /t 8 /nobreak >nul

echo 正在打开浏览器...
start http://localhost:5173

echo.
echo ========================================
echo 项目启动完成！
echo 后端API: http://localhost:8000
echo 前端页面: http://localhost:5173
echo 按任意键关闭此窗口...
echo ========================================
pause >nul