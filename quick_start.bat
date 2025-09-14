@echo off
chcp 65001
echo ========================================
echo        快速启动 AI Mouse Analysis
echo ========================================
echo.

echo 启动后端服务...
cd /d "%~dp0backend"
start "AI Mouse Analysis - Backend" cmd /k "uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo 启动前端服务...
cd /d "%~dp0frontend"
start "AI Mouse Analysis - Frontend" cmd /k "npm run dev"

echo.
echo 等待服务启动...
timeout /t 5 /nobreak >nul

echo 打开浏览器...
start http://localhost:5173

echo.
echo ========================================
echo 启动完成！
echo 前端: http://localhost:5173
echo 后端: http://localhost:8000
echo API文档: http://localhost:8000/docs
echo ========================================
echo.
echo 按任意键关闭此窗口...
pause >nul