@echo off
chcp 65001
echo ========================================
echo    AI Mouse Analysis Project Launcher
echo ========================================
echo.

echo [1/2] 启动后端服务...
start "Backend Server" cmd /k "conda activate base && cd /d \"%~dp0backend\" && python main.py"

echo [2/2] 启动前端服务...
start "Frontend Server" cmd /k "cd /d \"%~dp0frontend\" && npm i && npm run dev && start http://localhost:5175"

echo.
echo ========================================
echo 项目启动完成！
echo 后端和前端服务正在启动中...
echo 前端页面将自动打开: http://localhost:5175
echo 按任意键关闭此窗口...
echo ========================================
pause >nul