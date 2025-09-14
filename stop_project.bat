@echo off
chcp 65001
echo ========================================
echo        停止 AI Mouse Analysis 服务
echo ========================================
echo.

echo 正在停止后端服务...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8000" ^| find "LISTENING"') do (
    echo 找到后端进程 PID: %%a
    taskkill /f /pid %%a >nul 2>&1
    if not errorlevel 1 (
        echo ✓ 后端服务已停止
    ) else (
        echo ✗ 停止后端服务失败
    )
)

echo 正在停止前端服务...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":5173" ^| find "LISTENING"') do (
    echo 找到前端进程 PID: %%a
    taskkill /f /pid %%a >nul 2>&1
    if not errorlevel 1 (
        echo ✓ 前端服务已停止
    ) else (
        echo ✗ 停止前端服务失败
    )
)

echo.
echo 正在关闭相关命令行窗口...
for /f "tokens=2" %%a in ('tasklist /fi "windowtitle eq AI Mouse Analysis - Backend" /fo csv ^| find "cmd.exe"') do (
    taskkill /f /pid %%a >nul 2>&1
)
for /f "tokens=2" %%a in ('tasklist /fi "windowtitle eq AI Mouse Analysis - Frontend" /fo csv ^| find "cmd.exe"') do (
    taskkill /f /pid %%a >nul 2>&1
)

echo.
echo ========================================
echo 所有服务已停止！
echo ========================================
echo.
echo 按任意键关闭此窗口...
pause >nul