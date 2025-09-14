# AI Mouse Analysis Project Launcher (PowerShell版本)
# 使用方法：右键点击此文件，选择"使用PowerShell运行"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   AI Mouse Analysis Project Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查Python环境
Write-Host "[1/4] 检查Python环境..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python已安装: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ 错误: 未找到Python环境，请先安装Python 3.8+" -ForegroundColor Red
    Read-Host "按Enter键退出"
    exit 1
}

# 检查Node.js环境
Write-Host "[2/4] 检查Node.js环境..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✓ Node.js已安装: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ 错误: 未找到Node.js环境，请先安装Node.js" -ForegroundColor Red
    Read-Host "按Enter键退出"
    exit 1
}

# 获取脚本所在目录
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path

# 启动后端服务
Write-Host "[3/4] 启动后端服务..." -ForegroundColor Yellow
$backendPath = Join-Path $scriptPath "backend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendPath'; Write-Host '正在安装后端依赖...' -ForegroundColor Cyan; pip install -r requirements.txt; Write-Host '启动后端服务...' -ForegroundColor Green; uvicorn main:app --host 0.0.0.0 --port 8000 --reload" -WindowStyle Normal

# 启动前端服务
Write-Host "[4/4] 启动前端服务..." -ForegroundColor Yellow
$frontendPath = Join-Path $scriptPath "frontend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendPath'; Write-Host '正在安装前端依赖...' -ForegroundColor Cyan; npm install; Write-Host '启动前端服务...' -ForegroundColor Green; npm run dev" -WindowStyle Normal

# 等待服务启动
Write-Host "" 
Write-Host "等待服务启动..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# 打开浏览器
Write-Host "正在打开浏览器..." -ForegroundColor Yellow
Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "项目启动完成！" -ForegroundColor Green
Write-Host "后端API: http://localhost:8000" -ForegroundColor White
Write-Host "前端页面: http://localhost:5173" -ForegroundColor White
Write-Host "API文档: http://localhost:8000/docs" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "提示：关闭此窗口不会停止服务，请手动关闭后端和前端窗口" -ForegroundColor Yellow
Read-Host "按Enter键关闭此窗口"