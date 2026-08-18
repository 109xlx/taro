$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "已创建 .env 文件，请先填入 DEEPSEEK_API_KEY 后再运行。" -ForegroundColor Yellow
    notepad .env
    exit 1
}

$envContent = Get-Content ".env" -Raw
if ($envContent -notmatch "DEEPSEEK_API_KEY=sk-") {
    Write-Host "请在 .env 中配置有效的 DEEPSEEK_API_KEY（以 sk- 开头）" -ForegroundColor Yellow
    notepad .env
    exit 1
}

if (-not (Test-Path ".venv")) {
    Write-Host "正在创建虚拟环境..." -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "正在安装依赖..." -ForegroundColor Cyan
& .\.venv\Scripts\pip install -r requirements.txt -q

Write-Host "启动塔罗 AI 服务 → http://localhost:8000" -ForegroundColor Green
& .\.venv\Scripts\python main.py
