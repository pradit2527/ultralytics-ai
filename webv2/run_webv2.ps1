# รันเว็บ v2 (Django) บน Windows
# ใช้:  .\run_webv2.ps1
# ต้องรันบริการ Gradio (app.py ในโฟลเดอร์แม่) คู่กันด้วย เพื่อให้ iframe แสดงผล

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "สร้าง virtualenv ..." -ForegroundColor Cyan
    python -m venv .venv
}
& ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
& ".venv\Scripts\python.exe" manage.py migrate
Write-Host "เปิดเว็บ v2 ที่ http://127.0.0.1:8000" -ForegroundColor Green
& ".venv\Scripts\python.exe" manage.py runserver 127.0.0.1:8000
