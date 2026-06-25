# รันเว็บ v2 (Django) บน Windows — ทางเลือกแบบ PowerShell
# วิธีที่ง่ายที่สุดคือ "ดับเบิลคลิก start.bat" (เปิดทั้ง Gradio + Django ให้เลย)
# สคริปต์นี้เปิดเฉพาะฝั่ง Django (ต้องเปิด Gradio แยกเอง)
#
# ใช้:  คลิกขวาที่ไฟล์ > Run with PowerShell   หรือ   .\run_webv2.ps1

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# ใช้ Python จาก venv ของโปรเจกต์หลักเพื่อสร้าง venv (ไม่พึ่ง python ใน PATH
# ซึ่งบน Windows มักชี้ไปที่ตัว stub ของ Microsoft Store ที่รันไม่ได้)
$basePy = Join-Path $PSScriptRoot "..\venv\Scripts\python.exe"
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPy)) {
    Write-Host "[ตั้งค่าครั้งแรก] สร้าง virtualenv ..." -ForegroundColor Cyan
    & $basePy -m venv .venv
}

Write-Host "[1/3] ติดตั้ง Django ..." -ForegroundColor Cyan
& $venvPy -m pip install -q -r requirements.txt

Write-Host "[2/3] เตรียมฐานข้อมูล + บัญชีผู้ดูแล ..." -ForegroundColor Cyan
& $venvPy manage.py migrate
& $venvPy manage.py shell -c "from django.contrib.auth import get_user_model as g;U=g();U.objects.filter(username='admin').exists() or U.objects.create_superuser('admin','admin@aidc.local','admin12345')"

Write-Host "[3/3] เปิดเว็บที่ http://127.0.0.1:8000  (admin / admin12345)" -ForegroundColor Green
& $venvPy manage.py runserver 127.0.0.1:8000
