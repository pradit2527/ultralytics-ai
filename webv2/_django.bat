@echo off
chcp 65001 >nul
title AIDC Web v2 - Django (:8000)
cd /d "%~dp0"

set "BASEPY=%~dp0..\venv\Scripts\python.exe"
set "PYEXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYEXE%" (
  echo [ตั้งค่าครั้งแรก] กำลังสร้าง virtualenv ของ Django ...
  "%BASEPY%" -m venv .venv
)

echo [1/3] ติดตั้ง/ตรวจสอบ Django ...
"%PYEXE%" -m pip install -q -r requirements.txt

echo [2/3] เตรียมฐานข้อมูล + บัญชีผู้ดูแล ...
"%PYEXE%" manage.py migrate
"%PYEXE%" manage.py shell -c "from django.contrib.auth import get_user_model as g;U=g();U.objects.filter(username='admin').exists() or U.objects.create_superuser('admin','admin@aidc.local','admin12345')"

echo [3/3] เปิดเว็บที่ http://127.0.0.1:8000  (admin / admin12345)
echo.
"%PYEXE%" manage.py runserver 127.0.0.1:8000

echo.
echo [Django หยุดทำงานแล้ว] กดปุ่มใดก็ได้เพื่อปิดหน้าต่าง
pause >nul
