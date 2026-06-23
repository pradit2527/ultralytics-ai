@echo off
chcp 65001 >nul
title AIDC Gradio (:7860)
cd /d "%~dp0.."

set "PYEXE=%~dp0..\venv\Scripts\python.exe"
if not exist "%PYEXE%" (
  echo [ผิดพลาด] ไม่พบ Python ของ Gradio ที่ %PYEXE%
  echo ตรวจสอบว่าโฟลเดอร์ venv ของโปรเจกต์หลักอยู่ครบหรือไม่
  pause
  exit /b 1
)

echo กำลังเริ่มเครื่องมือ Gradio ...
echo (ครั้งแรกอาจใช้เวลาโหลดโมเดลสักครู่ - รอจนขึ้น Running on local URL)
echo.
"%PYEXE%" app.py

echo.
echo [Gradio หยุดทำงานแล้ว] กดปุ่มใดก็ได้เพื่อปิดหน้าต่าง
pause >nul
