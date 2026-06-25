@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo    AIDC Tech - เว็บ v2  (Django + Gradio)
echo ============================================================
echo.
echo เปิด 2 บริการ: เครื่องมือ Gradio (:7860) + เว็บ Django (:8000)
echo (จะมีหน้าต่างขึ้นมา 2 อัน - อย่าเพิ่งปิด)
echo.

start "AIDC Gradio (:7860)" "%~dp0_gradio.bat"
start "AIDC Web v2 - Django (:8000)" "%~dp0_django.bat"

echo รอเว็บเริ่มทำงานสักครู่ แล้วจะเปิดเบราว์เซอร์ให้อัตโนมัติ ...
timeout /t 10 >nul
start "" http://127.0.0.1:8000

echo.
echo เปิดเรียบร้อย! เข้าใช้งานที่  http://127.0.0.1:8000
echo   ผู้ใช้ทดสอบ:  admin  /  admin12345
echo.
echo หากเบราว์เซอร์ยังไม่ขึ้น ให้เปิด http://127.0.0.1:8000 เอง
echo (ปิดบริการ = ปิดหน้าต่าง Gradio และ Django)
echo.
pause
