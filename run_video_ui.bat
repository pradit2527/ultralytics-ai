@echo off
REM เปิดเว็บ UI ประมวลผลวิดีโอด้วย YOLO
cd /d D:\yoloe
echo Starting YOLO Video Processor...
echo เปิดเบราว์เซอร์อัตโนมัติที่ http://127.0.0.1:7860
"D:\yoloe\venv\Scripts\python.exe" "D:\yoloe\app.py"
pause
