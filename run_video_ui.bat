@echo off
chcp 65001 >nul
cd /d D:\yoloe
echo ============================================================
echo    AIDC Tech Video Processor
echo    Starting... please wait, the browser will open at
echo    http://127.0.0.1:7860
echo ============================================================
echo.
"D:\yoloe\venv\Scripts\python.exe" "D:\yoloe\app.py"
echo.
echo ------------------------------------------------------------
echo  The server has stopped. If there is a red error above,
echo  take a screenshot and send it to the administrator.
echo ------------------------------------------------------------
pause
