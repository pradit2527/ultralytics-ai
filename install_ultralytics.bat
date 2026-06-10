@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM  Ultralytics YOLO installer (pip + venv, NVIDIA CUDA)
REM  Target folder: D:\yoloe
REM  Just double-click this file, or run it from a terminal.
REM ============================================================

REM ---- Choose your CUDA build here if needed --------------------------------
REM  cu124 works for most recent NVIDIA drivers. If torch.cuda.is_available()
REM  returns False after install, try cu121 or cu128 instead.
set "TORCH_CUDA=cu128"
set "ROOT=D:\yoloe"
REM ---------------------------------------------------------------------------

cd /d "%ROOT%"
echo.
echo === Working directory: %CD%
echo === Using PyTorch CUDA build: %TORCH_CUDA%
echo.

REM ---- Find a usable Python (cu128 wheels support 3.10 - 3.14) --------------
REM  Prefer 3.12 (most third-party wheels), then 3.13, 3.11, 3.10, then 3.14.
set "PYL="
set "PYVER="
for %%V in (3.12 3.13 3.11 3.10 3.14) do (
    if not defined PYL (
        py -%%V --version >nul 2>&1 && set "PYL=py -%%V" && set "PYVER=%%V"
    )
)

if not defined PYL (
    echo [ERROR] No usable Python found (need 3.10 - 3.14).
    echo.
    echo Currently detected on this system:
    py --version 2>nul
    python --version 2>nul
    echo.
    echo Please install Python from https://www.python.org/downloads/
    echo During setup, tick "Add python.exe to PATH", then re-run this script.
    pause
    exit /b 1
)
echo === Found supported Python !PYVER!
echo.

REM ---- Check whether the existing venv is usable ----------------------------
set "NEEDNEW=1"
if exist "%ROOT%\venv\Scripts\python.exe" (
    "%ROOT%\venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<(3,15) else 1)" >nul 2>&1
    if not errorlevel 1 set "NEEDNEW=0"
)

if "!NEEDNEW!"=="1" (
    if exist "%ROOT%\venv" (
        echo === Existing venv uses an unsupported Python version. Removing it...
        rmdir /s /q "%ROOT%\venv"
    )
    echo === Creating a fresh virtual environment with Python !PYVER! ...
    !PYL! -m venv "%ROOT%\venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo === Reusing existing compatible venv at %ROOT%\venv
)
echo.

REM ---- Activate venv ---------------------------------------------------------
call "%ROOT%\venv\Scripts\activate.bat"

REM ---- Show which Python we are using ---------------------------------------
echo === venv Python:
python --version
echo.

REM ---- Upgrade pip -----------------------------------------------------------
echo === Upgrading pip ...
python -m pip install --upgrade pip
echo.

REM ---- Install PyTorch (CUDA) then Ultralytics ------------------------------
echo === Installing PyTorch (%TORCH_CUDA%) + torchvision ...
pip install torch torchvision --index-url https://download.pytorch.org/whl/%TORCH_CUDA%
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed. Try changing TORCH_CUDA at the top of this script.
    pause
    exit /b 1
)

echo.
echo === Installing ultralytics ...
pip install -U ultralytics
if errorlevel 1 (
    echo [ERROR] ultralytics installation failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  VERIFICATION
echo ============================================================
python -c "import torch, ultralytics; print('ultralytics', ultralytics.__version__); print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
echo.
echo Running 'yolo checks' ...
yolo checks

echo.
echo ============================================================
echo  DONE. Installation complete.
echo.
echo  To use it later, open a terminal and run:
echo      %ROOT%\venv\Scripts\activate.bat
echo  Then run yolo / python as usual inside %ROOT%.
echo ============================================================
echo.
pause
endlocal
