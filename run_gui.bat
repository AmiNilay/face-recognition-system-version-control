@echo off
setlocal
echo ========================================================
echo Starting Face Recognition System (GUI Mode) - IST
echo ========================================================

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment (.venv) not found!
    echo Please run INSTALL.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python main.py --gui
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
)
