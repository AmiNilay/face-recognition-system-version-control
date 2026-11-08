@echo off
echo ========================================
echo Face Recognition System - Easy Install
echo ========================================
echo.

echo [1/3] Creating virtual environment (with system site packages for dlib/face-recognition)...
python -m venv --system-site-packages .venv

echo.
echo [2/3] Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo [3/3] Installing/updating dependencies from requirements.txt...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To run the application:
echo   1. Run: run_gui.bat (for GUI)
echo   2. Or run: run_cli.bat (for CLI)
echo.
pause
