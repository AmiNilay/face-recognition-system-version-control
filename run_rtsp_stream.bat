@echo off
title Face Recognition System - RTSP IP Camera Streaming
echo ====================================================================
echo      FACE RECOGNITION SYSTEM - RTSP IP CAMERA STREAMING
echo      Developer: Nilay Naha ^| Timezone: IST (UTC+05:30)
echo ====================================================================
echo.
set /p RTSP_URL="Enter RTSP Stream URL (e.g. rtsp://admin:pass@192.168.1.100:554/h264): "

if "%RTSP_URL%"=="" (
    echo [ERROR] No RTSP URL provided. Exiting.
    pause
    exit /b
)

echo.
echo Launching Face Recognition on RTSP feed with Motion Gating...
echo.

if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe main.py --rtsp "%RTSP_URL%" --motion
) else (
    python main.py --rtsp "%RTSP_URL%" --motion
)

pause
