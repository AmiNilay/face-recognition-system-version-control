@echo off
title Biometric Face Recognition - FastAPI Edge Microservice
echo ====================================================================
echo      BIOMETRIC FACE RECOGNITION SYSTEM - EDGE MICROSERVICE
echo      Developer: Nilay Naha ^| Timezone: IST (UTC+05:30)
echo ====================================================================
echo.
echo Starting FastAPI Uvicorn Server on http://127.0.0.1:8000 ...
echo Interactive Swagger Docs: http://127.0.0.1:8000/docs
echo Interactive ReDoc Docs:   http://127.0.0.1:8000/redoc
echo.

if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe main.py --api --port 8000
) else (
    python main.py --api --port 8000
)

pause
