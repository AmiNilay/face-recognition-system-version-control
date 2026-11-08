@echo off
title Apple Face ID Multi-Angle Enrollment Wizard
echo ====================================================================
echo    APPLE FACE ID MULTI-ANGLE COMPOSITE ENROLLMENT WIZARD
echo    Developer: Nilay Naha ^| Timezone: IST (UTC+05:30)
echo ====================================================================
echo.
echo Launching guided multi-angle enrollment (Frontal, Left, Right, Up, Down)...
echo.

if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe add_faces_wizard.py
) else (
    python add_faces_wizard.py
)

pause
