@echo off
echo ====================================================
echo Building Face Recognition System (v2.1.0 Enterprise)
echo ====================================================

echo.
echo Step 1: Cleaning old builds...
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul
del /q *.spec 2>nul

echo.
echo Step 2: Locating Python and PyInstaller...
set "PYINSTALLER=pyinstaller"
if exist ".venv\Scripts\pyinstaller.exe" (
    set "PYINSTALLER=.venv\Scripts\pyinstaller.exe"
)

set "ICON_PARAM="
if exist "icon.ico" (
    set "ICON_PARAM=--icon=icon.ico"
)

echo.
echo Step 3: Compiling executable with PyInstaller...
%PYINSTALLER% --name="FaceRecognitionSystem" ^
    --onedir ^
    --windowed ^
    %ICON_PARAM% ^
    --add-data="src;src" ^
    --add-data="gui;gui" ^
    --add-data="config;config" ^
    --add-data="README.md;." ^
    --hidden-import="PIL._tkinter_finder" ^
    --hidden-import="cv2" ^
    --hidden-import="face_recognition" ^
    --hidden-import="face_recognition_models" ^
    --hidden-import="numpy" ^
    --hidden-import="pandas" ^
    --hidden-import="yaml" ^
    --hidden-import="scipy" ^
    --hidden-import="scipy.spatial" ^
    --hidden-import="scipy.spatial.distance" ^
    --hidden-import="cryptography" ^
    --hidden-import="pyttsx3" ^
    --hidden-import="pyttsx3.drivers" ^
    --hidden-import="pyttsx3.drivers.sapi5" ^
    --hidden-import="fastapi" ^
    --hidden-import="uvicorn" ^
    --hidden-import="uvicorn.logging" ^
    --hidden-import="uvicorn.loops" ^
    --hidden-import="uvicorn.loops.auto" ^
    --hidden-import="uvicorn.protocols" ^
    --hidden-import="uvicorn.protocols.http" ^
    --hidden-import="uvicorn.protocols.http.auto" ^
    --hidden-import="uvicorn.protocols.websockets" ^
    --hidden-import="uvicorn.protocols.websockets.auto" ^
    --hidden-import="uvicorn.lifespan" ^
    --hidden-import="uvicorn.lifespan.on" ^
    --hidden-import="requests" ^
    --hidden-import="customtkinter" ^
    --exclude-module="tensorflow" ^
    --exclude-module="torch" ^
    --exclude-module="torchvision" ^
    --exclude-module="matplotlib" ^
    --exclude-module="IPython" ^
    --exclude-module="tensorboard" ^
    --exclude-module="notebook" ^
    --exclude-module="jupyter" ^
    --exclude-module="scipy.ndimage" ^
    main.py

if errorlevel 1 (
    echo.
    echo [ERROR] PyInstaller compilation failed!
    exit /b 1
)

echo.
echo Step 4: Creating runtime directories in dist...
mkdir "dist\FaceRecognitionSystem\known_faces" 2>nul
mkdir "dist\FaceRecognitionSystem\data" 2>nul
mkdir "dist\FaceRecognitionSystem\captured_images" 2>nul
mkdir "dist\FaceRecognitionSystem\logs" 2>nul
mkdir "dist\FaceRecognitionSystem\models" 2>nul

echo.
echo Step 5: Copying models and launch scripts...
if exist "models" (
    xcopy /e /i /y "models\*" "dist\FaceRecognitionSystem\models\" 2>nul
)
copy README.md "dist\FaceRecognitionSystem\" 2>nul
copy add_faces.py "dist\FaceRecognitionSystem\" 2>nul
copy add_faces_wizard.py "dist\FaceRecognitionSystem\" 2>nul
copy run_gui.bat "dist\FaceRecognitionSystem\" 2>nul
copy run_api.bat "dist\FaceRecognitionSystem\" 2>nul
copy run_face_id_wizard.bat "dist\FaceRecognitionSystem\" 2>nul
copy run_rtsp_stream.bat "dist\FaceRecognitionSystem\" 2>nul

echo.
echo ====================================================
echo Build Complete!
echo ====================================================
echo Executable location: dist\FaceRecognitionSystem\FaceRecognitionSystem.exe
echo.
if not defined CI (
    if not defined GITHUB_ACTIONS (
        pause
    )
)
