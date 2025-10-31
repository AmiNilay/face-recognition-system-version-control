@echo off
echo ================================
echo Building Face Recognition System
echo ================================

echo.
echo Step 1: Cleaning old builds...
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul
del /q *.spec 2>nul

echo.
echo Step 2: Creating executable...
pyinstaller --name="FaceRecognitionSystem" ^
    --onedir ^
    --windowed ^
    --icon=icon.ico ^
    --add-data="src;src" ^
    --add-data="gui;gui" ^
    --add-data="config;config" ^
    --add-data="README.md;." ^
    --hidden-import="PIL._tkinter_finder" ^
    --hidden-import="cv2" ^
    --hidden-import="face_recognition" ^
    --hidden-import="numpy" ^
    --hidden-import="pandas" ^
    --hidden-import="yaml" ^
    main.py

echo.
echo Step 3: Creating required folders in dist...
mkdir "dist\FaceRecognitionSystem\known_faces" 2>nul
mkdir "dist\FaceRecognitionSystem\data" 2>nul
mkdir "dist\FaceRecognitionSystem\captured_images" 2>nul
mkdir "dist\FaceRecognitionSystem\logs" 2>nul

echo.
echo Step 4: Copying additional files...
copy README.md "dist\FaceRecognitionSystem\" 2>nul
copy add_faces.py "dist\FaceRecognitionSystem\" 2>nul

echo.
echo ================================
echo Build Complete!
echo ================================
echo.
echo Executable location: dist\FaceRecognitionSystem\FaceRecognitionSystem.exe
echo.
pause