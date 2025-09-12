@echo off
title Digit Recognition - Setup & Run
echo [1/4] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create venv. Make sure Python is installed and added to PATH.
    pause
    exit /b 1
)
echo [2/4] Activating venv and upgrading pip...
call venv\Scripts\activate
python -m pip install --upgrade pip

echo [3/4] Installing requirements (this may take a few minutes)...
pip install -r D:\Majd.A.Alazazi\requirements.txt

echo [4/4] Training the model (first time only)...
python D:\Majd.A.Alazazi\train.py --epochs 3 --batch-size 128

echo Starting the web demo...
python D:\Majd.A.Alazazi\app.py
pause
