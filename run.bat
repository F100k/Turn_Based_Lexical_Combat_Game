@echo off
setlocal enabledelayedexpansion

:: Navigate to repository root directory
cd /d "%~dp0"

echo ========================================================
echo   Turn-Based Lexical Combat Game - Windows Launcher
echo ========================================================
echo.

:: Detect Python
where python >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON_EXE=python"
) else (
    where py >nul 2>nul
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=py"
    ) else (
        echo [ERROR] Python was not found in your PATH!
        echo Please install Python 3.10+ from https://www.python.org/
        echo Be sure to check "Add Python to PATH" during installation.
        echo.
        pause
        exit /b 1
    )
)

:: Check for existing virtualenv or create one
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment 'venv'...
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment '.venv'...
    call ".venv\Scripts\activate.bat"
) else (
    echo [INFO] Virtual environment not found. Creating 'venv'...
    %PYTHON_EXE% -m venv venv
    if exist "venv\Scripts\activate.bat" (
        echo [INFO] Virtual environment created successfully.
        call "venv\Scripts\activate.bat"
        echo [INFO] Installing dependencies from requirements.txt...
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    ) else (
        echo [WARNING] Could not create virtual environment. Running with %PYTHON_EXE%...
    )
)

:: Check if Django is installed
python -c "import django" >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
)

:: Apply database migrations
echo.
echo [INFO] Applying database migrations...
python dev\manage.py migrate
if %errorlevel% neq 0 (
    echo [WARNING] Migrations encountered an issue. Proceeding with caution...
)

echo.
echo ========================================================
echo   Turn-Based Lexical Combat Game is running!
echo   Open your browser at: http://127.0.0.1:8000/
echo   Press Ctrl+C in this window to stop the server.
echo ========================================================
echo.

:: Automatically open browser after 2 seconds in background
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000/"

:: Start Django development server
python dev\manage.py runserver 127.0.0.1:8000

echo.
echo Server stopped.
pause
