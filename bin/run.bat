@echo off
TITLE Car ML Predictor Launcher
COLOR 0A

echo ===================================================
echo      AI Car Assistant Initialization
echo ===================================================

:: Tento trik zaruci, ze skript vzdy zacne ve slozce, kde fyzicky lezi (bin)
cd /d "%~dp0"
:: Presun do hlavniho adresare projektu
cd ..

echo [*] Kontrola pracovni slozky: %CD%

:: Kontrola, jestli pocitac vubec zna prikaz 'python'
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    COLOR 0C
    echo [CHYBA] Python neni nainstalovany, nebo neni pridany v systemovych promennych PATH.
    echo [RESENI] Pri instalaci Pythonu musis zaskrtnout Add Python to PATH.
    pause
    exit /b
)

if not exist ".venv" (
    echo [*] Vytvarim lokalni virtualni prostredi...
    python -m venv .venv

    echo [*] Aktivuji virtualni prostredi...
    call .venv\Scripts\activate.bat

    echo [*] Instaluji potrebne knihovny...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo [*] Virtualni prostredi nalezeno. Aktivuji...
    call .venv\Scripts\activate.bat
)

echo [*] Spoustim aplikaci...
cd src\ui
python app.py

echo.
COLOR 0E
echo Aplikace byla ukoncena, nebo doslo k chybe.
pause