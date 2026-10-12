@echo off
chcp 65001 >nul
echo ============================================================
echo   TESTS - comprueba que todo funciona ^(sin internet^)
echo ============================================================
echo.
if not exist .venv (
    echo [AVISO] Primero ejecuta INSTALAR_WINDOWS.bat
    echo.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
cd /d "%~dp0"
python -m pytest tests/ -q
echo.
pause
