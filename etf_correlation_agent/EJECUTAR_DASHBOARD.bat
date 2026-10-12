@echo off
chcp 65001 >nul
echo ============================================================
echo   DASHBOARD INTERACTIVO - http://localhost:8501
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

echo Se abrira el navegador en http://localhost:8501
echo Para cerrar el servidor: pulsa Ctrl+C en esta ventana y despues cierrala.
echo.
start "" http://localhost:8501
timeout /t 3 >nul
python -m etf_agent dashboard
pause
