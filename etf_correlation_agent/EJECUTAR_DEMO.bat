@echo off
chcp 65001 >nul
echo ============================================================
echo   MODO DEMO - datos sinteticos, NO necesita internet
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
python -m etf_agent demo
echo.
echo Informe generado en la carpeta: reports\
echo Abre el archivo report.html con tu navegador.
echo.
pause
