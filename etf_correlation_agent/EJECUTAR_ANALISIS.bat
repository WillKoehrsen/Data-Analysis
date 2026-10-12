@echo off
chcp 65001 >nul
echo ============================================================
echo   ANALISIS CON DATOS REALES - descarga de Yahoo Finance
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

echo Puedes cambiar los ETFs y ventanas editando este archivo ^(linea set TICKERS^).
echo.
rem ---------------------------------------------------------------
rem  PERSONALIZA AQUI: que ETFs y que ventanas analizar
rem  Ejemplos:
rem    set TICKERS=SPY,TLT,GLD,UUP
rem    set TICKERS=SPY,TLT,VWO,AGG,SMH,VIXY
rem    set TICKERS=            (vacio = los 22 ETFs por defecto)
rem  Ventanas: max,ytd,3m,1m,6m,1y,3y,5y
rem ---------------------------------------------------------------
set TICKERS=
set WINDOWS=max,ytd,3m,1m

if "%TICKERS%"=="" (
    python -m etf_agent analyze --windows %WINDOWS%
) else (
    python -m etf_agent analyze --tickers %TICKERS% --windows %WINDOWS%
)

echo.
echo Informe generado en la carpeta: reports\
echo Abre el archivo report.html con tu navegador.
echo.
pause
