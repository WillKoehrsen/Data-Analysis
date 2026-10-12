@echo off
chcp 65001 >nul
echo ============================================================
echo   AGENTE DE CORRELACIONES ENTRE ETFs - INSTALACION (Windows)
echo ============================================================
echo.
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] No se encontro Python en el PATH.
    echo.
    echo 1) Instala Python 3.10 o superior desde https://www.python.org/downloads/
    echo 2) IMPORTANTE: marca la casilla "Add Python to PATH" en el instalador
    echo 3) Vuelve a ejecutar este archivo
    echo.
    pause
    exit /b 1
)

echo Creando entorno virtual (.venv)...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] No se pudo crear el entorno virtual.
    pause
    exit /b 1
)

echo Activando entorno virtual...
call .venv\Scripts\activate.bat

echo Actualizando pip...
python -m pip install --upgrade pip

echo Instalando dependencias (puede tardar varios minutos)...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   [OK] Instalacion completada.
echo   Ejecuta ahora: EJECUTAR_DEMO.bat        ^(prueba sin internet^)
echo                  EJECUTAR_ANALISIS.bat    ^(datos reales^)
echo                  EJECUTAR_DASHBOARD.bat   ^(interactivo^)
echo ============================================================
pause
