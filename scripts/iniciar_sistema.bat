@echo off
chcp 65001 > nul
title Grupo Collins — Demand Forecast Hub (Proyecto Integrador 3)

echo =========================================================================
echo    GRUPO FARMACÉUTICO COLLINS — SISTEMA UNIFICADO DE PRONÓSTICO
echo    Proyecto Integrador 3 · Universidad de Guadalajara (UDG)
echo    Autor: Roberto Alejandro Chagra Martínez
echo =========================================================================
echo.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."

set "PYTHON_CMD=python"
if exist "subproyectos\Proyecto Integrador Estadistica\.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%CD%\subproyectos\Proyecto Integrador Estadistica\.venv\Scripts\python.exe"
    echo [OK] Entorno virtual detectado en 'subproyectos\Proyecto Integrador Estadistica\.venv'.
) else if exist "Proyecto Integrador Estadistica\.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%CD%\Proyecto Integrador Estadistica\.venv\Scripts\python.exe"
    echo [OK] Entorno virtual detectado en 'Proyecto Integrador Estadistica\.venv'.
) else (
    echo [INFO] Utilizando intérprete de Python global del sistema.
)

echo.
echo [1/2] Verificando estado del sistema y persistencia...
"%PYTHON_CMD%" main.py --status

echo.
echo [2/2] Iniciando Dashboard Ejecutivo en Streamlit...
echo Por favor espere mientras se levanta el servidor web...
"%PYTHON_CMD%" -m streamlit run app.py --browser.gatherUsageStats=false

pause
