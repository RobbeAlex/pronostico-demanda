@echo off
chcp 65001 > nul
title Grupo Collins — Pipeline de Pronostico y BD

echo =========================================================================
echo    EJECUTOR DE PIPELINE DE DATOS Y MACHINE LEARNING (Top 10x5)
echo    Proyecto Integrador 3 · Universidad de Guadalajara (UDG)
echo =========================================================================
echo.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."

set "PYTHON_CMD=python"
if exist "subproyectos\Proyecto Integrador Estadistica\.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%CD%\subproyectos\Proyecto Integrador Estadistica\.venv\Scripts\python.exe"
) else if exist "Proyecto Integrador Estadistica\.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%CD%\Proyecto Integrador Estadistica\.venv\Scripts\python.exe"
)

echo Iniciando ejecucion completa del pipeline...
"%PYTHON_CMD%" main.py --pipeline

echo.
echo Presione cualquier tecla para salir...
pause > nul
