@echo off
:: Cambia al directorio donde está este archivo
cd /d "%~dp0"

:: Ejecuta la aplicación de Streamlit
streamlit run app_dashboard.py