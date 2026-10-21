"""
Módulo de Configuración Centralizada — Proyecto Integrador 3 (Grupo Collins)
Define rutas dinámicas, hiperparámetros de modelos y constantes globales.
"""

from pathlib import Path

# ==========================================
# 1. RESOLUCIÓN DINÁMICA DE RUTAS
# ==========================================
# Ruta raíz del proyecto integrador (2 niveles arriba de este archivo: src/config.py -> Proyecto Integrador 3/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Rutas estándar del proyecto unificado
DATA_DIR = ROOT_DIR / "data"
DATABASE_DIR = ROOT_DIR / "database"
RESULTS_DIR = ROOT_DIR / "results"
DIAGRAMS_DIR = ROOT_DIR / "diagrams"

# Archivos de datos principales (con fallback a submódulos originales si no existieran en la raíz)
CSV_FILENAME = "Datos Históricos de Pedidos y de Facturación.csv"
DB_FILENAME = "Collins_Demand_DB.db"
FORECAST_FILENAME = "Pronostico_Demanda_Top10x5_ENSEMBLE_FINAL.csv"

def get_data_csv_path() -> Path:
    """Retorna la ruta al archivo CSV histórico buscando en orden de prioridad."""
    candidatos = [
        DATA_DIR / CSV_FILENAME,
        ROOT_DIR / "subproyectos" / "Proyecto Integrador Estadistica" / "data" / CSV_FILENAME,
        ROOT_DIR / "subproyectos" / "Proyecto Integrador Programacion" / "data" / CSV_FILENAME,
        ROOT_DIR / "subproyectos" / "Proyecto Integrador ML & Matematicas" / "Machine Learning e Inferencia Bayesiana" / "data" / CSV_FILENAME,
        ROOT_DIR / "Proyecto Integrador Estadistica" / "data" / CSV_FILENAME,
        ROOT_DIR / "Proyecto Integrador Programacion" / "data" / CSV_FILENAME,
        ROOT_DIR / "Proyecto Integrador ML & Matematicas" / "Machine Learning e Inferencia Bayesiana" / "data" / CSV_FILENAME,
    ]
    for p in candidatos:
        if p.exists():
            return p
    return DATA_DIR / CSV_FILENAME

def get_database_path() -> Path:
    """Retorna la ruta a la base de datos SQLite."""
    candidatos = [
        DATABASE_DIR / DB_FILENAME,
        ROOT_DIR / "subproyectos" / "Proyecto Integrador Estadistica" / "database" / DB_FILENAME,
        ROOT_DIR / "subproyectos" / "Proyecto Integrador ML & Matematicas" / "Machine Learning e Inferencia Bayesiana" / "database" / DB_FILENAME,
        ROOT_DIR / "Proyecto Integrador Estadistica" / "database" / DB_FILENAME,
        ROOT_DIR / "Proyecto Integrador ML & Matematicas" / "Machine Learning e Inferencia Bayesiana" / "database" / DB_FILENAME,
    ]
    for p in candidatos:
        if p.exists():
            return p
    return DATABASE_DIR / DB_FILENAME

def get_forecast_csv_path() -> Path:
    """Retorna la ruta al archivo CSV consolidado de pronósticos."""
    candidatos = [
        RESULTS_DIR / FORECAST_FILENAME,
        ROOT_DIR / "subproyectos" / "Proyecto Integrador Estadistica" / "results" / FORECAST_FILENAME,
        ROOT_DIR / "subproyectos" / "Proyecto Integrador ML & Matematicas" / "results" / FORECAST_FILENAME,
        ROOT_DIR / "Proyecto Integrador Estadistica" / "results" / FORECAST_FILENAME,
        ROOT_DIR / "Proyecto Integrador ML & Matematicas" / "results" / FORECAST_FILENAME,
    ]
    for p in candidatos:
        if p.exists():
            return p
    return RESULTS_DIR / FORECAST_FILENAME

# ==========================================
# 2. PARÁMETROS DEL DOMINIO FARMACÉUTICO
# ==========================================
COL_METRICA = "Pedido_Piezas"
COLS_ID = ["Producto_Descripcion", "Cliente_Descripcion"]

TOP_PRODUCTOS = 10
TOP_CLIENTES = 5
SERIES_TOTALES = TOP_PRODUCTOS * TOP_CLIENTES  # 50 series Top 10x5

# Horizonte de pronóstico y división temporal
FECHA_CORTE = "2025-01-01"
PERIODOS_FUTUROS = 12
FREQ_SERIE = "MS"  # Month Start
MIN_DATOS_ENTRENAMIENTO = 12
PAIS_FESTIVOS = "MX"

# ==========================================
# 3. HIPERPARÁMETROS DE MODELOS
# ==========================================
# SARIMA
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)

# Random Forest
RF_N_ESTIMADORES = 100
RF_MAX_DEPTH = 8
RF_RANDOM_STATE = 42
RF_FEATURES = ["year", "month", "dayofyear", "quarter", "is_holiday_month"]

# Clustering no supervisado
PCA_COMPONENTES = 2
K_CLUSTERS = 4
KMEANS_RANDOM_STATE = 42

# Base de datos
TABLA_VENTAS = "ventas_historicas"
TABLA_PRONOSTICOS = "pronosticos_activos"
