"""
Punto de Entrada Secundario para el Dashboard Streamlit dentro del paquete src/.
Redirige a la aplicación principal app.py.
"""

from pathlib import Path
import sys

# Asegurar que el directorio raíz del proyecto esté en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Cargar y ejecutar app.py
from app import *
