"""
Lanzador del Dashboard Ejecutivo — Proyecto Integrador 3 (Grupo Collins).
Detecta automáticamente el entorno virtual de Python y ejecuta Streamlit.
"""

import sys
import subprocess
from pathlib import Path

# Directorio raíz del repositorio (un nivel arriba de scripts/)
ROOT_DIR = Path(__file__).resolve().parent.parent
APP_PATH = ROOT_DIR / "app.py"

# Buscar ejecutables de Python prioritarios (.venv si existe, o el actual)
posibles_venvs = [
    ROOT_DIR / "subproyectos" / "Proyecto Integrador Estadistica" / ".venv" / "Scripts" / "python.exe",
    ROOT_DIR / "Proyecto Integrador Estadistica" / ".venv" / "Scripts" / "python.exe",
    ROOT_DIR / ".venv" / "Scripts" / "python.exe",
]
PYTHON_EXE = sys.executable
for vp in posibles_venvs:
    if vp.exists():
        PYTHON_EXE = str(vp)
        break


def main():
    print("=" * 65)
    print("🚀 Iniciando Collins Demand Forecast Hub (Streamlit)...")
    print(f"Directorio Raíz: {ROOT_DIR}")
    print(f"Python:          {PYTHON_EXE}")
    print(f"App:             {APP_PATH}")
    print("=" * 65)

    if not APP_PATH.exists():
        print(f"❌ Error: No se encontró el archivo de la aplicación en: {APP_PATH}")
        sys.exit(1)

    cmd = [PYTHON_EXE, "-m", "streamlit", "run", str(APP_PATH), "--browser.gatherUsageStats=false"]
    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT_DIR))
    except KeyboardInterrupt:
        print("\nServidor finalizado por el usuario.")
    except Exception as e:
        print(f"\n❌ Error ejecutando Streamlit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
