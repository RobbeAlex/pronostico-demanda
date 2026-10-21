import sys
import os
from streamlit.web import cli as stcli

def main():
    # 1. Obtener la ruta absoluta del directorio donde está este archivo (run.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Construir la ruta hacia el dashboard dentro de la carpeta 'src'
    # os.path.join se encarga de usar '\' en Windows o '/' en Linux automáticamente
    script_path = os.path.join(base_dir, "src", "app_dashboard.py")

    # 3. Verificación de seguridad
    if not os.path.exists(script_path):
        print("❌ Error Crítico:")
        print(f"No se encontró el archivo del dashboard en: {script_path}")
        print("Asegúrate de haber creado la carpeta 'src' y movido 'app_dashboard.py' dentro.")
        return

    # 4. Preparar el comando para Streamlit
    # Esto equivale a escribir "streamlit run src/app_dashboard.py" en la terminal
    sys.argv = ["streamlit", "run", script_path]

    # 5. Ejecutar
    print(f"🚀 Iniciando Collins Demand Planner desde: {script_path} ...")
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()