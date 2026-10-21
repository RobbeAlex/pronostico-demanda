"""
CLI Unificado — Proyecto Integrador 3 (Grupo Collins)
Punto de control central para ejecutar el pipeline de pronóstico,
lanzar el dashboard interactivo de Streamlit, correr benchmarks de complejidad,
o inspeccionar la persistencia en base de datos.
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# Configurar salida segura para caracteres en consola Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Agregar raíz al sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import get_database_path, get_data_csv_path, get_forecast_csv_path
from src.database.db_manager import DatabaseManager
from src.complexity.benchmarks import BenchmarkAlgoritmos, cifrado_xor, test_solovay_strassen


def comando_status():
    """Muestra el estado general del ecosistema unificado."""
    print("\n" + "=" * 65)
    print("📋 ESTADO DEL PROYECTO INTEGRADOR 3 — GRUPO COLLINS")
    print("=" * 65)
    print(f"Directorio Raíz: {ROOT_DIR}")
    
    csv_path = get_data_csv_path()
    print(f"Archivo CSV Histórico: {'✅ ' + str(csv_path) if csv_path.exists() else '❌ NO ENCONTRADO'}")
    
    db_path = get_database_path()
    print(f"Base de Datos SQLite:   {'✅ ' + str(db_path) if db_path.exists() else '❌ NO ENCONTRADO'}")
    
    forecast_path = get_forecast_csv_path()
    print(f"CSV de Pronósticos:     {'✅ ' + str(forecast_path) if forecast_path.exists() else '⚠️ NO GENERADO'}")
    
    if db_path.exists():
        db = DatabaseManager(db_path)
        stats = db.obtener_resumen_estadistico()
        print("\nEstadísticas en SQLite:")
        print(f"  • Registros Ventas Históricas: {stats.get('total_registros_historicos', 0):,}")
        print(f"  • Registros Pronóstico Activo: {stats.get('total_registros_pronostico', 0):,}")
        print(f"  • Rango Histórico:             {stats.get('rango_historico')}")
        print(f"  • Rango Pronóstico:            {stats.get('rango_pronostico')}")
    print("=" * 65 + "\n")


def comando_init_db():
    """Inicializa la base de datos a partir del CSV histórico."""
    print("\n[Inicializando Base de Datos Relacional]...")
    from src.data.loader import GestorDatos
    gestor = GestorDatos()
    df_procesado = gestor.transformar_a_largo()
    
    db = DatabaseManager()
    db.inicializar_tablas()
    n = db.guardar_ventas_historicas(df_procesado)
    print(f"✅ Base de datos inicializada exitosamente con {n:,} registros históricos.")


def comando_pipeline():
    """Ejecuta el pipeline completo de machine learning y ensamble."""
    from src.pipeline import PipelinePronostico
    pipeline = PipelinePronostico(verbose=True)
    pipeline.ejecutar(guardar_en_bd=True, exportar_csv=True)


def comando_benchmarks():
    """Ejecuta benchmarks de complejidad computacional."""
    print("\n" + "=" * 65)
    print("⚡ EJECUTANDO BENCHMARKS DE COMPLEJIDAD COMPUTACIONAL")
    print("=" * 65)
    
    print("\n1. Slowsort vs Quicksort:")
    df_sort = BenchmarkAlgoritmos.benchmark_ordenamiento(tamanos=[10, 15, 20, 25, 30])
    print(df_sort.to_string(index=False))
    
    print("\n2. Torres de Hanoi O(2^n):")
    df_hanoi = BenchmarkAlgoritmos.benchmark_hanoi(discos=[5, 10, 15, 18])
    print(df_hanoi.to_string(index=False))
    
    print("\n3. Determinante vs Permanente O(n!):")
    df_det = BenchmarkAlgoritmos.benchmark_determinante_vs_permanente(dimensiones=[2, 4, 6, 8])
    print(df_det.to_string(index=False))
    
    print("\n4. Test de Primalidad Solovay-Strassen:")
    for num in [17, 31, 101, 1009, 10007]:
        es_primo = test_solovay_strassen(num)
        print(f"  • n = {num}: {'Primo' if es_primo else 'Compuesto'}")
        
    print("\n5. Cifrado XOR O(n):")
    msg = "Grupo Collins - Pronostico Farmaceutico"
    clave = "KEY2026"
    cif = cifrado_xor(msg, clave)
    descif = cifrado_xor(cif, clave)
    print(f"  • Mensaje:   {msg}")
    print(f"  • Cifrado:   {cif.encode('unicode_escape').decode()}")
    print(f"  • Descifrado: {descif}")
    print("=" * 65 + "\n")


def comando_dashboard():
    """Lanza la aplicación web interactiva en Streamlit."""
    print("\n🚀 Lanzando Dashboard Ejecutivo en Streamlit...")
    app_file = ROOT_DIR / "app.py"
    
    # Intentar ejecutar con el python del venv si existe, o con el sistema
    posibles_venvs = [
        ROOT_DIR / "subproyectos" / "Proyecto Integrador Estadistica" / ".venv" / "Scripts" / "python.exe",
        ROOT_DIR / "Proyecto Integrador Estadistica" / ".venv" / "Scripts" / "python.exe",
        ROOT_DIR / ".venv" / "Scripts" / "python.exe",
    ]
    python_cmd = sys.executable
    for vp in posibles_venvs:
        if vp.exists():
            python_cmd = str(vp)
            break
    
    cmd = [python_cmd, "-m", "streamlit", "run", str(app_file)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServidor de Streamlit detenido.")
    except Exception as e:
        print(f"Error al iniciar Streamlit: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Sistema Integral de Pronóstico de la Demanda — Grupo Farmacéutico Collins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--status", "-s", action="store_true", help="Diagnosticar estado del proyecto y bases de datos")
    parser.add_argument("--dashboard", "-d", action="store_true", help="Lanzar el dashboard interactivo en Streamlit")
    parser.add_argument("--pipeline", "-p", action="store_true", help="Ejecutar el pipeline completo de pronóstico")
    parser.add_argument("--benchmarks", "-b", action="store_true", help="Ejecutar benchmarks de complejidad computacional")
    parser.add_argument("--init-db", "-i", action="store_true", help="Inicializar base de datos con datos históricos")

    args = parser.parse_args()

    if args.status:
        comando_status()
    elif args.dashboard:
        comando_dashboard()
    elif args.pipeline:
        comando_pipeline()
    elif args.benchmarks:
        comando_benchmarks()
    elif args.init_db:
        comando_init_db()
    else:
        # Si no se pasó argumento, mostrar status y ofrecer dashboard
        comando_status()
        print("Uso sugerido:")
        print("  python main.py --dashboard    # Iniciar el tablero visual interactivo")
        print("  python main.py --pipeline     # Correr el pipeline predictivo completo")
        print("  python main.py --benchmarks   # Ejecutar análisis de complejidad")
        print("  python main.py --help         # Ver todas las opciones disponibles\n")


if __name__ == "__main__":
    main()
