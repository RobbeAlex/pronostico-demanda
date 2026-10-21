"""
Test Runner Unificado — Proyecto Integrador 3 (Grupo Collins).
Ejecuta todas las pruebas unitarias de datos, modelos, base de datos y complejidad.
"""

import sys
from pathlib import Path

# Configurar salida UTF-8 para consola Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.test_complexity import (
    test_quicksort,
    test_slowsort,
    test_cifrado_xor,
    test_torres_hanoi,
    test_solovay_strassen,
    test_determinante_y_permanente,
)
from tests.test_data_loader import (
    test_gestor_datos_carga,
    test_gestor_datos_transformacion,
    test_gestor_datos_filtrado_top,
)
from tests.test_database import (
    test_database_manager,
    test_nosql_manager_perfiles,
)
from tests.test_models import (
    test_modelo_random_forest,
    test_clustering_kmeans,
)


def run_all_tests():
    pruebas = [
        ("Complejidad: Quicksort O(n log n)", test_quicksort),
        ("Complejidad: Slowsort O(n^log n)", test_slowsort),
        ("Complejidad: Cifrado XOR O(n)", test_cifrado_xor),
        ("Complejidad: Torres de Hanoi O(2^n)", test_torres_hanoi),
        ("Complejidad: Solovay-Strassen O(k log^3 n)", test_solovay_strassen),
        ("Complejidad: Determinante vs Permanente O(n!)", test_determinante_y_permanente),
        ("Datos: Carga CSV Crudo", test_gestor_datos_carga),
        ("Datos: Transformación Ancho a Largo", test_gestor_datos_transformacion),
        ("Datos: Filtrado Top 10x5 Series", test_gestor_datos_filtrado_top),
        ("BD: Persistencia y Estadísticas SQLite", test_database_manager),
        ("BD: Esquema NoSQL y Perfiles RBAC", test_nosql_manager_perfiles),
        ("Modelos: Random Forest Regressor", test_modelo_random_forest),
        ("Modelos: Clustering PCA + K-Means", test_clustering_kmeans),
    ]

    print("\n" + "=" * 65)
    print("🧪 EJECUCIÓN DE PRUEBAS UNITARIAS — SUITE COMPLETA")
    print("=" * 65)

    pasadas = 0
    falladas = 0

    for nombre, fn in pruebas:
        try:
            fn()
            print(f"  ✅ [PASS] {nombre}")
            pasadas += 1
        except Exception as e:
            print(f"  ❌ [FAIL] {nombre}: {e}")
            falladas += 1

    print("=" * 65)
    print(f"Total: {len(pruebas)} | Pasadas: {pasadas} | Falladas: {falladas}")
    if falladas == 0:
        print("🎉 TODAS LAS PRUEBAS PASARON EXITOSAMENTE.")
    else:
        print("⚠️ ALGUNAS PRUEBAS FALLARON.")
    print("=" * 65 + "\n")
    return falladas


if __name__ == "__main__":
    errores = run_all_tests()
    sys.exit(errores)
