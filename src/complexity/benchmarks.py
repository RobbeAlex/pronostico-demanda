"""
Análisis de Complejidad Computacional y Benchmarking Empírico — UDG.
Implementa los 6 algoritmos de estudio del Proyecto Integrador y benchmarks
de modelos de Machine Learning con perfiles de escalabilidad temporal O(f(n)).
"""

import time
import random
import math
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd


# =====================================================================
# 1. PROBLEMA 1: SLOWSORT vs QUICKSORT
# =====================================================================
def slowsort(arr: List[int], i: int = 0, j: int = None) -> None:
    """
    Algoritmo de ordenamiento pesimista basado en 'Multiply and Surrender'.
    Complejidad Asintótica Temporal: O(n^(log n / (2 + epsilon))) > O(n^2)
    """
    if j is None:
        j = len(arr) - 1
    if i >= j:
        return
    m = (i + j) // 2
    slowsort(arr, i, m)
    slowsort(arr, m + 1, j)
    if arr[m] > arr[j]:
        arr[m], arr[j] = arr[j], arr[m]
    slowsort(arr, i, j - 1)


def quicksort(arr: List[int]) -> List[int]:
    """
    Algoritmo divide y vencerás óptimo.
    Complejidad Promedio: O(n log n) | Espacio: O(log n)
    """
    if len(arr) <= 1:
        return arr
    pivote = arr[len(arr) // 2]
    menores = [x for x in arr if x < pivote]
    iguales = [x for x in arr if x == pivote]
    mayores = [x for x in arr if x > pivote]
    return quicksort(menores) + iguales + quicksort(mayores)


# =====================================================================
# 2. PROBLEMA 2: DETERMINANTE vs PERMANENTE
# =====================================================================
def submatriz(mat: List[List[float]], fila: int, col: int) -> List[List[float]]:
    return [
        [mat[r][c] for c in range(len(mat[r])) if c != col]
        for r in range(len(mat))
        if r != fila
    ]


def determinante_recursivo(mat: List[List[float]]) -> float:
    """
    Cálculo de determinante por expansión de Laplace (cofactores).
    Complejidad Asintótica: O(n!)
    """
    n = len(mat)
    if n == 1:
        return mat[0][0]
    if n == 2:
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

    det = 0.0
    for c in range(n):
        signo = (-1) ** c
        det += signo * mat[0][c] * determinante_recursivo(submatriz(mat, 0, c))
    return det


def permanente_recursivo(mat: List[List[float]]) -> float:
    """
    Permanente de una matriz (idéntico a determinante pero sin alternancia de signos).
    Clase de Complejidad: #P-Completo | O(n!) recursivo / O(n * 2^n) Ryser.
    """
    n = len(mat)
    if n == 1:
        return mat[0][0]
    if n == 2:
        return mat[0][0] * mat[1][1] + mat[0][1] * mat[1][0]

    perm = 0.0
    for c in range(n):
        perm += mat[0][c] * permanente_recursivo(submatriz(mat, 0, c))
    return perm


# =====================================================================
# 3. PROBLEMA 3: TEST DE PRIMALIDAD SOLOVAY-STRASSEN
# =====================================================================
def simbolo_jacobi(a: int, n: int) -> int:
    """Calcula el símbolo de Jacobi (a/n) en O(log^2 n)."""
    if a == 0:
        return 0
    if a == 1:
        return 1
    if a % 2 == 0:
        s = simbolo_jacobi(a // 2, n)
        if n % 8 in [3, 5]:
            s = -s
        return s
    if a >= n:
        return simbolo_jacobi(a % n, n)
    if a % 4 == 3 and n % 4 == 3:
        return -simbolo_jacobi(n, a)
    return simbolo_jacobi(n, a)


def test_solovay_strassen(n: int, k_iter: int = 5) -> bool:
    """
    Test probabilístico de primalidad.
    Complejidad: O(k log^3 n)
    """
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    for _ in range(k_iter):
        a = random.randint(2, n - 1)
        x = simbolo_jacobi(a, n)
        if x == 0:
            return False
        # Criterio de Euler: a^((n-1)/2) mod n == (a/n) mod n
        mod_exp = pow(a, (n - 1) // 2, n)
        jacobi_mod = (x + n) % n
        if mod_exp != jacobi_mod:
            return False
    return True


# =====================================================================
# 4. PROBLEMA 4: TORRES DE HANOI
# =====================================================================
def torres_hanoi(
    n: int,
    origen: str = "A",
    destino: str = "C",
    auxiliar: str = "B",
    contador: List[int] = None,
) -> int:
    """
    Problema de recurrencia T(n) = 2T(n-1) + 1.
    Complejidad Temporal: O(2^n) | Total movimientos requeridos: 2^n - 1
    """
    if contador is None:
        contador = [0]
    if n == 1:
        contador[0] += 1
        return contador[0]

    torres_hanoi(n - 1, origen, auxiliar, destino, contador)
    contador[0] += 1
    torres_hanoi(n - 1, auxiliar, destino, origen, contador)
    return contador[0]


# =====================================================================
# 5. PROBLEMA 5: ALGORITMO DE FLEURY (CAMINO EULERIANO)
# =====================================================================
def fleury_euleriano(grafo_dict: Dict[int, List[int]]) -> List[Tuple[int, int]]:
    """
    Encuentra un circuito o camino Euleriano evitando cruzar aristas 'puente'
    a menos que sea la única opción disponible.
    Complejidad Temporal: O(E^2)
    """
    grafo = {u: list(vecinos) for u, vecinos in grafo_dict.items()}
    aristas_recorridas = []

    def dfs_count(u: int, visitados: set) -> int:
        visitados.add(u)
        count = 1
        for v in grafo.get(u, []):
            if v not in visitados:
                count += dfs_count(v, visitados)
        return count

    def es_arista_valida(u: int, v: int) -> bool:
        if len(grafo[u]) == 1:
            return True
        c1 = dfs_count(u, set())
        grafo[u].remove(v)
        grafo[v].remove(u)
        c2 = dfs_count(u, set())
        grafo[u].append(v)
        grafo[v].append(u)
        return c1 <= c2

    # Nodo de inicio (preferir grado impar si existe)
    u_curr = next(iter(grafo.keys()))
    for nodo, vecinos in grafo.items():
        if len(vecinos) % 2 != 0:
            u_curr = nodo
            break

    while any(len(v) > 0 for v in grafo.values()):
        vecinos = list(grafo.get(u_curr, []))
        if not vecinos:
            break
        siguiente = vecinos[0]
        for v in vecinos:
            if es_arista_valida(u_curr, v):
                siguiente = v
                break
        aristas_recorridas.append((u_curr, siguiente))
        grafo[u_curr].remove(siguiente)
        grafo[siguiente].remove(u_curr)
        u_curr = siguiente

    return aristas_recorridas


# =====================================================================
# 6. PROBLEMA 6: CIFRADO XOR
# =====================================================================
def cifrado_xor(mensaje: str, clave: str) -> str:
    """
    Cifrado simétrico stream por aplicación del operador bitwise XOR (^).
    Complejidad Temporal: O(n) | Espacio: O(n)
    """
    if not clave:
        return mensaje
    resultado = []
    len_clave = len(clave)
    for i, c in enumerate(mensaje):
        k = clave[i % len_clave]
        resultado.append(chr(ord(c) ^ ord(k)))
    return "".join(resultado)


# =====================================================================
# 7. BENCHMARKING RUNNER Y MEDICIÓN EMPÍRICA
# =====================================================================
class BenchmarkAlgoritmos:
    """Mide empíricamente tiempos de CPU en función de n para contrastar con O(f(n))."""

    @staticmethod
    def benchmark_ordenamiento(tamanos: List[int] = [10, 15, 20, 25, 30]) -> pd.DataFrame:
        resultados = []
        for n in tamanos:
            datos_base = [random.randint(1, 1000) for _ in range(n)]

            # Slowsort
            arr1 = list(datos_base)
            t0 = time.perf_counter()
            slowsort(arr1)
            t_slow = time.perf_counter() - t0

            # Quicksort
            arr2 = list(datos_base)
            t0 = time.perf_counter()
            quicksort(arr2)
            t_quick = time.perf_counter() - t0

            resultados.append({
                "n": n,
                "Slowsort_seg": round(t_slow, 6),
                "Quicksort_seg": round(t_quick, 6),
                "Ratio_Slow_vs_Quick": round(t_slow / max(t_quick, 1e-9), 2),
            })
        return pd.DataFrame(resultados)

    @staticmethod
    def benchmark_hanoi(discos: List[int] = [5, 10, 15, 18, 20]) -> pd.DataFrame:
        resultados = []
        for d in discos:
            t0 = time.perf_counter()
            movs = torres_hanoi(d)
            t_total = time.perf_counter() - t0
            resultados.append({
                "Discos (n)": d,
                "Movimientos (2^n - 1)": movs,
                "Tiempo_seg": round(t_total, 6),
            })
        return pd.DataFrame(resultados)

    @staticmethod
    def benchmark_determinante_vs_permanente(dimensiones: List[int] = [2, 4, 6, 8, 9]) -> pd.DataFrame:
        resultados = []
        for n in dimensiones:
            mat = [[random.uniform(0.1, 5.0) for _ in range(n)] for _ in range(n)]

            t0 = time.perf_counter()
            det = determinante_recursivo(mat)
            t_det = time.perf_counter() - t0

            t0 = time.perf_counter()
            perm = permanente_recursivo(mat)
            t_perm = time.perf_counter() - t0

            resultados.append({
                "Dimension (n)": n,
                "Det_Tiempo_seg": round(t_det, 6),
                "Perm_Tiempo_seg": round(t_perm, 6),
                "Operaciones_O_n_fact": float(math.factorial(n)),
            })
        return pd.DataFrame(resultados)
