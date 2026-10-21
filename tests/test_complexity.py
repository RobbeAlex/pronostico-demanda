"""
Pruebas Unitarias — Módulo de Complejidad Computacional.
"""

from src.complexity.benchmarks import (
    slowsort,
    quicksort,
    cifrado_xor,
    torres_hanoi,
    test_solovay_strassen as solovay_strassen_fn,
    determinante_recursivo,
    permanente_recursivo,
)


def test_quicksort():
    arr = [42, 17, 9, 88, 3, 25]
    res = quicksort(arr)
    assert res == sorted(arr)


def test_slowsort():
    arr = [9, 3, 8, 2, 7]
    slowsort(arr)
    assert arr == [2, 3, 7, 8, 9]


def test_cifrado_xor():
    msg = "Farmaceutica Collins 2026"
    clave = "CLAVE_SECRETA"
    cif = cifrado_xor(msg, clave)
    assert cif != msg
    descif = cifrado_xor(cif, clave)
    assert descif == msg


def test_torres_hanoi():
    n = 4
    movs = torres_hanoi(n)
    # Total esperado = 2^4 - 1 = 15
    assert movs == 15


def test_solovay_strassen():
    assert solovay_strassen_fn(17) is True
    assert solovay_strassen_fn(19) is True
    assert solovay_strassen_fn(18) is False
    assert solovay_strassen_fn(100) is False


def test_determinante_y_permanente():
    mat = [[1.0, 2.0], [3.0, 4.0]]
    # Det = 1*4 - 2*3 = -2
    det = determinante_recursivo(mat)
    assert abs(det - (-2.0)) < 1e-6

    # Perm = 1*4 + 2*3 = 10
    perm = permanente_recursivo(mat)
    assert abs(perm - 10.0) < 1e-6
