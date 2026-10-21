"""
Pruebas Unitarias — Módulo de Carga y Preprocesamiento de Datos.
"""

from src.data.loader import GestorDatos
from src.config import TOP_PRODUCTOS, TOP_CLIENTES


def test_gestor_datos_carga():
    gestor = GestorDatos()
    df_raw = gestor.cargar_datos_crudos()
    assert df_raw is not None
    assert len(df_raw) > 0
    assert "Producto_Descripcion" in df_raw.columns
    assert "Cliente_Descripcion" in df_raw.columns


def test_gestor_datos_transformacion():
    gestor = GestorDatos()
    df_long = gestor.transformar_a_largo()
    assert df_long is not None
    assert "Fecha" in df_long.columns
    assert "Pedido_Piezas" in df_long.columns
    assert df_long["Fecha"].dtype.kind == "M"  # Datetime


def test_gestor_datos_filtrado_top():
    gestor = GestorDatos()
    df_filtrado, prods, clies = gestor.filtrar_top_series(TOP_PRODUCTOS, TOP_CLIENTES)
    assert len(prods) == TOP_PRODUCTOS
    assert len(clies) == TOP_CLIENTES
    assert len(df_filtrado) > 0

    df_pivot = gestor.obtener_datos_para_clustering()
    assert not df_pivot.empty
    # Las series Top 10 x 5 = 50 combinaciones
    assert len(df_pivot) == TOP_PRODUCTOS * TOP_CLIENTES
