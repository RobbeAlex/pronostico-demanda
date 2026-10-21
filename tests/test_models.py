"""
Pruebas Unitarias — Módulo de Modelos y Clustering.
"""

import pandas as pd
import numpy as np
from src.models.random_forest_model import ModeloRandomForest
from src.models.clustering import AnalizadorClusters


def test_modelo_random_forest():
    fechas = pd.date_range("2023-01-01", periods=24, freq="MS")
    valores = np.linspace(1000, 2000, 24) + np.random.normal(0, 50, 24)
    df_train = pd.DataFrame({"Fecha": fechas, "Pedido_Piezas": valores})

    modelo = ModeloRandomForest(n_estimators=10, max_depth=4)
    modelo.entrenar(df_train)
    assert modelo.esta_entrenado is True

    fechas_fut = pd.date_range("2025-01-01", periods=6, freq="MS")
    preds = modelo.predecir(fechas_fut)
    assert len(preds) == 6
    assert np.all(preds >= 0)


def test_clustering_kmeans():
    # Matriz sintética 10x12
    index = pd.MultiIndex.from_tuples([(f"Prod_{i}", f"Cli_{j}") for i in range(2) for j in range(5)], names=["Producto_Descripcion", "Cliente_Descripcion"])
    fechas = pd.date_range("2024-01-01", periods=12, freq="MS")
    df_pivot = pd.DataFrame(np.random.randint(100, 1000, size=(10, 12)), index=index, columns=fechas)

    clusterer = AnalizadorClusters(n_components=2, n_clusters=3)
    df_res = clusterer.ejecutar_analisis(df_pivot)
    assert df_res is not None
    assert "Cluster" in df_res.columns
    assert "PCA_1" in df_res.columns
    assert len(df_res) == 10
