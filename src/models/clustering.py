"""
Módulo de Agrupamiento No Supervisado (PCA + K-Means) — Segmentación Top 10x5.
Reduce la dimensionalidad de las series temporales y asigna perfiles de comportamiento.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..config import (
    PCA_COMPONENTES,
    K_CLUSTERS,
    KMEANS_RANDOM_STATE,
    COLS_ID,
)


class AnalizadorClusters:
    """
    Segmentador de combinaciones Producto-Cliente mediante reducción de dimensionalidad PCA
    y particionamiento K-Means.
    """

    def __init__(
        self,
        n_components: int = PCA_COMPONENTES,
        n_clusters: int = K_CLUSTERS,
        random_state: int = KMEANS_RANDOM_STATE,
    ):
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )

        self.varianza_explicada: np.ndarray = np.array([])
        self.df_clusters: Optional[pd.DataFrame] = None

    def ejecutar_analisis(self, df_pivot: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de clustering sobre la matriz pivotada.
        Retorna un DataFrame con [Producto_Descripcion, Cliente_Descripcion, Cluster, PCA_1, PCA_2].
        """
        if df_pivot is None or df_pivot.empty:
            raise ValueError("La matriz pivotada está vacía. No es posible ejecutar clustering.")

        # 1. Estandarización de patrones de venta
        x_scaled = self.scaler.fit_transform(df_pivot)

        # 2. Análisis de Componentes Principales (PCA)
        x_pca = self.pca.fit_transform(x_scaled)
        self.varianza_explicada = self.pca.explained_variance_ratio_

        # 3. K-Means
        clusters = self.kmeans.fit_predict(x_pca)

        # 4. Consolidación de Resultados
        df_res = pd.DataFrame(index=df_pivot.index).reset_index()
        df_res["Cluster"] = clusters
        df_res["Cluster_Grupo"] = clusters

        for i in range(self.n_components):
            df_res[f"PCA_{i+1}"] = x_pca[:, i]

        self.df_clusters = df_res
        return self.df_clusters

    def obtener_resumen_metricas(self) -> Dict[str, Any]:
        """Devuelve estadísticas del agrupamiento y varianza explicada."""
        var_total = float(np.sum(self.varianza_explicada)) * 100.0 if len(self.varianza_explicada) > 0 else 0.0
        return {
            "n_clusters": self.n_clusters,
            "n_components_pca": self.n_components,
            "varianza_por_componente": [round(float(v) * 100, 2) for v in self.varianza_explicada],
            "varianza_acumulada_pct": round(var_total, 2),
            "inercia_kmeans": round(float(self.kmeans.inertia_), 2) if hasattr(self.kmeans, "inertia_") else 0.0,
        }
