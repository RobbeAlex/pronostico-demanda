"""
Implementación del Modelo Machine Learning Random Forest Regressor (Scikit-Learn).
Aprende patrones no lineales y estacionales mediante ensamble de árboles de decisión.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import ModeloPronostico
from ..config import (
    COL_METRICA,
    RF_N_ESTIMADORES,
    RF_MAX_DEPTH,
    RF_RANDOM_STATE,
    RF_FEATURES,
)


class ModeloRandomForest(ModeloPronostico):
    """
    Ensamble de bagging basado en árboles de decisión con ingeniería de características temporales.
    """

    def __init__(
        self,
        n_estimators: int = RF_N_ESTIMADORES,
        max_depth: int = RF_MAX_DEPTH,
        random_state: int = RF_RANDOM_STATE,
        features: Optional[List[str]] = None,
    ):
        super().__init__(nombre="RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.features = features or RF_FEATURES
        self.intervalos_incertidumbre: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def _extraer_caracteristicas(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Genera variables exógenas de calendario a partir de la columna fecha."""
        df = df_in.copy()
        col_fecha = "Fecha" if "Fecha" in df.columns else "ds"
        fechas = pd.to_datetime(df[col_fecha])

        df["year"] = fechas.dt.year
        df["month"] = fechas.dt.month
        df["dayofyear"] = fechas.dt.dayofyear
        df["quarter"] = fechas.dt.quarter
        # Meses típicos de alta demanda farmacéutica estacional (noviembre, diciembre, enero)
        df["is_holiday_month"] = df["month"].isin([11, 12, 1]).astype(int)

        return df

    def entrenar(self, df_train: pd.DataFrame) -> None:
        df_feat = self._extraer_caracteristicas(df_train)
        x = df_feat[self.features]
        y = pd.to_numeric(df_feat[COL_METRICA], errors="coerce").fillna(0.0)

        self.modelo = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.modelo.fit(x, y)
        self.esta_entrenado = True

    def predecir(self, fechas_futuras: pd.DatetimeIndex) -> np.ndarray:
        if not self.esta_entrenado or self.modelo is None:
            raise ValueError("El modelo RandomForest debe ser entrenado antes de predecir.")

        df_future = pd.DataFrame({"Fecha": pd.to_datetime(fechas_futuras)})
        df_feat = self._extraer_caracteristicas(df_future)
        preds = self.modelo.predict(df_feat[self.features])
        yhat = np.maximum(preds, 0.0)

        # Incertidumbre empírica a partir de la dispersión de predicciones entre árboles individuales
        try:
            predicciones_arboles = np.array([tree.predict(df_feat[self.features].values) for tree in self.modelo.estimators_])
            std_empirica = np.std(predicciones_arboles, axis=0)
            lower = np.maximum(yhat - 1.28 * std_empirica, 0.0)
            upper = yhat + 1.28 * std_empirica
            self.intervalos_incertidumbre = (lower, upper)
        except Exception:
            self.intervalos_incertidumbre = (yhat * 0.85, yhat * 1.15)

        return yhat
