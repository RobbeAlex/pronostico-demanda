"""
Implementación del Modelo Estadístico SARIMAX (Statsmodels).
Modela componentes autorregresivos (p,d,q) y estacionales (P,D,Q)s con inversión matricial.
"""

import warnings
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from .base import ModeloPronostico
from ..config import (
    COL_METRICA,
    SARIMA_ORDER,
    SARIMA_SEASONAL_ORDER,
)


class ModeloSARIMA(ModeloPronostico):
    """
    Modelo autorregresivo integrado de media móvil con componente estacional multiplicativo.
    Formulación: SARIMA(p,d,q) x (P,D,Q)_s
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = SARIMA_ORDER,
        seasonal_order: Tuple[int, int, int, int] = SARIMA_SEASONAL_ORDER,
    ):
        super().__init__(nombre="SARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
        self.result_fit = None
        self.intervalos_incertidumbre: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def entrenar(self, df_train: pd.DataFrame) -> None:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError("La librería 'statsmodels' no está instalada en el entorno.")

        df_s = df_train.copy()
        df_s["Fecha"] = pd.to_datetime(df_s["Fecha"])
        serie = df_s.set_index("Fecha")[COL_METRICA].asfreq("MS").fillna(0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.modelo = SARIMAX(
                serie,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.result_fit = self.modelo.fit(disp=False)
            self.esta_entrenado = True

    def predecir(self, fechas_futuras: pd.DatetimeIndex) -> np.ndarray:
        if not self.esta_entrenado or self.result_fit is None:
            raise ValueError("El modelo SARIMA debe ser entrenado antes de predecir.")

        pasos = len(fechas_futuras)
        pred_res = self.result_fit.get_forecast(steps=pasos)
        yhat = np.maximum(pred_res.predicted_mean.values, 0.0)

        # Intervalos de confianza
        try:
            ci = pred_res.conf_int(alpha=0.20)  # 80% CI
            lower = np.maximum(ci.iloc[:, 0].values, 0.0)
            upper = np.maximum(ci.iloc[:, 1].values, 0.0)
            self.intervalos_incertidumbre = (lower, upper)
        except Exception:
            self.intervalos_incertidumbre = (yhat * 0.85, yhat * 1.15)

        return yhat
