"""
Implementación del Modelo Prophet (Meta) para Series Temporales.
Ajusta componentes aditivos de tendencia, estacionalidad y días festivos de México.
"""

import logging
import warnings
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from .base import ModeloPronostico
from ..config import COL_METRICA, PAIS_FESTIVOS

# Silenciar logs verbosos de Stan / Prophet
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


class ModeloProphet(ModeloPronostico):
    """
    Modelo de pronóstico basado en curvas aditivas por descomposición de Fourier
    y detección de puntos de cambio con regularización Bayesiana L-BFGS.
    """

    def __init__(self, pais_festivos: str = PAIS_FESTIVOS):
        super().__init__(nombre="Prophet")
        self.pais_festivos = pais_festivos
        self.intervalos_incertidumbre: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def entrenar(self, df_train: pd.DataFrame) -> None:
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("La librería 'prophet' no está instalada en el entorno.")

        df_p = df_train.rename(columns={"Fecha": "ds", COL_METRICA: "y"})[["ds", "y"]].copy()
        df_p["ds"] = pd.to_datetime(df_p["ds"])
        df_p["y"] = pd.to_numeric(df_p["y"], errors="coerce").fillna(0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.modelo = Prophet(
                seasonality_mode="additive",
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=True,
                interval_width=0.80,
            )
            if self.pais_festivos:
                try:
                    self.modelo.add_country_holidays(country_name=self.pais_festivos)
                except Exception:
                    pass

            self.modelo.fit(df_p)
            self.esta_entrenado = True

    def predecir(self, fechas_futuras: pd.DatetimeIndex) -> np.ndarray:
        if not self.esta_entrenado or self.modelo is None:
            raise ValueError("El modelo Prophet debe ser entrenado antes de predecir.")

        df_future = pd.DataFrame({"ds": pd.to_datetime(fechas_futuras)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.modelo.predict(df_future)

        yhat = np.maximum(forecast["yhat"].values, 0.0)
        yhat_lower = np.maximum(forecast["yhat_lower"].values, 0.0)
        yhat_upper = np.maximum(forecast["yhat_upper"].values, 0.0)

        self.intervalos_incertidumbre = (yhat_lower, yhat_upper)
        return yhat
