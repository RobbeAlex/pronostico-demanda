"""
Superclase Abstracta de Modelos de Pronóstico — Arquitectura POO.
Define la interfaz uniforme para Prophet, SARIMA y Random Forest.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class ModeloPronostico(ABC):
    """
    Superclase abstracta que establece el contrato polimórfico
    para cualquier algoritmo de predicción de series de tiempo en el sistema.
    """

    def __init__(self, nombre: str = "BaseModel"):
        self.nombre: str = nombre
        self.modelo: Any = None
        self.esta_entrenado: bool = False
        self.metricas_validacion: Dict[str, float] = {}

    @abstractmethod
    def entrenar(self, df_train: pd.DataFrame) -> None:
        """
        Ajusta el modelo sobre la serie histórica.
        Debe recibir un DataFrame con columnas ['Fecha', 'Pedido_Piezas'].
        """
        pass

    @abstractmethod
    def predecir(self, fechas_futuras: pd.DatetimeIndex) -> np.ndarray:
        """
        Genera el vector de predicciones para los periodos temporales indicados.
        Retorna un array 1D de números flotantes (no negativos).
        """
        pass

    def calcular_metricas(
        self, y_real: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcula indicadores estadísticos de error:
        RMSE (Root Mean Squared Error), MAE (Mean Absolute Error) y MAPE (Mean Absolute Percentage Error).
        """
        y_real = np.asarray(y_real, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        # Evitar longitudes inconsistentes
        n = min(len(y_real), len(y_pred))
        y_r = y_real[:n]
        y_p = y_pred[:n]

        mae = float(np.mean(np.abs(y_r - y_p)))
        rmse = float(np.sqrt(np.mean((y_r - y_p) ** 2)))

        # Evitar división por cero en MAPE
        mask_no_cero = y_r > 0
        if np.any(mask_no_cero):
            mape = float(np.mean(np.abs((y_r[mask_no_cero] - y_p[mask_no_cero]) / y_r[mask_no_cero])) * 100.0)
        else:
            mape = 0.0

        self.metricas_validacion = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE": round(mape, 2),
        }
        return self.metricas_validacion

    def __repr__(self) -> str:
        estado = "Entrenado" if self.esta_entrenado else "No entrenado"
        return f"<{self.__class__.__name__} (nombre='{self.nombre}', estado={estado})>"
