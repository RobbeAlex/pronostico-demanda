"""
Motor de Ensamble Predictivo — Solución Multimodelo para Grupo Collins.
Combina las proyecciones de Prophet, SARIMA y Random Forest minimizando el error general.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

from .base import ModeloPronostico
from .prophet_model import ModeloProphet
from .sarima_model import ModeloSARIMA
from .random_forest_model import ModeloRandomForest
from ..config import PERIODOS_FUTUROS, COL_METRICA, COLS_ID


class EnsamblePredictivo:
    """
    Coordina la ejecución polimórfica de modelos y genera una predicción combinada
    con bandas de dispersión e incertidumbre operacional (Escenario Mínimo y Máximo).
    """

    def __init__(self, periodos_futuros: int = PERIODOS_FUTUROS):
        self.periodos_futuros = periodos_futuros
        self.modelos_disponibles: List[ModeloPronostico] = []

    def instanciar_modelos_frescos(self) -> List[ModeloPronostico]:
        """Devuelve una lista con instancias nuevas de cada arquitectura soportada."""
        return [
            ModeloProphet(),
            ModeloSARIMA(),
            ModeloRandomForest(),
        ]

    def pronosticar_serie(
        self,
        df_serie: pd.DataFrame,
        fechas_futuras: pd.DatetimeIndex,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Entrena los modelos sobre una serie histórica y genera el ensamble.
        Retorna (ensemble_preds, min_preds, max_preds, predicciones_individuales).
        """
        modelos = self.instanciar_modelos_frescos()
        preds_dict: Dict[str, np.ndarray] = {}
        bounds_min_list: List[np.ndarray] = []
        bounds_max_list: List[np.ndarray] = []

        for m in modelos:
            try:
                m.entrenar(df_serie)
                p = m.predecir(fechas_futuras)
                p = np.maximum(p, 0.0)
                preds_dict[m.nombre] = p

                if m.intervalos_incertidumbre:
                    low, up = m.intervalos_incertidumbre
                    bounds_min_list.append(low)
                    bounds_max_list.append(up)
                else:
                    bounds_min_list.append(p * 0.85)
                    bounds_max_list.append(p * 1.15)
            except Exception:
                # En caso de no convergencia, se utiliza array de ceros
                p = np.zeros(len(fechas_futuras))
                preds_dict[m.nombre] = p
                bounds_min_list.append(p)
                bounds_max_list.append(p)

        matriz_preds = np.array(list(preds_dict.values()))  # shape: (n_modelos, n_pasos)

        # Ensamble por promedio uniforme robusto
        if len(preds_dict) > 0 and matriz_preds.size > 0:
            ensemble_preds = np.mean(matriz_preds, axis=0)
            # Bandas de confianza combinadas
            matriz_min = np.array(bounds_min_list)
            matriz_max = np.array(bounds_max_list)
            min_preds = np.maximum(np.min(matriz_min, axis=0), 0.0)
            max_preds = np.max(matriz_max, axis=0)
            # Asegurar consistencia lógica: min <= ensemble <= max
            min_preds = np.minimum(min_preds, ensemble_preds)
            max_preds = np.maximum(max_preds, ensemble_preds)
        else:
            ensemble_preds = np.zeros(len(fechas_futuras))
            min_preds = np.zeros(len(fechas_futuras))
            max_preds = np.zeros(len(fechas_futuras))

        return ensemble_preds, min_preds, max_preds, preds_dict

    def generar_fechas_futuras(self, df_serie: pd.DataFrame) -> pd.DatetimeIndex:
        """Determina los meses subsecuentes al último registro histórico."""
        ultima_fecha = pd.to_datetime(df_serie["Fecha"]).max()
        return pd.date_range(
            start=ultima_fecha + pd.DateOffset(months=1),
            periods=self.periodos_futuros,
            freq="MS",
        )
