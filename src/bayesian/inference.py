"""
Inferencia Bayesiana y Descomposición Matemática de la Demanda — Grupo Collins.
Cuantifica la incertidumbre epistémica de la demanda mensual mediante familias conjugadas,
actualización bayesiana de parámetros e intervalos de credibilidad para planificación de stock.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

from ..config import COL_METRICA


class InferenciaBayesianaDemanda:
    """
    Modelo Bayesiano Normal-Normal conjugado para la estimación de demanda media
    e inferencia de riesgo de desabasto (Stockout Risk).
    """

    def __init__(self, mu_0: Optional[float] = None, sigma_0: Optional[float] = None):
        """
        mu_0: Media a priori (conocimiento previo del catálogo farmacéutico).
        sigma_0: Incertidumbre a priori sobre la media.
        """
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0

    def ajustar_a_posteriori(
        self, valores_historicos: np.ndarray
    ) -> Dict[str, Any]:
        """
        Ejecuta la actualización bayesiana formal con los datos observados X = [x_1, ..., x_n].
        """
        datos = np.asarray(valores_historicos, dtype=float)
        datos = datos[datos > 0]  # Filtrar periodos inactivos si existieran

        if len(datos) == 0:
            return {"error": "Sin datos suficientes para inferencia"}

        n = len(datos)
        x_bar = float(np.mean(datos))
        s_var = float(np.var(datos, ddof=1)) if n > 1 else max(x_bar * 0.2, 1.0)
        sigma_muestral = np.sqrt(s_var)

        # Si no se especificaron hiperparámetros a priori, usar prior difuso / empírico
        prior_mu = self.mu_0 if self.mu_0 is not None else x_bar * 0.95
        prior_sigma = self.sigma_0 if self.sigma_0 is not None else sigma_muestral * 1.5

        # Precisiones (tau = 1 / sigma^2)
        tau_0 = 1.0 / (prior_sigma ** 2)
        tau_muestra = n / (sigma_muestral ** 2)

        # Precisión y varianza a posteriori
        tau_n = tau_0 + tau_muestra
        sigma_n = np.sqrt(1.0 / tau_n)

        # Media a posteriori (promedio ponderado por precisión)
        mu_n = (tau_0 * prior_mu + tau_muestra * x_bar) / tau_n

        # Varianza de la predictiva a posteriori (para un nuevo mes futuro)
        sigma_pred = np.sqrt(sigma_muestral ** 2 + sigma_n ** 2)

        # Intervalos de credibilidad bayesiana al 95%
        ci_95_mu = (mu_n - 1.96 * sigma_n, mu_n + 1.96 * sigma_n)
        ci_95_pred = (max(0.0, mu_n - 1.96 * sigma_pred), mu_n + 1.96 * sigma_pred)

        return {
            "n_observaciones": n,
            "media_muestral": round(x_bar, 2),
            "desv_muestral": round(sigma_muestral, 2),
            "prior": {
                "mu_0": round(prior_mu, 2),
                "sigma_0": round(prior_sigma, 2),
            },
            "posterior": {
                "mu_n": round(mu_n, 2),
                "sigma_n": round(sigma_n, 2),
                "ic_95_media": (round(ci_95_mu[0], 2), round(ci_95_mu[1], 2)),
            },
            "predictiva_posterior": {
                "media_futura": round(mu_n, 2),
                "desv_futura": round(sigma_pred, 2),
                "ic_95_demanda": (round(ci_95_pred[0], 2), round(ci_95_pred[1], 2)),
            },
        }

    @staticmethod
    def calcular_probabilidad_desabasto(
        media_pred: float, desv_pred: float, nivel_inventario: float
    ) -> float:
        """
        Calcula P(Demanda > Inventario): Riesgo de quiebre de stock.
        """
        if desv_pred <= 0:
            return 0.0 if nivel_inventario >= media_pred else 1.0
        z = (nivel_inventario - media_pred) / desv_pred
        # 1 - cdf(z)
        prob = 1.0 - float(stats.norm.cdf(z))
        return round(prob, 4)

    @staticmethod
    def generar_curvas_distribucion(
        res_bayesiano: Dict[str, Any], puntos: int = 200
    ) -> Dict[str, np.ndarray]:
        """Genera los puntos (x, y) de las densidades a priori y a posteriori para graficar."""
        p_mu0 = res_bayesiano["prior"]["mu_0"]
        p_sig0 = res_bayesiano["prior"]["sigma_0"]
        post_mu = res_bayesiano["posterior"]["mu_n"]
        post_sig = res_bayesiano["posterior"]["sigma_n"]

        x_min = min(p_mu0 - 3 * p_sig0, post_mu - 3 * post_sig)
        x_max = max(p_mu0 + 3 * p_sig0, post_mu + 3 * post_sig)
        x_vals = np.linspace(max(0, x_min), x_max, puntos)

        y_prior = stats.norm.pdf(x_vals, p_mu0, p_sig0)
        y_post = stats.norm.pdf(x_vals, post_mu, post_sig)

        return {
            "x": x_vals,
            "y_prior": y_prior,
            "y_posterior": y_post,
        }


class DescomposicionMatematicaSerie:
    """
    Descomposición clásica de series de tiempo: Y_t = T_t + S_t + R_t
    Permite aislar la tendencia a largo plazo, el ciclo estacional anual y los residuos estocásticos.
    """

    @staticmethod
    def descomponer(
        df_serie: pd.DataFrame, modelo: str = "additive", periodo: int = 12
    ) -> Optional[Dict[str, pd.Series]]:
        """Aplica seasonal_decompose sobre una serie mensual."""
        if df_serie is None or len(df_serie) < periodo * 2:
            return None

        df_sorted = df_serie.sort_values("Fecha").copy()
        df_sorted["Fecha"] = pd.to_datetime(df_sorted["Fecha"])
        serie = df_sorted.set_index("Fecha")[COL_METRICA].asfreq("MS").fillna(0.0)

        try:
            res = seasonal_decompose(serie, model=modelo, period=periodo)
            return {
                "observada": res.observed,
                "tendencia": res.trend,
                "estacionalidad": res.seasonal,
                "residuos": res.resid,
            }
        except Exception:
            return None
