"""
Orquestador del Pipeline Predictivo y Persistencia — Grupo Collins.
Integra de punta a punta: Ingesta ETL -> PCA/K-Means -> Ensamble POO -> SQL/NoSQL -> Reportes.
"""

from typing import Dict, Any, Optional
import pandas as pd
from tqdm import tqdm

from .config import (
    COLS_ID,
    COL_METRICA,
    get_forecast_csv_path,
    RESULTS_DIR,
)
from .data.loader import GestorDatos
from .database.db_manager import DatabaseManager
from .database.nosql_manager import NoSQLManager
from .models.clustering import AnalizadorClusters
from .models.ensemble import EnsamblePredictivo


class PipelinePronostico:
    """
    Ejecutor central del flujo de trabajo analítico y operacional del proyecto.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.gestor_datos = GestorDatos()
        self.db_manager = DatabaseManager()
        self.nosql_manager = NoSQLManager()
        self.clusterer = AnalizadorClusters()
        self.ensamble = EnsamblePredictivo()

        self.df_historico_top: Optional[pd.DataFrame] = None
        self.df_clusters: Optional[pd.DataFrame] = None
        self.df_pronosticos: Optional[pd.DataFrame] = None

    def log(self, mensaje: str) -> None:
        if self.verbose:
            print(mensaje)

    def ejecutar(self, guardar_en_bd: bool = True, exportar_csv: bool = True) -> pd.DataFrame:
        """
        Ejecuta todas las fases del ciclo de ciencia de datos.
        """
        self.log("\n=======================================================")
        self.log("🚀 INICIANDO PIPELINE UNIFICADO — GRUPO COLLINS (Top 10x5)")
        self.log("=======================================================")

        # FASE 1: ETL Y LIMPIEZA
        self.log("\n[1/5] Ingesta, transformación de ancho a largo y filtrado...")
        self.df_historico_top = self.gestor_datos.ejecutar_pipeline_completo()
        self.log(
            f"✓ Series filtradas: {len(self.gestor_datos.top_productos)} productos x "
            f"{len(self.gestor_datos.top_clientes)} clientes mayoristas."
        )

        # FASE 2: SEGMENTACIÓN NO SUPERVISADA
        self.log("\n[2/5] Segmentación mediante PCA y K-Means (k=4)...")
        df_pivot = self.gestor_datos.obtener_datos_para_clustering()
        self.df_clusters = self.clusterer.ejecutar_analisis(df_pivot)
        res_cluster = self.clusterer.obtener_resumen_metricas()
        self.log(f"✓ Varianza total explicada PCA: {res_cluster['varianza_acumulada_pct']}%")

        # FASE 3: MODELADO MULTIARQUITECTURA Y ENSAMBLE
        self.log("\n[3/5] Entrenamiento iterativo de Prophet, SARIMA y Random Forest...")
        grupos = self.df_historico_top.groupby(COLS_ID)
        total_grupos = len(grupos)
        lista_resultados = []

        iterador = tqdm(grupos, total=total_grupos, desc="Series Top 10x5") if self.verbose else grupos

        for (prod, cli), df_serie in iterador:
            fechas_futuras = self.ensamble.generar_fechas_futuras(df_serie)
            y_ens, y_min, y_max, _ = self.ensamble.pronosticar_serie(df_serie, fechas_futuras)

            # Buscar cluster asignado
            c_val = 0
            if self.df_clusters is not None:
                mask = (
                    (self.df_clusters["Producto_Descripcion"] == prod)
                    & (self.df_clusters["Cliente_Descripcion"] == cli)
                )
                if mask.any():
                    c_val = int(self.df_clusters.loc[mask, "Cluster"].iloc[0])

            df_serie_pron = pd.DataFrame({
                "Fecha_Pronostico": fechas_futuras,
                "Producto_Descripcion": prod,
                "Cliente_Descripcion": cli,
                "Cluster": c_val,
                "Pronostico_Ensemble_PedidoPiezas": y_ens,
                "Pronostico_Min": y_min,
                "Pronostico_Max": y_max,
            })
            lista_resultados.append(df_serie_pron)

        self.df_pronosticos = pd.concat(lista_resultados, ignore_index=True)
        self.log(f"✓ Total predicciones mensuales generadas: {len(self.df_pronosticos)}")

        # FASE 4: PERSISTENCIA EN BASE DE DATOS SQL
        if guardar_en_bd:
            self.log("\n[4/5] Persistiendo registros en base de datos SQLite...")
            self.db_manager.inicializar_tablas()
            # Guardamos todo el histórico completo procesado
            n_hist = self.db_manager.guardar_ventas_historicas(self.gestor_datos.df_procesado)
            n_pron = self.db_manager.guardar_pronosticos_activos(self.df_pronosticos)
            self.log(f"✓ Guardados {n_hist} registros históricos y {n_pron} registros de pronóstico.")

        # FASE 5: EXPORTACIÓN DE ENTREGABLES
        if exportar_csv:
            self.log("\n[5/5] Exportando artefactos consolidados CSV y NoSQL...")
            ruta_csv_salida = get_forecast_csv_path()
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            self.df_pronosticos.to_csv(ruta_csv_salida, index=False)
            self.log(f"✓ Archivo CSV exportado a: {ruta_csv_salida}")

            # Exportar muestras NoSQL
            self.nosql_manager.exportar_colecciones_json(
                df_hist=self.gestor_datos.df_procesado,
                df_pron=self.df_pronosticos,
                limite_muestra=200,
            )
            self.log("✓ Colecciones NoSQL JSON generadas para MongoDB.")

        self.log("\n🎉 PIPELINE FINALIZADO CON ÉXITO.")
        return self.df_pronosticos
