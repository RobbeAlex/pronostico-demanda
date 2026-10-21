"""
Gestor de Datos y Preprocesamiento ETL — Proyecto Integrador 3 (Grupo Collins)
Transforma el histórico de ventas de formato ancho a largo, estandariza fechas,
y filtra el top de productos y clientes mayoristas para pronóstico.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

from ..config import (
    get_data_csv_path,
    COL_METRICA,
    COLS_ID,
    TOP_PRODUCTOS,
    TOP_CLIENTES,
)


class GestorDatos:
    """
    Clase responsable del flujo de extracción, transformación y carga (ETL)
    para el análisis de series de tiempo de Grupo Collins.
    """

    def __init__(self, ruta_csv: Optional[Path] = None):
        self.ruta_csv = Path(ruta_csv) if ruta_csv else get_data_csv_path()
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_procesado: Optional[pd.DataFrame] = None
        self.df_filtrado: Optional[pd.DataFrame] = None
        self.top_productos: List[str] = []
        self.top_clientes: List[str] = []

    def cargar_datos_crudos(self) -> pd.DataFrame:
        """Carga el archivo CSV con detección automática de codificación."""
        if not self.ruta_csv.exists():
            raise FileNotFoundError(f"No se encontró el archivo de datos en: {self.ruta_csv}")

        try:
            self.df_raw = pd.read_csv(self.ruta_csv, encoding="utf-8")
        except UnicodeDecodeError:
            self.df_raw = pd.read_csv(self.ruta_csv, encoding="latin1")

        return self.df_raw

    def transformar_a_largo(self) -> pd.DataFrame:
        """
        Convierte la matriz ancha (meses en columnas 3 a 84) a formato largo tidy.
        Extrae Fecha en formato DateTime y Pedido_Piezas numérico.
        """
        if self.df_raw is None:
            self.cargar_datos_crudos()

        # Las columnas 3 a 84 corresponden a los meses históricos de Pedido Piezas
        cols_fechas = self.df_raw.columns[3:84]

        # Melt (Ancho -> Largo)
        df_melt = self.df_raw.melt(
            id_vars=COLS_ID,
            value_vars=cols_fechas,
            var_name="Fecha_str",
            value_name=COL_METRICA,
        )

        # Conversión de Fecha ('19-Jan' -> 2019-01-01)
        df_melt["Fecha"] = pd.to_datetime(df_melt["Fecha_str"], format="%y-%b", errors="coerce")
        df_melt[COL_METRICA] = pd.to_numeric(df_melt[COL_METRICA], errors="coerce").fillna(0.0)

        # Agregación por Producto, Cliente y Fecha
        self.df_procesado = (
            df_melt.groupby(COLS_ID + ["Fecha"])[COL_METRICA]
            .sum()
            .reset_index()
            .sort_values(by=COLS_ID + ["Fecha"])
        )

        return self.df_procesado

    def filtrar_top_series(
        self, n_productos: int = TOP_PRODUCTOS, n_clientes: int = TOP_CLIENTES
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Filtra el dataset para retener únicamente los Top N Productos y Top M Clientes
        con mayor volumen acumulado de demanda histórica (Matriz Top 10x5 = 50 series).
        """
        if self.df_procesado is None:
            self.transformar_a_largo()

        # Ranking de productos por volumen total
        ranking_prods = (
            self.df_procesado.groupby("Producto_Descripcion")[COL_METRICA]
            .sum()
            .sort_values(ascending=False)
        )
        self.top_productos = ranking_prods.head(n_productos).index.tolist()

        # Ranking de clientes mayoristas por volumen total
        ranking_clies = (
            self.df_procesado.groupby("Cliente_Descripcion")[COL_METRICA]
            .sum()
            .sort_values(ascending=False)
        )
        self.top_clientes = ranking_clies.head(n_clientes).index.tolist()

        # Filtro de intersección
        mascara = (
            self.df_procesado["Producto_Descripcion"].isin(self.top_productos)
            & self.df_procesado["Cliente_Descripcion"].isin(self.top_clientes)
        )
        self.df_filtrado = self.df_procesado[mascara].copy()

        return self.df_filtrado, self.top_productos, self.top_clientes

    def obtener_datos_para_clustering(self) -> pd.DataFrame:
        """
        Genera la matriz pivot (Index: Producto, Cliente; Columns: Fecha)
        para escalamiento y descomposición PCA.
        """
        if self.df_filtrado is None:
            self.filtrar_top_series()

        return self.df_filtrado.pivot_table(
            index=COLS_ID,
            columns="Fecha",
            values=COL_METRICA,
            fill_value=0.0,
        )

    def ejecutar_pipeline_completo(self) -> pd.DataFrame:
        """Ejecuta secuencialmente carga, transformación y filtrado."""
        self.cargar_datos_crudos()
        self.transformar_a_largo()
        self.filtrar_top_series()
        return self.df_filtrado


# Alias para compatibilidad de nomenclatura
DataLoader = GestorDatos
