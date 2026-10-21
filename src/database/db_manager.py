"""
Gestor de Base de Datos Relacional (SQL / SQLite / MySQL) — Grupo Collins.
Maneja conexiones, esquema de persistencia, carga masiva y consultas analíticas.
"""

from pathlib import Path
import sqlite3
from typing import Optional, Tuple, Dict, Any
import pandas as pd

from ..config import (
    get_database_path,
    TABLA_VENTAS,
    TABLA_PRONOSTICOS,
)


class DatabaseManager:
    """
    Controlador de la base de datos relacional para el Proyecto Integrador.
    Soporta operaciones sobre SQLite local con extensibilidad a motores SQL externos.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else get_database_path()
        # Asegurar existencia del directorio padre
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Devuelve una conexión abierta a la base de datos SQLite."""
        return sqlite3.connect(str(self.db_path))

    def inicializar_tablas(self) -> None:
        """Crea las tablas relacionales normalizadas según el diseño de BD."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Tabla de Ventas Históricas
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLA_VENTAS} (
                    Producto_Descripcion TEXT NOT NULL,
                    Cliente_Descripcion TEXT NOT NULL,
                    Fecha DATETIME NOT NULL,
                    Pedido_Piezas REAL NOT NULL,
                    PRIMARY KEY (Producto_Descripcion, Cliente_Descripcion, Fecha)
                )
            """)

            # 2. Tabla de Pronósticos Activos (Ensemble)
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLA_PRONOSTICOS} (
                    Fecha_Pronostico DATETIME NOT NULL,
                    Producto_Descripcion TEXT NOT NULL,
                    Cliente_Descripcion TEXT NOT NULL,
                    Cluster INTEGER,
                    Pronostico_Ensemble_PedidoPiezas REAL,
                    Pronostico_Min REAL,
                    Pronostico_Max REAL,
                    PRIMARY KEY (Producto_Descripcion, Cliente_Descripcion, Fecha_Pronostico)
                )
            """)

            # Índices de aceleración analítica
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_ventas_prod_cli 
                ON {TABLA_VENTAS} (Producto_Descripcion, Cliente_Descripcion)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_pron_prod_cli 
                ON {TABLA_PRONOSTICOS} (Producto_Descripcion, Cliente_Descripcion)
            """)

            conn.commit()

    def guardar_ventas_historicas(self, df: pd.DataFrame, if_exists: str = "replace") -> int:
        """Persiste los registros históricos de pedidos en la tabla correspondiente."""
        if df is None or df.empty:
            return 0

        df_guardar = df.copy()
        if "Fecha" in df_guardar.columns:
            df_guardar["Fecha"] = pd.to_datetime(df_guardar["Fecha"])

        with self.get_connection() as conn:
            df_guardar.to_sql(TABLA_VENTAS, conn, if_exists=if_exists, index=False)
        return len(df_guardar)

    def guardar_pronosticos_activos(self, df: pd.DataFrame, if_exists: str = "replace") -> int:
        """Persiste el lote de pronósticos generados por el ensamble predictivo."""
        if df is None or df.empty:
            return 0

        df_guardar = df.copy()
        if "Fecha_Pronostico" in df_guardar.columns:
            df_guardar["Fecha_Pronostico"] = pd.to_datetime(df_guardar["Fecha_Pronostico"])

        with self.get_connection() as conn:
            df_guardar.to_sql(TABLA_PRONOSTICOS, conn, if_exists=if_exists, index=False)
        return len(df_guardar)

    def cargar_datos(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Carga tanto el histórico como los pronósticos activos con tipado DateTime.
        Retorna (df_historico, df_pronostico).
        """
        try:
            with self.get_connection() as conn:
                df_hist = pd.read_sql(f"SELECT * FROM {TABLA_VENTAS}", conn)
                df_pron = pd.read_sql(f"SELECT * FROM {TABLA_PRONOSTICOS}", conn)

            if not df_hist.empty and "Fecha" in df_hist.columns:
                df_hist["Fecha"] = pd.to_datetime(df_hist["Fecha"])

            if not df_pron.empty and "Fecha_Pronostico" in df_pron.columns:
                df_pron["Fecha_Pronostico"] = pd.to_datetime(df_pron["Fecha_Pronostico"])

            return df_hist, df_pron
        except Exception:
            return None, None

    def obtener_resumen_estadistico(self) -> Dict[str, Any]:
        """Obtiene métricas y conteos clave de la base de datos."""
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(f"SELECT COUNT(*) FROM {TABLA_VENTAS}")
                total_ventas = cur.fetchone()[0]

                cur.execute(f"SELECT COUNT(*) FROM {TABLA_PRONOSTICOS}")
                total_pronosticos = cur.fetchone()[0]

                cur.execute(f"SELECT COUNT(DISTINCT Producto_Descripcion) FROM {TABLA_VENTAS}")
                prods_ventas = cur.fetchone()[0]

                cur.execute(f"SELECT COUNT(DISTINCT Cliente_Descripcion) FROM {TABLA_VENTAS}")
                clies_ventas = cur.fetchone()[0]

                cur.execute(f"SELECT MIN(Fecha), MAX(Fecha) FROM {TABLA_VENTAS}")
                min_fecha, max_fecha = cur.fetchone()

                cur.execute(f"SELECT MIN(Fecha_Pronostico), MAX(Fecha_Pronostico) FROM {TABLA_PRONOSTICOS}")
                min_f_pron, max_f_pron = cur.fetchone()

            return {
                "total_registros_historicos": total_ventas,
                "total_registros_pronostico": total_pronosticos,
                "num_productos": prods_ventas,
                "num_clientes": clies_ventas,
                "rango_historico": (min_fecha, max_fecha),
                "rango_pronostico": (min_f_pron, max_f_pron),
                "db_path": str(self.db_path),
            }
        except Exception as e:
            return {"error": str(e), "db_path": str(self.db_path)}

    def ejecutar_consulta(self, sql: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """Ejecuta una consulta SQL arbitraria y retorna un DataFrame."""
        with self.get_connection() as conn:
            if params:
                return pd.read_sql_query(sql, conn, params=params)
            return pd.read_sql_query(sql, conn)
