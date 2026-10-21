"""
Gestor NoSQL (MongoDB / Document Store) — Grupo Collins.
Genera estructuras BSON/JSON compatibles con 'nosql_collections.js', gestiona
colecciones documentales (historical_sales, active_forecasts, user_profiles) y
proporciona utilidades para exportación e interoperabilidad.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from ..config import ROOT_DIR


class NoSQLManager:
    """
    Gestiona el modelo de datos NoSQL orientado a documentos (MongoDB)
    definido en el submódulo de Base de Datos del Proyecto Integrador.
    """

    COLECCION_HISTORICO = "historical_sales"
    COLECCION_PRONOSTICOS = "active_forecasts"
    COLECCION_USUARIOS = "user_profiles"

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else ROOT_DIR / "database" / "nosql_export"

    @staticmethod
    def obtener_perfiles_usuarios_default() -> List[Dict[str, Any]]:
        """
        Retorna los perfiles RBAC (Role-Based Access Control) especificados
        en las Políticas de Seguridad y DRP del proyecto.
        """
        return [
            {
                "username": "admin_collins",
                "role": "dbAdmin",
                "email": "admin@grupocollins.com",
                "department": "Tecnologías de la Información",
                "permissions": ["read", "write", "execute", "manage_users", "drp_execute"],
                "created_at": datetime.now().isoformat(),
            },
            {
                "username": "planner_lead",
                "role": "demandPlanner",
                "email": "planeacion@grupocollins.com",
                "department": "Cadena de Suministro y Operaciones",
                "permissions": ["read", "write_forecasts", "export_reports"],
                "created_at": datetime.now().isoformat(),
            },
            {
                "username": "auditor_externo",
                "role": "auditor",
                "email": "auditoria@grupocollins.com",
                "department": "Calidad y Cumplimiento Regulatorio",
                "permissions": ["read_only", "audit_logs_read"],
                "created_at": datetime.now().isoformat(),
            },
        ]

    def dataframe_a_documentos_ventas(
        self, df_hist: pd.DataFrame, limite: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Convierte registros de ventas históricas a documentos BSON/JSON."""
        if df_hist is None or df_hist.empty:
            return []

        df_target = df_hist.head(limite) if limite else df_hist
        docs = []
        for _, row in df_target.iterrows():
            fecha_val = (
                row["Fecha"].strftime("%Y-%m-%d")
                if hasattr(row["Fecha"], "strftime")
                else str(row["Fecha"])
            )
            doc = {
                "producto": str(row["Producto_Descripcion"]),
                "cliente": str(row["Cliente_Descripcion"]),
                "fecha": fecha_val,
                "pedido_piezas": float(row["Pedido_Piezas"]),
                "metadata": {
                    "unidad": "piezas",
                    "canal": "Mayorista Farmacéutico",
                    "version_pipeline": "3.0.0",
                },
            }
            docs.append(doc)
        return docs

    def dataframe_a_documentos_pronosticos(
        self, df_pron: pd.DataFrame, limite: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Convierte registros de pronóstico activo a documentos BSON/JSON."""
        if df_pron is None or df_pron.empty:
            return []

        df_target = df_pron.head(limite) if limite else df_pron
        docs = []
        for _, row in df_target.iterrows():
            fecha_val = (
                row["Fecha_Pronostico"].strftime("%Y-%m-%d")
                if hasattr(row["Fecha_Pronostico"], "strftime")
                else str(row["Fecha_Pronostico"])
            )
            doc = {
                "producto": str(row["Producto_Descripcion"]),
                "cliente": str(row["Cliente_Descripcion"]),
                "cluster": int(row["Cluster"]) if pd.notnull(row.get("Cluster")) else None,
                "fecha_pronostico": fecha_val,
                "pronostico_ensemble": float(row["Pronostico_Ensemble_PedidoPiezas"]),
                "intervalo_confianza": {
                    "min": float(row.get("Pronostico_Min", 0.0)),
                    "max": float(row.get("Pronostico_Max", 0.0)),
                    "amplitud": float(row.get("Pronostico_Max", 0.0) - row.get("Pronostico_Min", 0.0)),
                },
                "modelo_origen": "Ensemble Ponderado (Prophet + SARIMA + RandomForest)",
                "generado_en": datetime.now().isoformat(),
            }
            docs.append(doc)
        return docs

    def exportar_colecciones_json(
        self,
        df_hist: Optional[pd.DataFrame] = None,
        df_pron: Optional[pd.DataFrame] = None,
        limite_muestra: int = 500,
    ) -> Dict[str, Path]:
        """Exporta muestras JSON de las tres colecciones listas para mongoimport."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        archivos_generados = {}

        # 1. Usuarios
        path_users = self.output_dir / f"{self.COLECCION_USUARIOS}.json"
        with open(path_users, "w", encoding="utf-8") as f:
            json.dump(self.obtener_perfiles_usuarios_default(), f, indent=2, ensure_ascii=False)
        archivos_generados[self.COLECCION_USUARIOS] = path_users

        # 2. Histórico
        if df_hist is not None:
            path_hist = self.output_dir / f"{self.COLECCION_HISTORICO}.json"
            docs_hist = self.dataframe_a_documentos_ventas(df_hist, limite=limite_muestra)
            with open(path_hist, "w", encoding="utf-8") as f:
                json.dump(docs_hist, f, indent=2, ensure_ascii=False)
            archivos_generados[self.COLECCION_HISTORICO] = path_hist

        # 3. Pronósticos
        if df_pron is not None:
            path_pron = self.output_dir / f"{self.COLECCION_PRONOSTICOS}.json"
            docs_pron = self.dataframe_a_documentos_pronosticos(df_pron, limite=limite_muestra)
            with open(path_pron, "w", encoding="utf-8") as f:
                json.dump(docs_pron, f, indent=2, ensure_ascii=False)
            archivos_generados[self.COLECCION_PRONOSTICOS] = path_pron

        return archivos_generados
