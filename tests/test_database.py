"""
Pruebas Unitarias — Módulo de Base de Datos y Persistencia.
"""

from src.database.db_manager import DatabaseManager
from src.database.nosql_manager import NoSQLManager


def test_database_manager():
    db = DatabaseManager()
    stats = db.obtener_resumen_estadistico()
    assert "error" not in stats
    assert stats["total_registros_historicos"] > 0
    assert stats["total_registros_pronostico"] == 600


def test_nosql_manager_perfiles():
    nosql = NoSQLManager()
    perfiles = nosql.obtener_perfiles_usuarios_default()
    assert len(perfiles) == 3
    roles = [p["role"] for p in perfiles]
    assert "dbAdmin" in roles
    assert "demandPlanner" in roles
    assert "auditor" in roles
