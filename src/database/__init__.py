"""
Módulo de Persistencia Políglota — Bases de Datos Relacionales (SQLite/MySQL) y NoSQL (MongoDB).
"""

from .db_manager import DatabaseManager
from .nosql_manager import NoSQLManager

__all__ = ["DatabaseManager", "NoSQLManager"]
