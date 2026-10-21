# Documentación Técnica: Dashboard y Ensamble Estadístico

**Autor:** Roberto Alejandro Chagra Martínez  
**Tecnologías:** Streamlit, SQLite3, Plotly Graph Objects, Pandas, NumPy  
**Tipo:** Tablero Analítico y Desafío de Ensambles Predictivos  

---

## 1. Componentes

* `src/src/app_dashboard.py`: Dashboard interactivo desarrollado con **Streamlit** que permite a los tomadores de decisiones explorar el pronóstico mensual por producto y cliente, visualizar curvas de dispersión y analizar el error residual.
* `src/db_manager.py`: Módulo de conexión a base de datos **SQLite3** con persistencia relacional de predicciones históricas.
* `src/run.py`: Script lanzador para ejecución del entorno Streamlit.
* `SOLUTION_CHALLENGE_ENSAMBLE.ipynb`: Cuaderno de resolución de ensamble de modelos ponderados para minimización de RMSE y MAPE.
