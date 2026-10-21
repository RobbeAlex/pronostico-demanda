# 🏭 Proyecto Integrador 3 — Sistema Unificado de Pronóstico de la Demanda (Grupo Collins)

**Autor:** Roberto Alejandro Chagra Martínez  
**Institución:** Universidad de Guadalajara (UDG) — Centro Universitario de Guadalajara (CUGDL) — Ingeniería en Ciencia de Datos  
**Tecnologías:** Python 3, Prophet, SARIMAX (Statsmodels), Scikit-Learn (Random Forest, K-Means, PCA), Inferencia Bayesiana, Streamlit, Plotly, SQLite3, MongoDB (NoSQL)  
**Licencia:** MIT License (2026)  
**Versión Unificada:** 3.0.0 (Producción / Hub Integral)  

---

## 📋 Descripción General

El Proyecto es una plataforma integral de Ciencia de Datos e Ingeniería de Software orientada a la **planificación estratégica y pronóstico de la demanda**.

El repositorio unifica en una sola solución modular las 5 áreas disciplinares del semestre:
1. **Estadística / Solution Challenge:** Ensamble multimodelo ponderado y reducción de error (RMSE/MAPE).
2. **Programación Orientada a Objetos (POO):** Patrón abstracto polimórfico (`abc.ABC`), modelos encapsulados y clustering no supervisado (`PCA + K-Means`).
3. **Machine Learning & Matemáticas Aplicadas:** Inferencia Bayesiana de demanda (Normal-Normal conjugada), estimación de riesgo de desabasto ($P(\text{Stockout})$) y descomposición matemática de series ($Y_t = T_t + S_t + R_t$).
4. **Complejidad Computacional:** Análisis asintótico y benchmarking empírico de algoritmos ($\mathcal{O}(n)$, $\mathcal{O}(n \log n)$, $\mathcal{O}(2^n)$, $\mathcal{O}(n!)$).
5. **Base de Datos & Continuidad de Negocio (DRP):** Persistencia híbrida relacional (**SQL**) y documental (**MongoDB NoSQL**) con políticas de seguridad RBAC.

---

## 🚀 Inicio Rápido en 1 Clic

### En Windows (Acceso Directo):
Haz doble clic sobre:
- **[`iniciar_sistema.bat`](file:///g:/Mi%20unidad/Drive.alumno.udg.mx/Proyectos/Proyecto%20integrador%203/iniciar_sistema.bat)** (o [`scripts/iniciar_sistema.bat`](file:///g:/Mi%20unidad/Drive.alumno.udg.mx/Proyectos/Proyecto%20integrador%203/scripts/iniciar_sistema.bat)): Diagnostica el sistema y levanta el **Dashboard Ejecutivo en Streamlit** en tu navegador.
- **[`ejecutar_pipeline.bat`](file:///g:/Mi%20unidad/Drive.alumno.udg.mx/Proyectos/Proyecto%20integrador%203/ejecutar_pipeline.bat)** (o [`scripts/ejecutar_pipeline.bat`](file:///g:/Mi%20unidad/Drive.alumno.udg.mx/Proyectos/Proyecto%20integrador%203/scripts/ejecutar_pipeline.bat)): Ejecuta el pipeline completo de ETL, clustering, entrenamiento de modelos y persistencia.

### Mediante Terminal (CLI Unificado):
```bash
# 1. Diagnóstico general de persistencia y bases de datos
python main.py --status

# 2. Iniciar el Dashboard Ejecutivo Integral (Streamlit)
python main.py --dashboard
# (o alternativamente: python scripts/run_dashboard.py)

# 3. Ejecutar la suite completa de 13 pruebas unitarias
python scripts/run_tests.py

# 4. Ejecutar benchmarks de complejidad computacional
python main.py --benchmarks

# 5. Ejecutar el pipeline analítico de pronóstico Top 10x5
python main.py --pipeline
```

---

## 🧩 Estructura de Subcarpetas para GitHub

El repositorio está organizado en directorios claros, estandarizados para GitHub, con archivos `.gitkeep` en cada carpeta para garantizar su indexación:

```text
Proyecto integrador 3/
├── app.py                                # Dashboard Ejecutivo Unificado en Streamlit (5 pestañas)
├── main.py                               # CLI Central (status, dashboard, pipeline, benchmarks, init-db)
├── iniciar_sistema.bat                   # Acceso directo raíz para Windows
├── ejecutar_pipeline.bat                 # Acceso directo raíz para el Pipeline
├── pyproject.toml                        # Empaquetado pip estándar
├── requirements.txt                      # Dependencias fijadas del ecosistema
├── README.md                             # Documentación principal del repositorio
│
├── .github/                              # AUTOMATIZACIÓN Y CI/CD PARA GITHUB
│   └── workflows/
│       └── ci.yml                        # Pipeline de integración continua (pruebas automatizadas)
│
├── docs/                                 # DOCUMENTACIÓN TÉCNICA E INFORMES
│   ├── DOCUMENTACION_UNIFICADA.md        # Documentación técnica maestra consolidada
│   ├── DOCUMENTACION_TECNICA.md          # Documentación preliminar de arquitectura
│   └── README.md                         # Índice de documentación
│
├── scripts/                              # SCRIPTS DE EJECUCIÓN Y PRUEBAS
│   ├── run_dashboard.py                  # Lanzador directo de Streamlit
│   ├── run_tests.py                      # Suite de 13 pruebas unitarias automatizadas
│   ├── iniciar_sistema.bat               # Script batch detallado de arranque
│   └── ejecutar_pipeline.bat             # Script batch para entrenamiento
│
├── src/                                  # PAQUETE DE CÓDIGO FUENTE UNIFICADO
│   ├── config.py                         # Parámetros, rutas dinámicas e hiperparámetros
│   ├── pipeline.py                       # Orquestador del pipeline predictivo de punta a punta
│   ├── app.py                            # Alias del punto de entrada Streamlit
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py                     # Gestor ETL, melt ancho-a-largo y filtrado Top 10x5
│   ├── database/
│   │   ├── __init__.py
│   │   ├── db_manager.py                 # Persistencia relacional SQLite / MySQL
│   │   └── nosql_manager.py              # Exportación documental MongoDB BSON/JSON y perfiles RBAC
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                       # Superclase abstracta ModeloPronostico(ABC)
│   │   ├── prophet_model.py              # Prophet (Fourier + festivos México)
│   │   ├── sarima_model.py               # SARIMAX (1,1,1)x(1,1,1,12)
│   │   ├── random_forest_model.py        # Random Forest Regressor con features de calendario
│   │   ├── clustering.py                 # PCA (2 componentes) + K-Means (k=4)
│   │   └── ensemble.py                   # Ensamble multimodelo con bandas [Min, Max]
│   ├── bayesian/
│   │   ├── __init__.py
│   │   └── inference.py                  # Inferencia Bayesiana conjugada Normal-Normal y riesgo de stock
│   └── complexity/
│       ├── __init__.py
│       └── benchmarks.py                 # Benchmarks de Quicksort, Slowsort, Hanoi, Determinante, XOR
│
├── data/                                 # DATOS HISTÓRICOS CONSOLIDADOS
│   ├── Datos Históricos de Pedidos y de Facturación.csv
│   └── .gitkeep
│
├── database/                             # BASE DE DATOS Y SCRIPTS DDL
│   ├── Collins_Demand_DB.db              # SQLite con 53,946 ventas y 600 pronósticos
│   ├── sql_schema_script.sql             # Esquema DDL SQL
│   ├── nosql_collections.js              # Esquema DDL MongoDB
│   └── .gitkeep
│
├── results/                              # ENTREGABLES Y PRONÓSTICOS
│   ├── Pronostico_Demanda_Top10x5_ENSEMBLE_FINAL.csv
│   └── .gitkeep
│
├── tests/                                # PRUEBAS UNITARIAS AUTOMATIZADAS
│   ├── test_data_loader.py
│   ├── test_database.py
│   ├── test_models.py
│   └── test_complexity.py
│
└── subproyectos/                         # MATERIAS Y SUBMÓDULOS BASE ORIGINALES
    ├── README.md                         # Resumen de los 5 submódulos
    ├── Proyecto Integrador Estadistica/
    ├── Proyecto Integrador Programacion/
    ├── Proyecto Integrador ML & Matematicas/
    ├── Proyecto Integrador Complejidad Computacional/
    └── Proyecto Integrador Base de Datos/
```

---

## 🗂️ Mapeo de Subproyectos Base en `subproyectos/`

| # | Submódulo | Enfoque Principal | Directorio |
| :--- | :--- | :--- | :--- |
| **1** | **Estadística** | **Solution Challenge Ensamble**: combinación ponderada de modelos para minimizar RMSE/MAPE, persistencia en SQLite y Dashboard en Streamlit. | [`subproyectos/Proyecto Integrador Estadistica/`](file:///g:/Mi%20unidad/Drive.alumno.udg.mx/Proyectos/Proyecto%20integrador%203/subproyectos/Proyecto%20Integrador%20Estadistica/) |
| **2** | **Programación** | Motor empresarial bajo **Programación Orientada a Objetos (POO)** con clases abstractas (`abc.ABC`), polimorfismo (`Prophet`, `SARIMAX`, `RandomForest`) y segmentación `PCA + K-Means`. | [`subproyectos/Proyecto Integrador Programacion/`](file:///g:/Mi%20unidad/Drive.alumno.udg.mx/Proyectos/Proyecto%20integrador%203/subproyectos/Proyecto%20Integrador%20Programacion/) |
| **3** | **ML & Matemáticas** | Modelado estocástico, **Inferencia Bayesiana** (distribuciones *a priori* y *a posteriori*), cálculo de riesgo de agotamiento de stock y análisis espectral. | [`subproyectos/Proyecto Integrador ML & Matematicas/`](file:///g:/Mi%20unidad/Drive.alumno.udg.mx/Proyectos/Proyecto%20integrador%203/subproyectos/Proyecto%20Integrador%20ML%20&%20Matematicas/) |
| **4** | **Complejidad Computacional** | Análisis asintótico temporal y espacial ($\mathcal{O}(n)$, $\mathcal{O}(n \log n)$, $\mathcal{O}(2^n)$, $\mathcal{O}(n!)$), benchmarking empírico de algoritmos y reporte en $\LaTeX$. | [`subproyectos/Proyecto Integrador Complejidad Computacional/`](file:///g:/Mi%20unidad/Drive.alumno.udg.mx/Proyectos/Proyecto%20integrador%203/subproyectos/Proyecto%20Integrador%20Complejidad%20Computacional/) |
| **5** | **Base de Datos** | Arquitectura de persistencia poliglota: esquema relacional normalizado (**SQL**), colecciones documentales (**MongoDB NoSQL**), políticas de seguridad y Plan de Recuperación ante Desastres (**DRP**). | [`subproyectos/Proyecto Integrador Base de Datos/`](file:///g:/Mi%20unidad/Drive.alumno.udg.mx/Proyectos/Proyecto%20integrador%203/subproyectos/Proyecto%20Integrador%20Base%20de%20Datos/) |

---

## 📊 Arquitectura del Dashboard Ejecutivo ([`app.py`](file:///g:/Mi%20unidad/Drive.alumno.udg.mx/Proyectos/Proyecto%20integrador%203/app.py))

Al lanzar `app.py`, dispones de 5 pestañas interactivas:
1. **Demand Planner (Estadística):** Visualización interactiva en Plotly de las 50 series temporales (filtro por cliente, producto o global), bandas de incertidumbre $[\text{Min}, \text{Max}]$ y descarga directa en CSV.
2. **Arquitectura POO (Programación):** Explorador de jerarquía polimórfica de clases, hiperparámetros de modelos y gráfico de distribución de clusters $k=4$.
3. **Inferencia Bayesiana (ML & Matemáticas):** Curvas interactivas de distribución A Priori vs A Posteriori, intervalos de credibilidad bayesiana al 95% y calculadora en tiempo real de riesgo de desabasto ($P(\text{Stockout})$) según el inventario de seguridad.
4. **Complejidad Computacional:** Ejecutor de benchmarks en vivo con gráficos de tiempos de CPU comparando $\mathcal{O}(n \log n)$ vs $\mathcal{O}(n^{\log n})$, recursión de Torres de Hanoi $\mathcal{O}(2^n)$, y cifrador simétrico XOR $\mathcal{O}(n)$.
5. **Base de Datos & DRP:** Estadísticas del archivo SQLite [`database/Collins_Demand_DB.db`](file:///g:/Mi%20unidad/Drive.alumno.udg.mx/Proyectos/Proyecto%20integrador%203/database/Collins_Demand_DB.db), consola interactiva para ejecutar consultas SQL personalizadas, explorador de colecciones MongoDB y especificaciones de seguridad (RTO/RPO y matriz RBAC).

---

## 🧪 Validación y Pruebas Unitarias

Para garantizar la integridad y robustez del software, ejecuta:
```bash
python scripts/run_tests.py
```
> **Resultado:** Las 13 pruebas unitarias de carga de datos, modelos predictivos, clustering, persistencia en SQLite/NoSQL y algoritmos de complejidad pasan al 100% (**13/13 ✅**).

---

## 📄 Documentación Técnica Adicional

Para más detalles sobre las formulaciones matemáticas, demostraciones asintóticas y esquemas de base de datos, consulta [`docs/DOCUMENTACION_UNIFICADA.md`](file:///g:/Mi%20unidad/Drive.alumno.udg.mx/Proyectos/Proyecto%20integrador%203/docs/DOCUMENTACION_UNIFICADA.md).
