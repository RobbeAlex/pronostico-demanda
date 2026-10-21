"""
Dashboard Ejecutivo Integral — Proyecto Integrador 3 (Grupo Collins)
Autor: Roberto Alejandro Chagra Martínez
Institución: Universidad de Guadalajara (UDG) — Ingeniería en Ciencia de Datos

Integra los 5 submódulos del proyecto:
1. Estadística y Solution Challenge (Demand Planner & Ensamble)
2. Programación Orientada a Objetos (Arquitectura de Modelos & Clusters)
3. ML & Matemáticas Aplicadas (Inferencia Bayesiana & Descomposición)
4. Complejidad Computacional (Análisis Asintótico & Benchmarking)
5. Base de Datos & DRP (Persistencia Políglota SQL / NoSQL y Seguridad)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Importaciones de la arquitectura unificada src/
from src.config import (
    COL_METRICA,
    COLS_ID,
    TOP_PRODUCTOS,
    TOP_CLIENTES,
    PCA_COMPONENTES,
    K_CLUSTERS,
    SARIMA_ORDER,
    SARIMA_SEASONAL_ORDER,
    RF_N_ESTIMADORES,
    RF_MAX_DEPTH,
)
from src.database.db_manager import DatabaseManager
from src.database.nosql_manager import NoSQLManager
from src.bayesian.inference import InferenciaBayesianaDemanda, DescomposicionMatematicaSerie
from src.complexity.benchmarks import (
    BenchmarkAlgoritmos,
    cifrado_xor,
    torres_hanoi,
    test_solovay_strassen,
)

# ==============================================================================
# CONFIGURACIÓN GENERAL DE LA PÁGINA STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Grupo Collins — Demand Planner & Intelligence Hub",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS Corporativos
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #f7f9fc !important;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0e2c4c 0%, #1e4a78 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .tab-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0e2c4c;
        margin-top: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CARGA DE DATOS CENTRALIZADA
# ==============================================================================
@st.cache_data(show_spinner="Conectando a base de datos de Grupo Collins...")
def obtener_datos():
    db = DatabaseManager()
    df_hist, df_pron = db.cargar_datos()
    return df_hist, df_pron

df_hist, df_pron = obtener_datos()

if df_hist is None or df_pron is None:
    st.error("⚠️ No se pudieron cargar los datos desde la base de datos.")
    st.info("Ejecuta 'python main.py --pipeline' o 'python main.py --init-db' para poblar la base de datos.")
    st.stop()


# ==============================================================================
# HERO HEADER
# ==============================================================================
st.markdown("""
<div class="hero-banner">
    <h1 style="color: white; margin: 0; font-size: 2.2rem;">🏭 Grupo Collins — Sistema Integral de Pronóstico y Operaciones</h1>
    <p style="color: #cbd5e1; margin-top: 6px; font-size: 1.05rem;">
        <b>Proyecto Integrador 3</b> · Ingeniería en Ciencia de Datos · <b>Universidad de Guadalajara (UDG)</b><br>
        <i>Autor: Roberto Alejandro Chagra Martínez · Versión Unificada 3.0.0</i>
    </p>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PESTAÑAS PRINCIPALES (5 DISCIPLINAS UNIFICADAS)
# ==============================================================================
tab_demanda, tab_poo, tab_bayes, tab_comp, tab_db = st.tabs([
    "📈 1. Demand Planner (Estadística)",
    "⚙️ 2. Arquitectura POO (Programación)",
    "🧠 3. Inferencia Bayesiana (ML & Mate)",
    "⚡ 4. Complejidad Computacional",
    "🗄️ 5. Base de Datos & DRP (SQL/NoSQL)",
])


# ==============================================================================
# TAB 1: DEMAND PLANNER (ESTADÍSTICA / SOLUTION CHALLENGE)
# ==============================================================================
with tab_demanda:
    st.markdown('<div class="tab-header">📊 Tablero Ejecutivo de Pronóstico de Demanda Farmacéutica (Top 10x5)</div>', unsafe_allow_html=True)

    # Filtros Interactivos
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        clientes_opts = ["TODOS LOS CLIENTES (GLOBAL)"] + sorted(df_pron['Cliente_Descripcion'].unique().tolist())
        cliente_sel = st.selectbox("Filtrar por Cliente Mayorista:", clientes_opts, key="tab1_cli")

    with col_f2:
        if cliente_sel == "TODOS LOS CLIENTES (GLOBAL)":
            prods_opts = ["TODOS LOS PRODUCTOS (TOTAL)"] + sorted(df_pron['Producto_Descripcion'].unique().tolist())
        else:
            prods_sub = df_pron[df_pron['Cliente_Descripcion'] == cliente_sel]['Producto_Descripcion'].unique().tolist()
            prods_opts = ["TODOS LOS PRODUCTOS (TOTAL)"] + sorted(prods_sub)
        prod_sel = st.selectbox("Filtrar por Producto Farmacéutico (SKU):", prods_opts, key="tab1_prod")

    # Filtrado Dinámico de Vistas
    if cliente_sel == "TODOS LOS CLIENTES (GLOBAL)" and prod_sel == "TODOS LOS PRODUCTOS (TOTAL)":
        pron_view = df_pron.groupby('Fecha_Pronostico')[['Pronostico_Ensemble_PedidoPiezas', 'Pronostico_Min', 'Pronostico_Max']].sum().reset_index()
        hist_view = df_hist.groupby('Fecha')[['Pedido_Piezas']].sum().reset_index()
        titulo_ctx = "Visión Global — Total Empresa (Top 10x5)"
        cluster_val = "Mix Global Multicluster"
    elif cliente_sel == "TODOS LOS CLIENTES (GLOBAL)":
        pron_view = df_pron[df_pron['Producto_Descripcion'] == prod_sel].groupby('Fecha_Pronostico')[['Pronostico_Ensemble_PedidoPiezas', 'Pronostico_Min', 'Pronostico_Max']].sum().reset_index()
        hist_view = df_hist[df_hist['Producto_Descripcion'] == prod_sel].groupby('Fecha')[['Pedido_Piezas']].sum().reset_index()
        titulo_ctx = f"Consolidado Global: {prod_sel}"
        cluster_val = f"Cluster {df_pron[df_pron['Producto_Descripcion'] == prod_sel]['Cluster'].iloc[0]}"
    elif prod_sel == "TODOS LOS PRODUCTOS (TOTAL)":
        pron_view = df_pron[df_pron['Cliente_Descripcion'] == cliente_sel].groupby('Fecha_Pronostico')[['Pronostico_Ensemble_PedidoPiezas', 'Pronostico_Min', 'Pronostico_Max']].sum().reset_index()
        hist_view = df_hist[df_hist['Cliente_Descripcion'] == cliente_sel].groupby('Fecha')[['Pedido_Piezas']].sum().reset_index()
        titulo_ctx = f"Cartera Total Cliente: {cliente_sel}"
        cluster_val = "Mix de Productos"
    else:
        pron_view = df_pron[(df_pron['Cliente_Descripcion'] == cliente_sel) & (df_pron['Producto_Descripcion'] == prod_sel)].copy()
        hist_view = df_hist[(df_hist['Cliente_Descripcion'] == cliente_sel) & (df_hist['Producto_Descripcion'] == prod_sel)].copy()
        titulo_ctx = f"{cliente_sel} | {prod_sel}"
        cluster_val = f"Cluster {pron_view['Cluster'].iloc[0]}" if not pron_view.empty else "N/A"

    pron_view = pron_view.sort_values('Fecha_Pronostico')
    hist_view = hist_view.sort_values('Fecha')

    # Métricas KPIs
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    total_pron = pron_view['Pronostico_Ensemble_PedidoPiezas'].sum() if not pron_view.empty else 0
    total_min = pron_view['Pronostico_Min'].sum() if not pron_view.empty else 0
    total_max = pron_view['Pronostico_Max'].sum() if not pron_view.empty else 0
    prom_mes = total_pron / 12 if total_pron > 0 else 0

    kpi1.metric("Pronóstico 12 Meses", f"{total_pron:,.0f} pzas")
    kpi2.metric("Escenario Mínimo", f"{total_min:,.0f} pzas")
    kpi3.metric("Escenario Máximo", f"{total_max:,.0f} pzas")
    kpi4.metric("Promedio Mensual", f"{prom_mes:,.0f} pzas/mes")
    kpi5.metric("Segmento / Perfil", cluster_val)

    # Gráfico Plotly
    fig = go.Figure()
    if not hist_view.empty:
        fig.add_trace(go.Scatter(
            x=hist_view['Fecha'],
            y=hist_view['Pedido_Piezas'],
            mode='lines',
            name='Histórico Real',
            line=dict(color='#0e2c4c', width=2),
        ))
    if not pron_view.empty:
        fig.add_trace(go.Scatter(
            x=pron_view['Fecha_Pronostico'],
            y=pron_view['Pronostico_Ensemble_PedidoPiezas'],
            mode='lines+markers',
            name='Pronóstico Ensamble',
            line=dict(color='#e65100', width=3, dash='dash'),
            marker=dict(size=7, color='#e65100'),
        ))
        fig.add_trace(go.Scatter(
            x=pron_view['Fecha_Pronostico'],
            y=pron_view['Pronostico_Max'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=pron_view['Fecha_Pronostico'],
            y=pron_view['Pronostico_Min'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(230, 81, 0, 0.18)',
            name='Banda de Incertidumbre [Min, Max]',
            line=dict(width=0),
        ))

    fig.update_layout(
        title=f"Evolución y Proyección a 12 Meses: {titulo_ctx}",
        xaxis_title="Periodo Mensual",
        yaxis_title="Cantidad de Pedidos (Piezas)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabla Desplegable
    with st.expander("📄 Ver y Descargar Tabla Detallada de Proyecciones"):
        st.dataframe(pron_view, use_container_width=True)
        csv_data = pron_view.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Descargar Pronóstico en CSV",
            data=csv_data,
            file_name=f"Pronostico_{cliente_sel[:15]}_{prod_sel[:15]}.csv",
            mime="text/csv",
        )


# ==============================================================================
# TAB 2: ARQUITECTURA POO & MODELOS (PROGRAMACIÓN)
# ==============================================================================
with tab_poo:
    st.markdown('<div class="tab-header">⚙️ Arquitectura Orientada a Objetos, Clases Polimórficas y Clustering</div>', unsafe_allow_html=True)

    col_poo1, col_poo2 = st.columns([1, 1])

    with col_poo1:
        st.subheader("🧩 Jerarquía de Clases y Patrón Polimórfico")
        st.code("""
from abc import ABC, abstractmethod

class ModeloPronostico(ABC):
    @abstractmethod
    def entrenar(self, df_train: pd.DataFrame) -> None: ...
    @abstractmethod
    def predecir(self, fechas_futuras: pd.DatetimeIndex) -> np.ndarray: ...

class ModeloProphet(ModeloPronostico):
    # Descomposición de Fourier + Días Festivos MX

class ModeloSARIMA(ModeloPronostico):
    # SARIMAX (1,1,1) x (1,1,1,12) con inversión matricial

class ModeloRandomForest(ModeloPronostico):
    # Bagging de 100 árboles con features exógenos
        """, language="python")

        st.markdown("""
        **Principios POO implementados:**
        - **Abstracción:** Superclase `ModeloPronostico` que encapsula la lógica de predicción y cálculo de métricas (RMSE, MAE, MAPE).
        - **Polimorfismo:** `EnsamblePredictivo` itera de forma uniforme sobre la lista de modelos `[ModeloProphet(), ModeloSARIMA(), ModeloRandomForest()]`.
        - **Encapsulamiento:** `GestorDatos` y `DatabaseManager` protegen el estado interno de carga y conexión.
        """)

    with col_poo2:
        st.subheader("🎯 Segmentación No Supervisada (PCA + K-Means)")
        # Gráfica de clusters si están disponibles en df_pron
        if 'Cluster' in df_pron.columns:
            df_cluster_dist = df_pron[['Producto_Descripcion', 'Cliente_Descripcion', 'Cluster']].drop_duplicates()
            conteo_clusters = df_cluster_dist['Cluster'].value_counts().reset_index()
            conteo_clusters.columns = ['Cluster', 'Cantidad de Series']

            fig_clust = px.bar(
                conteo_clusters,
                x='Cluster',
                y='Cantidad de Series',
                color='Cluster',
                title="Distribución de las 50 Series Top 10x5 por Cluster (k=4)",
                color_continuous_scale="Tealgrn"
            )
            fig_clust.update_layout(height=340)
            st.plotly_chart(fig_clust, use_container_width=True)

            st.caption(f"Clustering aplicado sobre matriz pivotada de 81 meses estandarizada con StandardScaler.")

    st.markdown("---")
    st.subheader("📋 Resumen de Hiperparámetros del Ensamble")
    hp_data = {
        "Módulo / Modelo": ["SARIMAX", "Random Forest Regressor", "Prophet", "Clustering (K-Means)", "Reducción Dimensional (PCA)"],
        "Hiperparámetros Configurados": [
            f"order={SARIMA_ORDER}, seasonal_order={SARIMA_SEASONAL_ORDER}",
            f"n_estimators={RF_N_ESTIMADORES}, max_depth={RF_MAX_DEPTH}, n_jobs=-1",
            "seasonality_mode='additive', holidays='MX', interval_width=0.80",
            f"n_clusters={K_CLUSTERS}, n_init='auto', random_state=42",
            f"n_components={PCA_COMPONENTES}, StandardScaler",
        ],
        "Objetivo Analítico": [
            "Capturar autocorrelación e inversión matricial estacional anual",
            "Modelar interacciones no lineales y meses festivos de alta demanda",
            "Curvas continuas de Fourier con regularización Stan",
            "Segmentar perfiles de compra homogéneos entre farmacias mayoristas",
            "Filtrar ruido multivariado preservando más del 80% de varianza explicada",
        ]
    }
    st.table(pd.DataFrame(hp_data))


# ==============================================================================
# TAB 3: INFERENCIA BAYESIANA & MATEMÁTICAS (ML & MATEMÁTICAS APLICADAS)
# ==============================================================================
with tab_bayes:
    st.markdown('<div class="tab-header">🧠 Modelado Estocástico, Inferencia Bayesiana y Riesgo de Desabasto</div>', unsafe_allow_html=True)

    st.markdown(r"""
    En el sector farmacéutico, la demanda presenta incertidumbre debida a epidemias estacionales y variabilidad de pedidos.
    Modelamos la demanda mensual media $\mu$ bajo una **familia conjugada Normal-Normal**:
    $$\mu \sim \mathcal{N}(\mu_0, \sigma_0^2) \quad \xrightarrow{\text{Datos } X=\{x_1,\dots,x_n\}} \quad \mu | X \sim \mathcal{N}(\mu_n, \sigma_n^2)$$
    donde $\tau_0 = 1/\sigma_0^2$, $\tau = n/\sigma^2$, $\tau_n = \tau_0 + \tau$, y $\mu_n = \frac{\tau_0 \mu_0 + \tau \bar{x}}{\tau_n}$.
    """)

    col_b1, col_b2 = st.columns([1, 1])

    # Selección de producto para análisis bayesiano
    sku_bayes = col_b1.selectbox("Seleccionar Medicamento para Análisis Estocástico:", sorted(df_hist['Producto_Descripcion'].unique()), key="sku_bayes")

    serie_sku = df_hist[df_hist['Producto_Descripcion'] == sku_bayes].groupby('Fecha')['Pedido_Piezas'].sum().values
    inferencia = InferenciaBayesianaDemanda()
    res_bayes = inferencia.ajustar_a_posteriori(serie_sku)

    with col_b1:
        st.subheader("📊 Actualización Bayesiana de Parámetros")
        if "error" not in res_bayes:
            prior_mu = res_bayes["prior"]["mu_0"]
            prior_sig = res_bayes["prior"]["sigma_0"]
            post_mu = res_bayes["posterior"]["mu_n"]
            post_sig = res_bayes["posterior"]["sigma_n"]
            ic_low, ic_high = res_bayes["posterior"]["ic_95_media"]

            st.write(f"**Observaciones observadas (meses):** {res_bayes['n_observaciones']}")
            st.write(f"**Media Muestral Real ($\overline{{x}}$):** {res_bayes['media_muestral']:,.1f} piezas")
            st.write(f"**Media A Priori ($\mu_0$):** {prior_mu:,.1f} | Desv ($\sigma_0$): {prior_sig:,.1f}")
            st.write(f"**Media A Posteriori ($\mu_n$):** **{post_mu:,.1f}** | Desv ($\sigma_n$): **{post_sig:,.1f}**")
            st.info(f"**Intervalo de Credibilidad Bayesiana (95% HPD):** [{ic_low:,.1f} , {ic_high:,.1f}] piezas/mes")

            # Curvas Prior vs Posterior
            curvas = inferencia.generar_curvas_distribucion(res_bayes)
            fig_bayes = go.Figure()
            fig_bayes.add_trace(go.Scatter(x=curvas["x"], y=curvas["y_prior"], mode='lines', name='Distribución A Priori', line=dict(color='#94a3b8', dash='dot', width=2)))
            fig_bayes.add_trace(go.Scatter(x=curvas["x"], y=curvas["y_posterior"], mode='lines', name='Distribución A Posteriori', line=dict(color='#0e2c4c', width=3)))
            fig_bayes.update_layout(title="Contraste: Distribución A Priori vs A Posteriori", xaxis_title="Demanda Media Mensual (Piezas)", yaxis_title="Densidad de Probabilidad f(x)", height=320)
            st.plotly_chart(fig_bayes, use_container_width=True)

    with col_b2:
        st.subheader("🛡️ Simulación de Nivel de Servicio y Riesgo de Desabasto")
        st.write("Evalúa la probabilidad de agotar inventario ($P(\text{Demanda} > S)$) ante fluctuaciones estocásticas:")

        if "error" not in res_bayes:
            media_pred = res_bayes["predictiva_posterior"]["media_futura"]
            desv_pred = res_bayes["predictiva_posterior"]["desv_futura"]

            inv_sugerido = col_b2.slider(
                "Nivel de Inventario de Seguridad Disponible (Piezas):",
                min_value=int(media_pred * 0.5),
                max_value=int(media_pred * 2.5),
                value=int(media_pred * 1.15),
                step=500
            )

            riesgo_desabasto = inferencia.calcular_probabilidad_desabasto(media_pred, desv_pred, inv_sugerido)
            nivel_servicio = (1.0 - riesgo_desabasto) * 100.0

            c_r1, c_r2 = st.columns(2)
            c_r1.metric("Nivel de Servicio Garantizado", f"{nivel_servicio:.2f}%")
            c_r2.metric("Riesgo de Quiebre de Stock", f"{riesgo_desabasto * 100:.2f}%")

            if riesgo_desabasto > 0.05:
                st.warning("⚠️ El nivel de inventario está por debajo del estándar recomendado del 95% de nivel de servicio farmacéutico.")
            else:
                st.success("✅ Nivel de inventario óptimo para cumplimiento de entregas mayoristas.")


# ==============================================================================
# TAB 4: COMPLEJIDAD COMPUTACIONAL & BENCHMARKING
# ==============================================================================
with tab_comp:
    st.markdown('<div class="tab-header">⚡ Análisis Asintótico y Benchmarking de Algoritmos (O(n), O(n log n), O(2^n), O(n!))</div>', unsafe_allow_html=True)

    st.markdown("""
    En esta sección se evalúan los 6 problemas clásicos analizados en la materia de **Complejidad Computacional**:
    1. **Slowsort vs Quicksort** · 2. **Determinante vs Permanente** · 3. **Solovay-Strassen** · 4. **Torres de Hanoi** · 5. **Algoritmo de Fleury** · 6. **Cifrado XOR**.
    """)

    col_cp1, col_cp2 = st.columns([1, 1])

    with col_cp1:
        st.subheader("🔬 Benchmark 1: Slowsort $O(n^{\log n})$ vs Quicksort $O(n \log n)$")
        if st.button("▶️ Ejecutar Benchmark de Ordenamiento", key="btn_bench_sort"):
            with st.spinner("Midiendo tiempos de ejecución en CPU..."):
                df_sort_bench = BenchmarkAlgoritmos.benchmark_ordenamiento(tamanos=[10, 15, 20, 25, 30, 35])
                st.dataframe(df_sort_bench, use_container_width=True)

                fig_sort = go.Figure()
                fig_sort.add_trace(go.Scatter(x=df_sort_bench['n'], y=df_sort_bench['Slowsort_seg'], mode='lines+markers', name='Slowsort O(n^log n)', line=dict(color='#dc2626', width=2)))
                fig_sort.add_trace(go.Scatter(x=df_sort_bench['n'], y=df_sort_bench['Quicksort_seg'], mode='lines+markers', name='Quicksort O(n log n)', line=dict(color='#16a34a', width=2)))
                fig_sort.update_layout(title="Tiempo de CPU: Slowsort vs Quicksort", xaxis_title="Tamaño del Arreglo (n)", yaxis_title="Tiempo (segundos)", height=320)
                st.plotly_chart(fig_sort, use_container_width=True)

    with col_cp2:
        st.subheader("🗼 Benchmark 2: Torres de Hanoi $O(2^n)$")
        n_discos = st.slider("Número de Discos en Hanoi (n):", min_value=3, max_value=22, value=15)
        movimientos = 2 ** n_discos - 1
        st.write(f"**Total de Movimientos Mínimos ($2^n - 1$):** `{movimientos:,}` operaciones")

        if st.button("▶️ Medir Tiempo de Hanoi", key="btn_hanoi"):
            df_hanoi = BenchmarkAlgoritmos.benchmark_hanoi(discos=[5, 10, 15, 18, 20])
            st.dataframe(df_hanoi, use_container_width=True)

    st.markdown("---")
    st.subheader("🔐 Demostración en Vivo: Cifrado Simétrico XOR $O(n)$")
    col_x1, col_x2 = st.columns(2)
    msg_in = col_x1.text_input("Mensaje a cifrar:", "Grupo Collins - Pronostico Farmaceutico 2026")
    key_in = col_x1.text_input("Clave de cifrado:", "COLLINS_KEY")
    if msg_in and key_in:
        cifrado = cifrado_xor(msg_in, key_in)
        descifrado = cifrado_xor(cifrado, key_in)
        col_x2.write(f"**Texto Cifrado (Raw Bytes):** `{cifrado.encode('unicode_escape').decode()}`")
        col_x2.write(f"**Texto Descifrado (XOR Inverso):** `{descifrado}`")
        col_x2.success("Complejidad temporal estrictamente lineal $O(n)$, espacio adicional $O(n)$.")


# ==============================================================================
# TAB 5: BASE DE DATOS & DRP (SQL / NOSQL)
# ==============================================================================
with tab_db:
    st.markdown('<div class="tab-header">🗄️ Persistencia Políglota (SQLite, MySQL, MongoDB) y Políticas DRP</div>', unsafe_allow_html=True)

    col_db1, col_db2 = st.columns([1, 1])

    with col_db1:
        st.subheader("💾 Base de Datos Relacional SQL (`Collins_Demand_DB.db`)")
        db = DatabaseManager()
        stats = db.obtener_resumen_estadistico()
        st.write(f"**Ubicación de archivo:** `{stats.get('db_path')}`")
        st.write(f"**Total registros históricos (`ventas_historicas`):** `{stats.get('total_registros_historicos'):,}` filas")
        st.write(f"**Total pronósticos persistidos (`pronosticos_activos`):** `{stats.get('total_registros_pronostico'):,}` filas")
        st.write(f"**Rango temporal histórico:** `{stats.get('rango_historico')}`")
        st.write(f"**Rango de pronóstico futuro:** `{stats.get('rango_pronostico')}`")

        # Consola SQL Interactiva
        st.markdown("#### 🔍 Explorador SQL Interactivo")
        query_sql = st.text_area(
            "Escribe una consulta SELECT:",
            "SELECT Producto_Descripcion, Cliente_Descripcion, SUM(Pronostico_Ensemble_PedidoPiezas) as Total_Pronosticado FROM pronosticos_activos GROUP BY Producto_Descripcion, Cliente_Descripcion ORDER BY Total_Pronosticado DESC LIMIT 5;"
        )
        if st.button("Ejecutar Consulta SQL", key="btn_run_sql"):
            try:
                res_query = db.ejecutar_consulta(query_sql)
                st.dataframe(res_query, use_container_width=True)
            except Exception as err:
                st.error(f"Error en consulta SQL: {err}")

    with col_db2:
        st.subheader("🍃 Colecciones NoSQL Documentales (MongoDB)")
        nosql = NoSQLManager()
        perfiles = nosql.obtener_perfiles_usuarios_default()
        st.write(f"**Colecciones configuradas en MongoDB (`demand_db`):**")
        st.markdown("""
        - `historical_sales`: Series de tiempo en formato anidado JSON/BSON.
        - `active_forecasts`: Pronósticos consolidados con bandas de dispersión.
        - `user_profiles`: Roles y control de acceso (RBAC).
        """)

        st.markdown("#### 👥 Perfiles RBAC y Seguridad DRP")
        st.json(perfiles)

        st.markdown("""
        **Métricas de Continuidad de Negocio (DRP):**
        - **RTO (Recovery Time Objective):** < 4 horas ante desastre de servidor.
        - **RPO (Recovery Point Objective):** < 1 hora de transacciones perdidas con réplica en caliente.
        """)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.88rem; padding: 12px;">
    Grupo Farmacéutico Collins S.A. de C.V. · Sistema Unificado de Pronóstico de la Demanda · UDG Centro Universitario de Guadalajara (CUGDL)<br>
    Desarrollado por: <b>Roberto Alejandro Chagra Martínez</b> · Tecnologías: Python, Prophet, Statsmodels, Scikit-Learn, Streamlit, SQLite, MongoDB
</div>
""", unsafe_allow_html=True)
