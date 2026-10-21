# Caso de Estudio: Sistema Predictivo de Planificación de Demanda (Demand Planner) impulsado por IA

## 1. El Desafío del Negocio

En la industria farmacéutica, la precisión en el pronóstico de la demanda no es solo una cuestión de eficiencia, sino una necesidad crítica. Un exceso de inventario inmoviliza capital operativo y genera riesgo de caducidad, mientras que el desabastecimiento se traduce directamente en pérdida de ventas y, más importante aún, en posibles afectaciones a la salud de los pacientes. 

Este artículo detalla la conceptualización y desarrollo de una solución integral de *Demand Planning* (Planificación de Demanda) impulsada por Machine Learning. El objetivo central fue transformar datos transaccionales históricos reales en pronósticos accionables a 12 meses, segmentando el comportamiento de ventas para optimizar la toma de decisiones en toda la cadena de suministro.

## 2. Arquitectura de la Solución

El proyecto se diseñó bajo una arquitectura robusta, escalable y dividida en tres capas principales que garantizan la separación de responsabilidades:

1. **Motor de Machine Learning (Backend Analítico):** Un pipeline automatizado desarrollado en Python que gestiona la Extracción, Transformación y Carga (ETL) de los datos, el preprocesamiento matemático, la clusterización no supervisada y el entrenamiento masivo de los modelos predictivos.
2. **Capa de Persistencia (Hub Central de Datos):** Una base de datos relacional ligera que funciona como el puente óptimo entre el motor analítico y la interfaz de usuario. Aquí se almacena de forma estructurada tanto el histórico limpio de ventas como las proyecciones futuras generadas por la IA.
3. **Capa de Presentación (Frontend Ejecutivo):** Un panel de control (Dashboard) interactivo y dinámico diseñado específicamente para abstraer la complejidad matemática y presentar *insights* de forma limpia a los directivos y áreas operativas.

---

## 3. Pipeline de Datos y Metodología de Algoritmos

El núcleo técnico de la solución sigue un ciclo de vida riguroso de ciencia de datos, diseñado para manejar la volatilidad y el ruido natural de las operaciones comerciales B2B.

### 3.1. Ingeniería de Datos y Segmentación Analítica
El procesamiento comienza transformando matrices masivas de datos transaccionales brutos a una estructura de series temporales estandarizadas. Se aplicó un filtrado analítico utilizando el principio de Pareto para concentrar el esfuerzo computacional en los productos y clientes (*Top Performers*) de mayor impacto financiero para la empresa.

### 3.2. Aprendizaje No Supervisado: Descubriendo Patrones (PCA + K-Means)
Dado que las combinaciones de producto y cliente presentan patrones de demanda drásticamente distintos, el sistema implementa un enfoque de segmentación automatizada:
* **Análisis de Componentes Principales (PCA):** Reduce la alta dimensionalidad de las curvas históricas, capturando la varianza más significativa del comportamiento de compra.
* **K-Means Clustering:** Agrupa matemáticamente las series temporales en clústeres optimizados (validados mediante el método del codo). Esto permite categorizar macro-comportamientos sin intervención manual, diferenciando, por ejemplo, los productos de alta rotación constante de aquellos con demanda estacional o intermitente.

### 3.3. Modelado Predictivo: El Poder del "Ensemble"
Para garantizar proyecciones robustas a largo plazo, se descartó el uso de un "modelo único". En su lugar, el sistema entrena dinámicamente un **Ensamble** que combina las fortalezas de tres algoritmos de naturalezas distintas para cada combinación individual de producto y cliente:

1. **SARIMA (Seasonal ARIMA):** Modelo estadístico clásico, excelente para capturar autocorrelaciones, tendencias sostenidas y estacionalidades puras.
2. **Prophet:** Algoritmo aditivo avanzado que maneja de manera excepcional los valores atípicos (*outliers*) y los picos generados por eventos locales y días festivos.
3. **Random Forest Regressor:** Un modelo de Machine Learning basado en árboles de decisión adaptado para series de tiempo mediante ingeniería de características (*feature engineering*), inyectando variables temporales (mes, año, día del año) y banderas de temporadas altas.

**Consolidación:** El sistema promedia algorítmicamente las salidas de los tres modelos. Este enfoque estadístico reduce drásticamente el sesgo y la varianza, blindando el pronóstico contra el sobreajuste (*overfitting*) y logrando un Error Cuadrático Medio (RMSE) sistemáticamente inferior al de cualquier modelo individual. Además, se integraron intervalos de confianza probabilísticos para proyectar escenarios de riesgo "Mínimo" y "Máximo".

---

## 4. Diseño de la Interfaz y UX Corporativa

La salida algorítmica más sofisticada carece de valor si no logra adopción corporativa. Por ello, se desarrolló un *frontend* totalmente orientado al negocio:

* **Navegación *Drill-Down*:** Capacidad de explorar la demanda a nivel macro (visión total de la empresa) o profundizar hasta el nivel más granular (un cliente comprando un producto específico).
* **Métricas Clave (KPIs):** Visualización instantánea de la proyección a futuro con cuantificación de riesgos y promedios operativos mensuales.
* **Visualización Interactiva Inmersiva:** Gráficos dinámicos que empalman sin interrupción la línea de tiempo histórica con el pronóstico del ensamble, incluyendo áreas de sombra paramétricas para visualizar los rangos de confianza.
* **Diseño UI Ejecutivo:** Estilos personalizados para adaptar la plataforma a lineamientos corporativos, utilizando alto contraste para facilitar la lectura de los datos y asegurar una experiencia de usuario limpia y profesional.

---

## 5. Impacto y Valor de Negocio

Esta solución transforma fundamentalmente la manera en que las organizaciones abordan su cadena de suministro. Al automatizar la ingesta de datos y el entrenamiento masivo de modelos predictivos, se elimina la dependencia de proyecciones estáticas en hojas de cálculo y metodologías basadas únicamente en la intuición.

El resultado es una herramienta proactiva impulsada por datos (*data-driven*) que permite a las áreas de Operaciones, Finanzas y Ventas:
* **Optimizar el Capital de Trabajo (Working Capital):** Alineando estratégicamente la producción y las compras con predicciones estadísticas de alta precisión.
* **Minimizar Pérdidas de Ventas (Stockouts):** Anticipando de forma proactiva picos de demanda y estacionalidades complejas.
* **Democratizar el Acceso a la IA:** Otorgando a perfiles ejecutivos no técnicos la capacidad de consumir predicciones matemáticas avanzadas de forma directa y autónoma.
