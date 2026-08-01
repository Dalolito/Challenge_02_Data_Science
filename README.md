# Challenge 02, Equipo TechLogistics DSS

**Curso:** Fundamentos en Ciencia de Datos — Maestría en Ciencia de Datos y Analítica, EAFIT

**Docente:** Jorge Iván Padilla-Buriticá

**Cliente (ficticio):** TechLogistics S.A.S.

**Fecha límite de entrega:** 31 de enero de 2026

**Integrantes del equipo:**

| Nombre completo | Cédula |
| --- | --- |
| Samuel Gutiérrez Jaramillo | 1036449975 |
| David Lopera Londoño | 1011392448 |
| Juan Diego Acuña Giraldo | 1020222381 |

---

## 1. Resumen ejecutivo

TechLogistics S.A.S., un retailer tecnológico ficticio, detectó erosión de margen y caída de
lealtad de clientes, y sospechaba que la causa raíz era la invisibilidad operativa entre sus tres
sistemas (ERP de Inventarios, Logística y Feedback). Se construyó un Sistema de Soporte a la
Decisión (DSS) en Streamlit a partir de 3 datasets crudos (17.000 registros en total) con fallas
de calidad deliberadas. Se confirmó que el 41.2% de las ventas se hacen con margen negativo
(pérdida acumulada de $1.73M USD, distribuida de forma prácticamente pareja entre los 4 canales
de venta — entre 23.1% y 26.6% cada uno — lo que descarta una falla de precios aislada en un
canal y apunta a un problema sistemático de pricing) y que el 17.5% del ingreso
total ($1.74M USD) corresponde a ventas de productos sin registro en inventario. En cambio, no
se encontró evidencia fuerte de que la logística sea el principal motor de la insatisfacción del
cliente: la correlación entre tiempo de entrega y NPS resultó débil incluso en la ciudad más
afectada. El hallazgo con mayor señal de riesgo operativo fue la bodega Occidente, con 356 días
promedio sin revisión de inventario y una tasa de tickets de soporte notablemente correlacionada
con ese rezago (0.36).

## 2. Preguntas de negocio

El reto exige responder 5 preguntas estratégicas obligatorias planteadas por la junta directiva
de TechLogistics:

1. **Fuga de Capital y Rentabilidad** — ¿Los SKU con margen negativo representan una pérdida
   aceptable por volumen o una falla crítica de precios?
2. **Crisis Logística y Cuellos de Botella** — ¿En qué ciudad o bodega la correlación entre
   Tiempo de Entrega y NPS bajo es más fuerte?
3. **Análisis de la Venta Invisible** — ¿Cuál es el impacto financiero de las ventas cuyo SKU no
   está en el maestro de inventario?
4. **Diagnóstico de Fidelidad** — ¿Existen categorías con stock alto pero sentimiento de cliente
   negativo? ¿Es calidad de producto o sobrecosto?
5. **Storytelling de Riesgo Operativo** — ¿Qué bodegas están operando a ciegas y cómo impacta
   esto en la satisfacción final?

Las 5 se responden con evidencia cuantitativa en la pestaña **Análisis Final** del dashboard y en
el Documento de Hallazgos (PDF).

## 3. Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── .env.example               # plantilla sin usar (el proyecto usa secrets.toml, no .env)
├── .gitignore
├── .devcontainer/
│   └── devcontainer.json
├── .streamlit/
│   ├── config.toml           # tema visual de la app
│   └── secrets.toml          # GROQ_API_KEY (NO se versiona, está en .gitignore)
├── data/
│   ├── raw/                  # los 3 CSV originales (NO se versionan, ver .gitignore)
│   └── processed/            # datasets ya limpios, exportados por el notebook
│       ├── inventario_limpio.csv
│       ├── transacciones_limpio.csv
│       └── feedback_limpio.csv
├── notebooks/
│   └── challenge02_exploracion.ipynb   # EDA, limpieza aplicada y decisiones de imputación
├── src/
│   ├── data_loader.py        # lectura de los 3 CSV crudos, sin transformar
│   ├── cleaning.py           # limpieza y curación (Fase 1), con reporte de trazabilidad
│   ├── quality_metrics.py    # Health Score en 4 dimensiones (completitud/unicidad/validez/consistencia)
│   ├── feature_engineering.py  # merge de los 3 datasets + variables derivadas (Fase 2)
│   └── ai_insights.py        # integración con Groq / Llama-3 (Fase 3) — la usa tab_ia_insights.py
├── app/
│   ├── app.py                 # punto de entrada de Streamlit (orquestación, sin lógica de negocio)
│   └── tabs/
│       ├── ui_helpers.py            # componentes visuales compartidos entre pestañas (badges, gráficos fijos, botones)
│       ├── tab_auditoria.py         # Módulo de Transparencia: 4 dimensiones, correcciones, registros marcados
│       ├── tab_operaciones.py       # Preguntas 1, 2 y 3
│       ├── tab_cliente.py           # Preguntas 4 y 5
│       ├── tab_ia_insights.py       # botón de recomendación con Groq
│       ├── tab_resumen_ejecutivo.py # Análisis Final: contexto, hallazgos por pregunta, plan de acción, descarga en PDF
│       └── ai_insights.py           # ⚠️ borrador sin usar, no se importa desde ningún lado — pendiente de limpiar
├── results/
│   ├── figuras/                     # gráficas exportadas en .png (health score antes/después)
│   └── tabla_diagnostico_gigo.csv   # nulidad, duplicados, outliers y acción tomada por dataset
├── assets/
│   └── screenshots/                 # (vacío por ahora)
├── taller_practico/
│   └── Challenge_02_Informe_Hallazgos.pdf   # ⚠️ vacío, sin usar — el PDF real está en docs/
└── docs/
    ├── declaracion_uso_IA.md
    └── Challenge_02_Informe_Hallazgos.pdf   # Documento de Hallazgos oficial, ver sección 5.1
```

## 4. Cómo reproducir el análisis

### 4.1 Notebook de exploración y limpieza

```bash
# 1. Clonar el repositorio
git clone https://github.com/Dalolito/Challenge_02_Data_Science.git
cd Challenge_02_Data_Science

# 2. Crear entorno e instalar dependencias
pip install -r requirements.txt

# 3. Colocar los 3 CSV originales en data/raw/:
#    inventario_central_v2.csv, transacciones_logistica_v2.csv, feedback_clientes_v2.csv

# 4. Ejecutar el notebook de inicio a fin
jupyter notebook notebooks/challenge02_exploracion.ipynb
```

### 4.2 Dashboard (Streamlit)

```bash
# 1. Configurar la API Key de Groq en .streamlit/secrets.toml
#    GROQ_API_KEY = "tu_api_key_real"

# 2. Ejecutar la app desde la raíz del repositorio
streamlit run app/app.py
```

También disponible en línea: **https://challenge02datascience-4j6wtdj9qwmizlnbgyptjt.streamlit.app**

## 5. Cómo navegar el dashboard

El dashboard tiene un **sidebar** a la izquierda con filtros globales (rango de fechas,
categoría, bodega de origen, ciudad destino) y un botón **🔄 Refrescar Análisis** que limpia el
caché y vuelve a correr todo el pipeline desde cero. Todas las pestañas, salvo Auditoría, se
recalculan según los filtros aplicados.

- **🔍 Auditoría** — Módulo de Transparencia. Muestra el Health Score en 4 dimensiones
  (completitud, unicidad, validez, consistencia) antes/después de la limpieza, el detalle de
  cada corrección aplicada (identificación → decisión → justificación), comparaciones antes/
  después fila por fila, los registros marcados como `SKU_Fantasma` / `Ciudad_Invalida` /
  `Cantidad_Corregida` (descargables), y el reporte completo de limpieza en CSV. No reacciona a
  los filtros del sidebar porque trabaja siempre sobre el dataset completo.

- **📦 Operaciones** — Responde las Preguntas 1, 2 y 3: SKU con margen negativo (KPIs, top 15
  por pérdida, desglose por canal), correlación entre tiempo de entrega y NPS por ciudad, e
  impacto financiero de las ventas con SKU fantasma.

- **😊 Cliente** — Responde las Preguntas 4 y 5: cruce de stock disponible, NPS y calificación
  de producto por categoría (paradoja stock/sentimiento), y relación entre antigüedad de
  revisión de inventario y tasa de tickets de soporte por bodega.

- **🤖 Insights de IA** — Genera una recomendación estratégica con Llama-3 (Groq), a partir del
  resumen estadístico de los datos actualmente filtrados. Requiere una `GROQ_API_KEY` válida
  configurada en `.streamlit/secrets.toml` (o en los Secrets de Streamlit Cloud si se corre en
  línea).

- **📋 Análisis Final** — Informe de consultoría completo: contexto del encargo, qué se
  analizó (incluida la matriz de correlación multivariable), un hallazgo por cada una de las 5
  preguntas estratégicas citadas textualmente, y el Plan de Acción priorizado por complejidad.
  Incluye un botón para descargar una versión en PDF generada en tiempo real a partir de los
  filtros actuales — ver sección 5.1 para la diferencia con el Documento de Hallazgos oficial.

### 5.1 Sobre los dos documentos de hallazgos en PDF

Este proyecto entrega **dos PDF distintos**, con propósitos distintos — no es una duplicación
accidental:

1. **`docs/Challenge_02_Informe_Hallazgos.pdf`** — el **Documento de Hallazgos oficial**,
   redactado y diagramado por el equipo. Cita textualmente las 5 preguntas de la junta, incluye
   una gráfica específica por cada hallazgo (no solo el heatmap general), y cierra con una
   sección de Conclusión que valida y matiza la hipótesis inicial de invisibilidad operativa.
   **Este es el entregable de referencia para la evaluación del reto.**

2. **Botón "Descargar análisis completo (PDF)"**, al final de la pestaña **Análisis Final** del
   dashboard — genera un PDF más simple **en tiempo real**, a partir de los datos que estén
   filtrados en ese momento en el sidebar (por ejemplo, si se filtra solo una categoría o un
   rango de fechas específico). Es una herramienta exploratoria del dashboard, útil para
   compartir un corte puntual del análisis; no reemplaza al documento oficial.

## 6. Principales hallazgos

| # | Hallazgo | Evidencia (tabla/figura) | Pregunta que responde |
| --- | --- | --- | --- |
| 1 | 3.400 ventas (41.2%) se hacen con margen negativo, acumulando una pérdida de $1.73M USD, distribuida de forma pareja entre los 4 canales (23.1%-26.6% cada uno) — es un problema sistemático de pricing, no aislado a un canal. | Top 15 SKU con peor margen, tab Operaciones | Pregunta 1 |
| 2 | La correlación entre tiempo de entrega y NPS es débil en todas las ciudades (máximo -0.02 en Bucaramanga) — la logística no aparece como el principal motor de insatisfacción del cliente en este dataset. | Tabla de correlación por ciudad, tab Operaciones | Pregunta 2 |
| 3 | 1.751 ventas (17.5%) corresponden a SKU sin registro en inventario, representando $1.74M USD (17.5% del ingreso total) sin trazabilidad de costo ni margen. | Gráfica de ventas fantasma por canal, tab Operaciones | Pregunta 3 |
| 4 | Smartphones combina alto stock disponible con el NPS más bajo entre categorías — paradoja stock/sentimiento, con rating de producto también bajo (apunta a calidad, no solo precio). | Cruce Stock/NPS/Rating por categoría, tab Cliente | Pregunta 4 |
| 5 | La bodega Occidente lleva en promedio 356 días sin revisión de stock, con una correlación de 0.36 entre antigüedad de revisión y tasa de tickets de soporte — es la más urgente de auditar. | Tabla comparativa por bodega, tab Cliente | Pregunta 5 |
| 6 | La relación más fuerte entre variables numéricas de negocio, fuera de las 5 preguntas dirigidas, es Margen_Utilidad↔Precio_Venta_Final (0.80) — esperable, sin relaciones ocultas relevantes adicionales. | Matriz de correlación, tab Análisis Final | Exploración multivariable |

## 7. Problemas de calidad de datos encontrados (resumen GIGO)

| Problema | Estrategia de corrección | Justificación |
| --- | --- | --- |
| `Lead_Time_Dias` mezclaba números, texto (`"25-30 días"`) y `"Inmediato"` | `"Inmediato"` → 0 días; rangos → promedio del rango; nulos genuinos restantes → mediana por Categoria | `"Inmediato"` es un dato real (reposición sin espera), tratarlo como nulo habría borrado 433 registros válidos |
| `Costo_Unitario_USD` con rango extremo ($0.05–$850.000) | Winsorizing por rango intercuartílico (IQR) | Acota el valor sin perder el resto del registro del SKU |
| `Categoria` con 8 valores para 5 categorías reales, incluyendo `"???"` (305 registros) | Normalización de variantes de mayúscula/guion + `"???"` → NaN explícito → imputación con moda por Bodega_Origen | `"???"` es un nulo técnico disfrazado de texto; `isna()` no lo detecta si no se convierte explícitamente |
| `Ciudad_Destino` contaminada con `"Ventas_Web"` (1.290 registros, 12.9%) | Se marca como NaN + columna de flag `Ciudad_Invalida`, sin imputar | Es un valor de `Canal_Venta` filtrado a la columna equivocada; no hay forma confiable de inferir la ciudad real |
| SKU vendidos sin registro en inventario (1.751 ventas, 17.5%) | Columna de flag `SKU_Fantasma`, sin eliminar ni imputar | Es una decisión de negocio (¿producto nuevo o error?), no una limpieza técnica |
| Colisión de `Feedback_ID` (969 filas con mismo ID, contenido distinto) | Regeneración de ID único con sufijo, se conserva el `Feedback_ID_Original`, ninguna fila se elimina | Es una colisión de identificador, no un duplicado real; eliminar habría descartado feedback legítimo de clientes |
| Fan-out en el merge del dataset maestro (10.000 → 10.877 filas) | Se agrega el feedback a nivel de `Transaccion_ID` antes del merge | 877 transacciones tenían más de un registro de feedback asociado; sin agregar antes, esas ventas se contaban varias veces en cualquier suma de ingresos |

*(Tabla completa exportada en `results/tabla_diagnostico_gigo.csv`; el detalle de identificación,
decisión y justificación de cada corrección está en la pestaña Auditoría del dashboard.)*

## 8. Decisión recomendada

- **Corregir precios y descatalogar los SKU con margen negativo estructural** (complejidad baja,
  1-2 semanas) — puede recuperar hasta $1.73M USD/periodo; el problema afecta a los 4 canales de
  venta casi por igual, por lo que la corrección debe aplicarse a nivel de política de precios,
  no a un canal específico.
- **Auditar y sincronizar el catálogo de inventario** (complejidad media, 3-6 semanas) — para
  recuperar visibilidad sobre el 17.5% del ingreso total actualmente sin trazabilidad de costo.
- **Implementar auditorías periódicas de stock en la bodega Occidente** (complejidad alta,
  2-3 meses) — es la de mayor correlación entre antigüedad de revisión y tickets de soporte.
- Dado que la evidencia no respalda la hipótesis logística de forma clara, **no se recomienda
  priorizar un cambio de operador logístico** sin antes investigar otras causas de insatisfacción
  (ver Pregunta 4).

El dashboard (pestaña **Análisis Final**) detalla las 5 recomendaciones completas — una por
pregunta, con objetivo, responsable sugerido, plazo, impacto esperado y pasos concretos. El
Documento de Hallazgos oficial prioriza y desarrolla en su sección 4 las 3 de menor complejidad
de implementación, como plan de acción inmediato para la junta directiva.

## 9. Declaración de uso de Inteligencia Artificial

Ver `docs/declaracion_uso_IA.md`. Resumen: se usó IA generativa (Claude, Anthropic) como apoyo en
la construcción del pipeline de datos, el dashboard en Streamlit y el Documento de Hallazgos,
verificando siempre el código generado contra los datos reales del proyecto — ese proceso de
verificación sacó a la luz varios errores no evidentes en el código a simple vista (detallados en
la declaración). La elección de estrategias de limpieza/imputación, la interpretación de los
hallazgos y las decisiones de negocio finales fueron discutidas, definidas y validadas por el
equipo.