# Declaración de uso de Inteligencia Artificial

**Challenge 02 — El Desafío de los Datos Erróneos e Interconectados**
**Equipo:** Samuel Gutiérrez Jaramillo, David Lopera Londoño, Juan Diego Acuña Giraldo
**Herramienta usada:** Claude (Anthropic), como apoyo durante la construcción del pipeline de
datos, el dashboard en Streamlit y el documento de hallazgos.

## Resumen

Usamos IA generativa como apoyo en tres frentes: (1) construir y depurar el pipeline de datos
(carga, limpieza, integración) contrastándolo siempre contra los CSV reales del proyecto, (2)
armar la estructura y el código del dashboard en Streamlit, y (3) generar el Documento de
Hallazgos (PDF) a partir de las cifras ya calculadas por el pipeline. En varios puntos, correr el
código generado contra nuestros datos reales sacó a la luz errores que no eran evidentes leyendo
el código a simple vista (detallados en la sección "Auditoría y corrección de errores" más
abajo) — esa verificación fue nuestra, no algo que la IA hiciera de forma autónoma.

## Contexto que se le dio a la IA

Se compartieron las columnas y una muestra de cada uno de los 3 CSV reales del proyecto
(`inventario_central_v2.csv`, `transacciones_logistica_v2.csv`, `feedback_clientes_v2.csv`),
junto con el diccionario de datos, la guía de validación y el enunciado del Challenge 02, para
que el código y las decisiones de limpieza se ajustaran a los problemas reales de nuestros datos
y no a suposiciones genéricas.

## Fase 1 — Carga y limpieza de datos (`src/data_loader.py`, `src/cleaning.py`)

- "Dame el código para `data_loader.py`, que lea los 3 CSV probando distintos encodings, sin
  transformar el contenido."
- "Ayúdame a diagnosticar por qué `Lead_Time_Dias` mezcla números y texto, y qué patrones
  distintos hay dentro de los valores no numéricos."
- "Dame el código para normalizar `Categoria` a un set fijo de valores, tratando `'???'` como
  nulo técnico explícito, no como categoría."
- "Ayúdame a decidir cómo tratar `Ciudad_Destino='Ventas_Web'` — no es una ciudad real, es un
  valor de `Canal_Venta` filtrado a la columna equivocada."
- "Dame el código para marcar las ventas con SKU que no existe en inventario (`SKU_Fantasma`),
  sin eliminar ni imputar esas filas, ya que la decisión de qué hacer con ellas depende del
  negocio."
- "Reestructura el reporte de `cleaning.py` para que cada corrección tenga 3 campos separados
  (identificación, decisión, justificación) en vez de un solo párrafo, para mostrarlo así en el
  dashboard."

## Fase 2 — Integración y Feature Engineering (`src/feature_engineering.py`)

- "Dame el código para unir los 3 datasets en un solo dataset maestro a nivel de transacción."
- "¿Por qué el dataset maestro después del merge tiene más filas que transacciones originales?"
  — esto llevó a encontrar que varias transacciones tenían más de un registro de feedback
  asociado, y a pedir el ajuste de agregar el feedback antes del merge para no duplicar ventas.
- "Ayúdame a calcular `Margen_Utilidad`, `Brecha_Entrega` y `Ratio_Soporte_Categoria` como las 3
  variables derivadas que pide la Fase 2, documentando los supuestos de negocio que no vienen
  dados en el diccionario (ej. el SLA de referencia para calcular la brecha de entrega)."

## Fase 3 — Inteligencia Artificial con Groq (`src/ai_insights.py`)

- "Dame el código para `ai_insights.py`, que resuma el dataset filtrado en un diccionario
  estructurado (no solo promedios) y arme un prompt pidiéndole a Llama-3 un análisis por cada
  pregunta de negocio, no solo generalidades."
- "Ayúdame a que la API key se lea desde `st.secrets`, nunca escrita directamente en el código."

## Dashboard en Streamlit (`app/app.py` y `app/tabs/`)

- "Ayúdame a estructurar `app.py`: carga y limpieza cacheadas, sidebar con filtros globales, y
  las pestañas del dashboard."
- "Dame el código de la pestaña de Auditoría, mostrando 4 dimensiones de calidad (completitud,
  unicidad, validez, consistencia) en vez de solo el % de nulos, porque medir solo nulos no
  detecta datos presentes pero inválidos (ej. `Rating_Producto=99`)."
- "Ayúdame a redactar los hallazgos de las pestañas de Operaciones y Cliente en formato
  narrativo ('de la gráfica anterior observamos que...'), en vez de cajas de alerta genéricas."
- "Dame el código de una pestaña de Análisis Final que recopile los hallazgos de todas las
  demás pestañas, ancladas explícitamente a las 5 preguntas del enunciado, con un plan de acción
  de varias recomendaciones (objetivo, responsable, plazo, impacto esperado, pasos concretos)
  por cada pregunta."
- "Ayúdame a agregar una matriz de correlación entre las variables numéricas de negocio como
  exploración multivariable general" — inicialmente se ubicó en la pestaña de Auditoría, y se
  reubicó en Análisis Final al notar que no correspondía a esa pestaña (Auditoría es solo
  transparencia de limpieza, no análisis exploratorio de negocio).

## Documento de Hallazgos (PDF)

- "Dame el código para generar las figuras del informe (matplotlib) a partir del mismo pipeline
  de datos que usa el dashboard, no números inventados."
- "Ayúdame a armar el PDF con reportlab: portada, contexto del encargo, qué se analizó,
  hallazgos por cada una de las 5 preguntas citadas textualmente, y plan de acción."
- "Ajusta el documento a un formato más estándar: sin colores de marca, sin pie de página, una
  sola tipografía y menos negrilla en el cuerpo del texto."

## Auditoría y corrección de errores

Varias veces el código generado se probó contra los datos reales y se corrigieron errores que no
eran evidentes solo leyendo el código:

- `Lead_Time_Dias`: la primera versión no reconocía el valor `"Inmediato"` y lo convertía en
  nulo, borrando 433 registros válidos en vez de tratarlos como 0 días.
- `Categoria`: la normalización inicial no fusionaba variantes como `"LAPTOP"`/`"Laptops"` ni
  detectaba `"???"` como nulo técnico (quedaba como texto vacío, invisible para `isna()`).
- `feature_engineering.py`: el primer merge infló el dataset de 10.000 a 10.877 filas por
  transacciones con múltiples registros de feedback asociados — se corrigió agregando el
  feedback a nivel de transacción antes del merge.
- Un compañero de equipo compartió una versión anterior de `cleaning.py` que fue auditada
  encontrando 4 bugs adicionales (`Recomienda_Marca` sin normalizar por mayúsculas,
  `Ciudad_Destino='Ventas_Web'` sin tratar, pérdida de registros de feedback por colisión de ID
  tratada como duplicado) antes de integrarla al proyecto.

Esta verificación —correr el código contra los datos reales y contrastar los resultados— fue
un paso manual del equipo en cada entrega, no una garantía automática de la IA.

## Formato general

- "No expliques código en el Documento de Hallazgos; explica por qué la empresa está perdiendo
  dinero y cómo los datos lo demuestran."
- "Ajusta la estructura del repositorio a nuestra plantilla de entrega (`data/`, `notebooks/`,
  `src/`, `app/`, `results/`, `taller_practico/`, `docs/`)."
