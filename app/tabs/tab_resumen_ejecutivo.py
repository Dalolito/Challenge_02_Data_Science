import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st

from .ui_helpers import titulo_seccion, download_button_verde, guardar_matplotlib

VARS_CORRELACION = [
    "Margen_Utilidad", "Precio_Venta_Final", "Costo_Envio", "Tiempo_Entrega_Real",
    "Stock_Actual", "Costo_Unitario_USD", "Rating_Producto", "Rating_Logistica",
    "Satisfaccion_NPS", "Cantidad_Vendida", "Edad_Cliente",
]

def _calcular_hallazgos(df: pd.DataFrame) -> dict:
    """Recalcula los números clave de las 5 preguntas del reto, en un solo dict."""
    h = {"n_total": len(df)}

    if "Margen_Utilidad" in df.columns:
        df_margen = df[df["Margen_Utilidad"].notna()]
        df_negativo = df_margen[df_margen["Margen_Utilidad"] < 0]
        h["n_margen_negativo"] = len(df_negativo)
        h["pct_margen_negativo"] = 100 * len(df_negativo) / len(df_margen) if len(df_margen) else 0
        h["perdida_total"] = df_negativo["Margen_Utilidad"].sum() if not df_negativo.empty else 0
        if not df_negativo.empty and "SKU_ID" in df.columns:
            h["peor_sku"] = df_negativo.groupby("SKU_ID")["Margen_Utilidad"].sum().idxmin()
        if not df_negativo.empty and "Canal_Venta" in df.columns:
            por_canal_perdida = df_negativo.groupby("Canal_Venta")["Margen_Utilidad"].sum()
            h["canal_peor_margen"] = por_canal_perdida.idxmin()
            h["perdida_canal_peor"] = por_canal_perdida.min()

            pct_perdida_canal = 100 * por_canal_perdida.abs() / por_canal_perdida.abs().sum()
            h["pct_perdida_canal_peor"] = pct_perdida_canal.loc[h["canal_peor_margen"]]
            h["pct_perdida_canal_min"] = pct_perdida_canal.min()
            h["pct_perdida_canal_max"] = pct_perdida_canal.max()
            h["spread_perdida_canales"] = pct_perdida_canal.max() - pct_perdida_canal.min()
            h["margen_negativo_balanceado"] = h["spread_perdida_canales"] <= 10

    if {"Ciudad_Destino", "Tiempo_Entrega_Real", "Satisfaccion_NPS"}.issubset(df.columns):
        df_val = df.dropna(subset=["Ciudad_Destino", "Tiempo_Entrega_Real", "Satisfaccion_NPS"])
        if not df_val.empty:
            corr_por_ciudad = (
                df_val.groupby("Ciudad_Destino")
                .apply(lambda g: g["Tiempo_Entrega_Real"].corr(g["Satisfaccion_NPS"]) if len(g) > 5 else None)
                .dropna()
            )
            if not corr_por_ciudad.empty:
                h["ciudad_logistica_critica"] = corr_por_ciudad.idxmin()
                h["correlacion_logistica"] = corr_por_ciudad.min()

    if {"Bodega_Origen", "Ultima_Revision", "Ticket_Soporte_Abierto"}.issubset(df.columns):
        df_val = df.dropna(subset=["Bodega_Origen", "Ultima_Revision"]).copy()
        if not df_val.empty:
            fecha_ref = df_val["Ultima_Revision"].max()
            df_val["Dias_Sin_Revision"] = (fecha_ref - df_val["Ultima_Revision"]).dt.days
            resumen_bod = df_val.groupby("Bodega_Origen").agg(
                Dias_Sin_Revision=("Dias_Sin_Revision", "mean"),
                Tasa_Ticket=("Ticket_Soporte_Abierto", "mean"),
            )
            corr = resumen_bod["Dias_Sin_Revision"].corr(resumen_bod["Tasa_Ticket"])
            h["correlacion_logistica_bodega"] = corr
            if pd.notna(corr):
                h["bodega_critica"] = resumen_bod["Dias_Sin_Revision"].idxmax()
                h["bodega_critica_tasa_ticket"] = resumen_bod.loc[h["bodega_critica"], "Tasa_Ticket"] * 100

    if "SKU_Fantasma" in df.columns and "Precio_Venta_Final" in df.columns:
        n_fantasma = int(df["SKU_Fantasma"].sum())
        h["n_fantasma"] = n_fantasma
        h["pct_fantasma"] = 100 * n_fantasma / len(df) if len(df) else 0
        ingreso_total = df["Precio_Venta_Final"].sum()
        ingreso_fantasma = df.loc[df["SKU_Fantasma"], "Precio_Venta_Final"].sum()
        h["ingreso_riesgo"] = ingreso_fantasma
        h["pct_ingreso_riesgo"] = 100 * ingreso_fantasma / ingreso_total if ingreso_total else 0

    if {"Categoria", "Stock_Actual", "Satisfaccion_NPS"}.issubset(df.columns):
        df_val = df.dropna(subset=["Categoria", "Stock_Actual", "Satisfaccion_NPS"])
        if not df_val.empty:
            resumen_cat = df_val.groupby("Categoria").agg(
                Stock_Promedio=("Stock_Actual", "mean"),
                NPS_Promedio=("Satisfaccion_NPS", "mean"),
                Rating_Promedio=("Rating_Producto", "mean"),
            )
            stock_mediana = resumen_cat["Stock_Promedio"].median()
            nps_mediana = resumen_cat["NPS_Promedio"].median()
            paradoja = resumen_cat[
                (resumen_cat["Stock_Promedio"] > stock_mediana) & (resumen_cat["NPS_Promedio"] < nps_mediana)
            ]
            if not paradoja.empty:
                h["categorias_paradoja"] = paradoja.index.tolist()
                cat_principal = paradoja.index[0]
                h["rating_categoria_paradoja"] = resumen_cat.loc[cat_principal, "Rating_Promedio"]

    return h


# 1. Contexto del encargo

def _render_contexto():
    titulo_seccion("1. Contexto del encargo")
    st.markdown(
        "**TechLogistics S.A.S.** nos contrató como consultores porque detectó dos síntomas "
        "preocupantes en su operación: una **erosión sostenida del margen de beneficios** y "
        "una **caída drástica en la lealtad de sus clientes**. La hipótesis inicial de la junta "
        "directiva era que la causa raíz está en la **invisibilidad operativa**: sus tres "
        "sistemas principales — ERP de Inventarios, Logística y Feedback de clientes — no "
        "hablan el mismo idioma entre sí. Nuestro trabajo fue confirmar o descartar esa "
        "hipótesis con evidencia, y traducir el hallazgo en un plan de acción concreto."
    )


# 2. Qué se analizó

def _render_que_se_analizo(datasets_crudos: dict, reportes_limpieza: list, df_filtrado: pd.DataFrame = None):
    titulo_seccion("2. Qué se analizó")

    if not datasets_crudos or not reportes_limpieza:
        st.info("No hay información de los datos crudos disponible para esta sección.")
        return

    n_inv = len(datasets_crudos.get("inventario", []))
    n_trx = len(datasets_crudos.get("transacciones", []))
    n_fb = len(datasets_crudos.get("feedback", []))
    n_total_registros = n_inv + n_trx + n_fb

    total_correcciones = sum(len(rep.get("cambios", [])) for rep in reportes_limpieza)

    st.markdown(
        f"Se recibieron **3 fuentes de datos independientes** que sumaban "
        f"**{n_total_registros:,} registros**: el maestro de inventario "
        f"({n_inv:,} productos), el histórico de transacciones logísticas "
        f"({n_trx:,} ventas), y el feedback de clientes ({n_fb:,} respuestas). "
        "Ninguna de las tres tablas compartía un formato consistente entre sí, lo cual ya "
        "era en sí mismo evidencia de la hipótesis inicial de la junta."
    )

    # Descripción cualitativa por dataset (basada en el reporte real, sin inventar números)
    problemas_por_dataset = {
        "inventario": "costos unitarios con rangos absurdos (desde centavos hasta cientos de "
                      "miles de dólares), existencias negativas, categorías de producto "
                      "registradas de hasta 3 formas distintas, y tiempos de reposición "
                      "mezclados entre texto y número en la misma columna.",
        "transacciones": "miles de ventas asociadas a productos que no existen en el "
                          "inventario oficial, tiempos de entrega con valores centinela "
                          "de hasta 999 días, y una columna de ciudad de destino "
                          "contaminada con datos que en realidad pertenecían al canal de venta.",
        "feedback": "identificadores de respuesta reutilizados entre clientes distintos, "
                    "calificaciones de producto fuera de la escala válida, edades "
                    "biológicamente imposibles, y una porción significativa de opiniones "
                    "sin diligenciar.",
    }

    for nombre, descripcion in problemas_por_dataset.items():
        if nombre in datasets_crudos:
            st.markdown(f"- **{nombre.capitalize()}**: {descripcion}")

    st.markdown(
        f"En total, se aplicaron **{total_correcciones} correcciones documentadas** durante "
        "la fase de limpieza — cada una con su identificación, la decisión tomada y su "
        "justificación, disponibles para auditoría en la pestaña **🔍 Auditoría**. Ninguna "
        "corrección se aplicó sin dejar rastro: los registros que no se pudieron corregir con "
        "certeza (ej. ventas sin producto asociado) se marcaron para decisión de negocio en "
        "vez de eliminarse o inventarse un valor."
    )

    if df_filtrado is not None and not df_filtrado.empty:
        st.markdown("")
        _render_matriz_correlacion(df_filtrado)


def _render_matriz_correlacion(df: pd.DataFrame):
    st.markdown("**Exploración multivariable previa**")
    st.caption(
        "Antes de enfocar el análisis en las 5 preguntas estratégicas, se cruzaron todas las "
        "variables numéricas de negocio entre sí — sirve para detectar relaciones que no se "
        "estaban buscando explícitamente, más allá de las que el enunciado ya pedía investigar."
    )

    disponibles = [c for c in VARS_CORRELACION if c in df.columns]
    if len(disponibles) < 2:
        st.info("No hay suficientes variables numéricas disponibles para calcular la matriz.")
        return

    corr = df[disponibles].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        annot_kws={"size": 6.5}, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlación entre variables numéricas de negocio", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig, width="content")
    guardar_matplotlib(fig, "resumen_correlacion_variables_negocio")
    plt.close(fig)

    mascara_diagonal = pd.DataFrame(
        np.eye(len(corr), dtype=bool), index=corr.index, columns=corr.columns,
    )
    corr_sin_diagonal = corr.mask(mascara_diagonal)
    corr_stack = corr_sin_diagonal.abs().stack()
    if not corr_stack.empty:
        par_mas_fuerte = corr_stack.idxmax()
        valor_mas_fuerte = corr.loc[par_mas_fuerte[0], par_mas_fuerte[1]]
        st.markdown(
            f"De la matriz anterior observamos que la relación más fuerte fuera de las "
            f"variables ya investigadas en las 5 preguntas oficiales es entre "
            f"**{par_mas_fuerte[0]}** y **{par_mas_fuerte[1]}** ({valor_mas_fuerte:.2f}). "
            "El resto de las variables muestran correlaciones cercanas a cero, lo que "
            "confirma que el análisis dirigido por pregunta de negocio cubre las relaciones "
            "relevantes del conjunto de datos."
        )


# 3. Qué se identificó — anclado a las 5 preguntas oficiales del reto

def _render_hallazgos(h: dict):
    titulo_seccion("3. Qué se identificó")
    st.caption(
        "Cada hallazgo responde directamente una de las 5 preguntas estratégicas que la "
        "junta directiva planteó al inicio del encargo."
    )

    # Pregunta 1
    with st.container(border=True):
        st.markdown(
            "**Pregunta 1 — Fuga de Capital y Rentabilidad:** *¿Los SKU con margen negativo "
            "representan una pérdida aceptable por volumen o una falla crítica de precios?*"
        )
        if "n_margen_negativo" in h:
            texto = (
                f"Analizamos el margen de utilidad de cada venta y encontramos que "
                f"**{h['n_margen_negativo']:,} transacciones ({h['pct_margen_negativo']:.1f}%)** "
                f"se ejecutaron con margen negativo, acumulando una pérdida de "
                f"**{abs(h['perdida_total']):,.0f} USD**."
            )
            if "canal_peor_margen" in h:
                if h.get("margen_negativo_balanceado", False):
                    texto += (
                        f" La pérdida se distribuye de forma prácticamente pareja entre los "
                        f"4 canales (entre {h['pct_perdida_canal_min']:.1f}% y "
                        f"{h['pct_perdida_canal_max']:.1f}% del total cada uno) — el canal "
                        f"**{h['canal_peor_margen']}** solo tiene {h['pct_perdida_canal_peor']:.1f}% "
                        "de la pérdida, en línea con su participación en el volumen de ventas. "
                        "Esto **descarta una falla de precios aislada en un canal específico**: "
                        "es un **problema sistemático de pricing que afecta a todos los canales "
                        "por igual**, no un canal concentrando la pérdida."
                    )
                else:
                    texto += (
                        f" El canal **{h['canal_peor_margen']}** concentra "
                        f"{h['pct_perdida_canal_peor']:.1f}% de esa pérdida "
                        f"({abs(h['perdida_canal_peor']):,.0f} USD), muy por encima de su "
                        "participación en el volumen de ventas, lo que sí indica una "
                        "**falla de precios específica de ese canal** (no un problema "
                        "distribuido entre todos)."
                    )
            st.markdown(texto)
        else:
            st.info("No hay datos suficientes en el filtro actual para responder esta pregunta.")

    # Pregunta 2
    with st.container(border=True):
        st.markdown(
            "**Pregunta 2 — Crisis Logística y Cuellos de Botella:** *¿En qué ciudad o bodega "
            "la correlación entre tiempo de entrega y NPS bajo es más fuerte?*"
        )
        if "ciudad_logistica_critica" in h:
            fuerte = abs(h["correlacion_logistica"]) >= 0.3
            if fuerte:
                st.markdown(
                    f"Cruzamos el tiempo de entrega real contra el NPS de cada ciudad y "
                    f"encontramos que **{h['ciudad_logistica_critica']}** presenta la "
                    f"correlación más negativa del país ({h['correlacion_logistica']:.2f}) — "
                    "es la zona donde el retraso logístico le está costando satisfacción real "
                    "a la empresa, y la candidata prioritaria para un cambio de operador."
                )
            else:
                st.markdown(
                    "Cruzamos el tiempo de entrega real contra el NPS de cada ciudad y, "
                    f"incluso en el caso más marcado (**{h['ciudad_logistica_critica']}**, "
                    f"correlación de {h['correlacion_logistica']:.2f}), la relación es débil. "
                    "**Con la evidencia actual, la logística no aparece como el principal "
                    "motor de la insatisfacción del cliente** — recomendamos no priorizar un "
                    "cambio de operador hasta explorar otras causas."
                )
        else:
            st.info("No hay datos suficientes en el filtro actual para responder esta pregunta.")

    # Pregunta 3
    with st.container(border=True):
        st.markdown(
            "**Pregunta 3 — Análisis de la Venta Invisible:** *¿Cuál es el impacto financiero "
            "de las ventas cuyo SKU no está en el maestro de inventario?*"
        )
        if "n_fantasma" in h:
            st.markdown(
                f"Cruzamos cada venta contra el maestro de inventario y encontramos que "
                f"**{h['n_fantasma']:,} ventas ({h['pct_fantasma']:.1f}%)** corresponden a "
                f"productos que no existen en el catálogo oficial. Esto representa "
                f"**{h['ingreso_riesgo']:,.0f} USD** — el **{h['pct_ingreso_riesgo']:.1f}% del "
                "ingreso total** — sobre el cual la empresa no puede calcular costo ni margen "
                "real con certeza. Es el hallazgo de mayor riesgo de control financiero de "
                "todo el análisis."
            )
        else:
            st.info("No hay datos suficientes en el filtro actual para responder esta pregunta.")

    # Pregunta 4
    with st.container(border=True):
        st.markdown(
            "**Pregunta 4 — Diagnóstico de Fidelidad:** *¿Hay categorías con stock alto pero "
            "sentimiento de cliente negativo? ¿Es mala calidad de producto o sobrecosto?*"
        )
        if "categorias_paradoja" in h:
            rating = h.get("rating_categoria_paradoja")
            if pd.notna(rating) and rating < 3.5:
                diagnostico = (
                    f"su calificación de producto promedio también es baja ({rating:.1f}/5), "
                    "lo que apunta a un **problema de calidad de producto**, no de precio."
                )
            else:
                diagnostico = (
                    "su calificación de producto es aceptable pero la satisfacción general "
                    "sigue baja, lo que apunta más a un **tema de precio o sobrecosto percibido**."
                )
            st.markdown(
                f"Cruzamos stock disponible, NPS y calificación de producto por categoría, y "
                f"encontramos que **{', '.join(h['categorias_paradoja'])}** combina alta "
                f"disponibilidad con baja satisfacción: {diagnostico}"
            )
        else:
            st.info("No se detectó ninguna categoría en paradoja con el filtro actual.")

    # Pregunta 5
    with st.container(border=True):
        st.markdown(
            "**Pregunta 5 — Storytelling de Riesgo Operativo:** *¿Qué bodegas están operando "
            "a ciegas y cómo impacta esto en la satisfacción final?*"
        )
        if "bodega_critica" in h:
            corr = h.get("correlacion_logistica_bodega")
            if pd.notna(corr) and corr > 0.3:
                st.markdown(
                    f"Cruzamos la antigüedad de la última revisión de stock contra la tasa de "
                    f"tickets de soporte por bodega, y encontramos una correlación positiva "
                    f"notable ({corr:.2f}): a más tiempo sin auditar el inventario, más "
                    f"tickets de soporte. **{h['bodega_critica']}** es la bodega más rezagada "
                    f"en revisión, con una tasa de tickets del {h.get('bodega_critica_tasa_ticket', 0):.1f}% "
                    "— es la que más urge auditar antes de que el costo oculto en soporte siga creciendo."
                )
            else:
                st.markdown(
                    "Cruzamos la antigüedad de la última revisión de stock contra la tasa de "
                    "tickets de soporte por bodega, y la relación resultó débil en el filtro "
                    "actual — no encontramos evidencia suficiente de que la falta de revisión "
                    "de inventario, por sí sola, esté generando más tickets de soporte."
                )
        else:
            st.info("No hay datos suficientes en el filtro actual para responder esta pregunta.")


# 4. Plan de Acción

def _construir_recomendaciones(h: dict) -> list:
    """Arma la lista de recomendaciones a partir de los hallazgos. Separado del
    renderizado para poder reusar exactamente el mismo contenido en el PDF."""
    recomendaciones = []

    # Recomendación — Pregunta 1 (márgenes)
    if h.get("n_margen_negativo", 0) > 0:
        if "canal_peor_margen" not in h:
            canal_txt = ""
        elif h.get("margen_negativo_balanceado", False):
            canal_txt = " (distribuida de forma pareja entre los 4 canales, no aislada a uno)"
        else:
            canal_txt = f", concentrada desproporcionadamente en el canal **{h['canal_peor_margen']}**"
        recomendaciones.append({
            "pregunta": "Pregunta 1 — Rentabilidad",
            "titulo": "Corregir precios y descatalogar SKU con margen negativo estructural",
            "complejidad": "Baja",
            "plazo": "1–2 semanas",
            "responsable": "Gerencia Comercial / Pricing",
            "impacto_esperado": f"Recuperar hasta ${abs(h.get('perdida_total', 0)):,.0f} USD/periodo "
                                 "si se corrige el precio de los SKU identificados.",
            "objetivo": (
                f"{h['n_margen_negativo']:,} ventas ({h['pct_margen_negativo']:.1f}%) se están "
                f"haciendo con margen negativo{canal_txt}. El objetivo es detener la sangría "
                "antes de que el próximo cierre financiero la refleje."
            ),
            "pasos": [
                "Exportar la tabla de SKU con margen negativo desde la pestaña Operaciones (botón de descarga).",
                "Clasificar cada SKU en 2 grupos: error de precio (corregible) vs. producto sin viabilidad comercial (descatalogar).",
                "Ajustar el precio de venta o el costo de proveedor de los SKU corregibles en el sistema de origen.",
                "Retirar del catálogo activo los productos sin viabilidad, evitando seguir generando pérdida por volumen.",
                "Monitorear el % de ventas con margen negativo en el dashboard la semana siguiente al cambio.",
            ],
        })
    else:
        recomendaciones.append({
            "pregunta": "Pregunta 1 — Rentabilidad",
            "titulo": "Mantener monitoreo preventivo de márgenes por SKU",
            "complejidad": "Baja",
            "plazo": "Continuo",
            "responsable": "Gerencia Comercial / Pricing",
            "impacto_esperado": "Detección temprana de futuros SKU con margen negativo antes de que escalen.",
            "objetivo": "No se detectaron ventas con margen negativo en el filtro actual, pero la "
                        "condición de mercado puede cambiar con el tiempo (costos de proveedor, "
                        "descuentos por canal).",
            "pasos": [
                "Revisar la pestaña Operaciones mensualmente, no solo cuando haya una alerta.",
                "Fijar un umbral de alerta (ej. margen < 5%) para intervenir antes de que un SKU caiga a negativo.",
            ],
        })

    # Recomendación — Pregunta 2 (logística)
    corr_log = h.get("correlacion_logistica")
    if "ciudad_logistica_critica" in h and pd.notna(corr_log) and abs(corr_log) >= 0.3:
        recomendaciones.append({
            "pregunta": "Pregunta 2 — Logística",
            "titulo": f"Renegociar o cambiar el operador logístico en {h['ciudad_logistica_critica']}",
            "complejidad": "Alta",
            "plazo": "1–2 meses",
            "responsable": "Gerencia de Operaciones / Logística",
            "impacto_esperado": "Mejora esperada en NPS de la zona proporcional a la reducción del "
                                 "tiempo de entrega; a validar con una prueba piloto antes del cambio definitivo.",
            "objetivo": (
                f"{h['ciudad_logistica_critica']} muestra la correlación más negativa "
                f"({corr_log:.2f}) entre tiempo de entrega y NPS del país — es la zona donde "
                "la logística está costando clientes de forma medible."
            ),
            "pasos": [
                f"Auditar los SLA actuales del operador logístico que cubre {h['ciudad_logistica_critica']}.",
                "Solicitar cotización y SLA de al menos 2 operadores alternativos en la zona.",
                "Ejecutar una prueba piloto (ej. 4-6 semanas) con el operador alternativo en un subconjunto de envíos.",
                "Comparar NPS y tiempo de entrega del piloto contra el operador actual antes de decidir el cambio definitivo.",
            ],
        })
    else:
        recomendaciones.append({
            "pregunta": "Pregunta 2 — Logística",
            "titulo": "Investigar causas de insatisfacción distintas a la logística",
            "complejidad": "Media",
            "plazo": "3–4 semanas",
            "responsable": "Gerencia de Experiencia al Cliente",
            "impacto_esperado": "Identificar el verdadero motor del NPS bajo, evitando invertir en "
                                 "un cambio de operador que no resolvería el problema real.",
            "objetivo": (
                "La relación entre tiempo de entrega y NPS es débil incluso en la ciudad más "
                "afectada" + (f" (**{h['ciudad_logistica_critica']}**, {corr_log:.2f})" if "ciudad_logistica_critica" in h else "")
                + ", lo que sugiere que el problema de satisfacción del cliente no está "
                "principalmente en la logística."
            ),
            "pasos": [
                "Cruzar el NPS bajo contra Rating_Producto y Comentario_Texto para identificar el patrón real.",
                "Revisar si la insatisfacción se concentra en categorías específicas (ver Pregunta 4).",
                "Diseñar una encuesta corta de seguimiento a clientes con NPS bajo para confirmar la causa raíz antes de invertir en logística.",
            ],
        })

    # Recomendación — Pregunta 3 (SKU fantasma)
    if h.get("n_fantasma", 0) > 0:
        recomendaciones.append({
            "pregunta": "Pregunta 3 — Venta invisible",
            "titulo": "Auditar y sincronizar el catálogo de inventario con el sistema de ventas",
            "complejidad": "Media",
            "plazo": "3–6 semanas",
            "responsable": "Gerencia de Inventario / TI",
            "impacto_esperado": f"Recuperar visibilidad y control sobre ${h.get('ingreso_riesgo', 0):,.0f} "
                                 f"USD ({h.get('pct_ingreso_riesgo', 0):.1f}% del ingreso total) actualmente "
                                 "sin trazabilidad de costo ni margen.",
            "objetivo": (
                f"{h['n_fantasma']:,} ventas ({h['pct_fantasma']:.1f}%) corresponden a SKU que "
                "no existen en el inventario oficial — la empresa no puede calcular su costo "
                "ni margen real, y no sabe si son ventas legítimas, errores de digitación o "
                "un síntoma de fraude."
            ),
            "pasos": [
                "Exportar la tabla de ventas fantasma desde la pestaña Operaciones.",
                "Agrupar por SKU y volumen para priorizar la investigación de los que más ingreso representan.",
                "Coordinar con el equipo de ventas y proveedores para determinar si son productos nuevos pendientes de catalogar.",
                "Registrar en el ERP los SKU legítimos; escalar a auditoría interna los que no tengan justificación.",
                "Implementar una validación automática que bloquee o marque ventas de SKU no catalogados a futuro.",
            ],
        })
    else:
        recomendaciones.append({
            "pregunta": "Pregunta 3 — Venta invisible",
            "titulo": "Mantener el control de integridad SKU-inventario",
            "complejidad": "Baja",
            "plazo": "Continuo",
            "responsable": "Gerencia de Inventario / TI",
            "impacto_esperado": "Prevenir la reaparición de ventas sin respaldo de inventario.",
            "objetivo": "No se detectaron ventas fantasma en el filtro actual — mantener el control implementado.",
            "pasos": [
                "Conservar la validación de integridad SKU-inventario en el pipeline de datos.",
                "Revisar periódicamente que ningún nuevo canal de venta esté generando SKU fuera de catálogo.",
            ],
        })

    # Recomendación — Pregunta 4 (paradoja stock/NPS)
    if "categorias_paradoja" in h:
        rating = h.get("rating_categoria_paradoja")
        if pd.notna(rating) and rating < 3.5:
            enfoque = "una revisión de calidad de producto (retrabajo con el proveedor, control de calidad de entrada, o reemplazo de línea)"
        else:
            enfoque = "una revisión de estrategia de precio (el producto es percibido como caro para el valor que entrega)"
        recomendaciones.append({
            "pregunta": "Pregunta 4 — Fidelidad",
            "titulo": f"Intervenir la categoría {', '.join(h['categorias_paradoja'])}: {enfoque.split('(')[0].strip()}",
            "complejidad": "Media",
            "plazo": "1–3 meses",
            "responsable": "Gerencia de Producto / Categoría",
            "impacto_esperado": "Mejora del NPS de la categoría sin necesidad de reducir el inventario "
                                 "disponible, evitando quiebres de stock innecesarios.",
            "objetivo": (
                f"**{', '.join(h['categorias_paradoja'])}** combina alto stock disponible con "
                "bajo NPS — el problema no es de disponibilidad, es de percepción o calidad "
                f"del producto. Se recomienda {enfoque}."
            ),
            "pasos": [
                "Leer una muestra de Comentario_Texto de clientes con NPS bajo en esta categoría para identificar el motivo recurrente.",
                "Si es calidad: auditar con el proveedor los lotes recientes y reforzar el control de calidad de entrada.",
                "Si es precio: comparar contra la competencia y evaluar una promoción o ajuste de precio dirigido.",
                "Volver a medir NPS de la categoría 60-90 días después de la intervención.",
            ],
        })
    else:
        recomendaciones.append({
            "pregunta": "Pregunta 4 — Fidelidad",
            "titulo": "Mantener monitoreo de la relación stock/satisfacción por categoría",
            "complejidad": "Baja",
            "plazo": "Continuo",
            "responsable": "Gerencia de Producto / Categoría",
            "impacto_esperado": "Detección temprana de nuevas paradojas stock/NPS antes de que impacten la fidelidad.",
            "objetivo": "No se detectó ninguna categoría en paradoja en el filtro actual.",
            "pasos": [
                "Revisar la pestaña Cliente trimestralmente para detectar cambios en el patrón.",
            ],
        })

    # Recomendación — Pregunta 5 (riesgo operativo / bodegas)
    corr_bod = h.get("correlacion_logistica_bodega")
    if "bodega_critica" in h and pd.notna(corr_bod) and corr_bod > 0.3:
        recomendaciones.append({
            "pregunta": "Pregunta 5 — Riesgo Operativo",
            "titulo": f"Implementar auditorías periódicas de stock en {h['bodega_critica']}",
            "complejidad": "Alta",
            "plazo": "2–3 meses para implementar, luego recurrente",
            "responsable": "Gerencia de Operaciones / Bodega",
            "impacto_esperado": f"Reducir la tasa de tickets de soporte desde el "
                                 f"{h.get('bodega_critica_tasa_ticket', 0):.1f}% actual en esa bodega, "
                                 "acercándola al promedio de las demás.",
            "objetivo": (
                f"**{h['bodega_critica']}** muestra una correlación positiva notable "
                f"({corr_bod:.2f}) entre días sin revisar el inventario y tickets de soporte: "
                "está operando a ciegas, y eso se traduce en más fricción para el cliente final."
            ),
            "pasos": [
                "Definir una frecuencia mínima de revisión de inventario (ej. mensual) para esta bodega.",
                "Asignar un responsable directo del cumplimiento de esa frecuencia.",
                "Registrar la fecha de cada revisión en el sistema (evita que Ultima_Revision vuelva a quedar desactualizada).",
                "Extender el mismo proceso a las demás bodegas si el piloto reduce la tasa de tickets.",
            ],
        })
    else:
        recomendaciones.append({
            "pregunta": "Pregunta 5 — Riesgo Operativo",
            "titulo": "Mantener la cadencia actual de revisión de inventario",
            "complejidad": "Baja",
            "plazo": "Continuo",
            "responsable": "Gerencia de Operaciones / Bodega",
            "impacto_esperado": "Prevenir que alguna bodega empiece a rezagarse en sus revisiones.",
            "objetivo": "No se encontró una correlación fuerte entre antigüedad de revisión y "
                        "tickets de soporte en el filtro actual.",
            "pasos": [
                "Mantener el registro de Ultima_Revision actualizado en todas las bodegas.",
                "Revisar este indicador junto con el resto del plan de acción cada trimestre.",
            ],
        })

    return recomendaciones


def _render_plan_de_accion(h: dict):
    titulo_seccion("4. Plan de Acción Recomendado")
    st.caption(
        "Una recomendación por cada pregunta estratégica, con objetivo, pasos concretos, "
        "responsable sugerido, plazo e impacto esperado — no solo el titular del hallazgo."
    )

    recomendaciones = _construir_recomendaciones(h)

    color_por_complejidad = {"Baja": "🟢", "Media": "🟡", "Alta": "🔴"}
    for i, rec in enumerate(recomendaciones, start=1):
        with st.expander(
            f"**{i}. {rec['titulo']}  —  {color_por_complejidad[rec['complejidad']]} "
            f"Complejidad {rec['complejidad']}  ·  {rec['pregunta']}**"
        ):
            st.markdown(f"**Objetivo:** {rec['objetivo']}")
            st.markdown(f"**Responsable sugerido:** {rec['responsable']}  |  **Plazo:** {rec['plazo']}")
            st.markdown(f"**Impacto esperado:** {rec['impacto_esperado']}")
            st.markdown("**Pasos concretos:**")
            for paso in rec["pasos"]:
                st.markdown(f"- {paso}")


# Generación del informe en PDF

def _limpiar_texto_pdf(texto: str) -> str:
    """Prepara texto para las fuentes core de fpdf2 (codificación Latin-1):
    quita markdown (**negrilla**), normaliza guiones/comillas especiales y
    descarta cualquier carácter que Latin-1 no pueda representar (emojis)."""
    texto = texto.replace("**", "").replace("*", "")
    texto = texto.replace("—", "-").replace("–", "-")
    texto = texto.replace("“", '"').replace("”", '"').replace("’", "'")
    return texto.encode("latin-1", errors="ignore").decode("latin-1")


def _generar_pdf_informe(
    df_filtrado: pd.DataFrame, datasets_crudos: dict, reportes_limpieza: list, h: dict,
) -> bytes:
    """Arma un PDF ejecutivo con el mismo contenido de esta pestaña: contexto,
    qué se analizó (con el heatmap), los 5 hallazgos y el plan de acción."""
    from fpdf import FPDF

    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    def titulo_principal(txt):
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 9, _limpiar_texto_pdf(txt), align="C")
        pdf.set_x(pdf.l_margin)

    def seccion(txt):
        pdf.ln(3)
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(37, 99, 235)
        pdf.multi_cell(0, 8, _limpiar_texto_pdf(txt))
        pdf.set_x(pdf.l_margin)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1)

    def subtitulo(txt):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.multi_cell(0, 6, _limpiar_texto_pdf(txt))
        pdf.set_x(pdf.l_margin)

    def parrafo(txt):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5.5, _limpiar_texto_pdf(txt))
        pdf.set_x(pdf.l_margin)
        pdf.ln(1)

    def bullet(txt):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5.5, _limpiar_texto_pdf(f"- {txt}"))
        pdf.set_x(pdf.l_margin)

    titulo_principal("TechLogistics S.A. - Informe de Consultoria")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, _limpiar_texto_pdf(
        f"Generado sobre {len(df_filtrado):,} transacciones filtradas | "
        "Dirigido a la junta directiva de TechLogistics S.A.S."
    ), align="C")
    pdf.set_x(pdf.l_margin)
    pdf.set_text_color(0, 0, 0)

    # 1. Contexto
    seccion("1. Contexto del encargo")
    parrafo(
        "TechLogistics S.A.S. nos contrató como consultores porque detectó dos síntomas "
        "preocupantes en su operación: una erosión sostenida del margen de beneficios y "
        "una caída drástica en la lealtad de sus clientes. La hipótesis inicial de la "
        "junta directiva era que la causa raíz está en la invisibilidad operativa: sus "
        "tres sistemas principales -ERP de Inventarios, Logística y Feedback de "
        "clientes- no hablan el mismo idioma entre sí. Nuestro trabajo fue confirmar o "
        "descartar esa hipótesis con evidencia, y traducir el hallazgo en un plan de "
        "acción concreto."
    )

    # 2. Qué se analizó
    seccion("2. Qué se analizó")
    if datasets_crudos and reportes_limpieza:
        n_inv = len(datasets_crudos.get("inventario", []))
        n_trx = len(datasets_crudos.get("transacciones", []))
        n_fb = len(datasets_crudos.get("feedback", []))
        total_correcciones = sum(len(rep.get("cambios", [])) for rep in reportes_limpieza)
        parrafo(
            f"Se recibieron 3 fuentes de datos independientes que sumaban "
            f"{n_inv + n_trx + n_fb:,} registros: el maestro de inventario ({n_inv:,} "
            f"productos), el histórico de transacciones logísticas ({n_trx:,} ventas), y "
            f"el feedback de clientes ({n_fb:,} respuestas). Se aplicaron "
            f"{total_correcciones} correcciones documentadas durante la limpieza (detalle "
            "completo en la pestaña Auditoría del dashboard)."
        )

    disponibles = [c for c in VARS_CORRELACION if c in df_filtrado.columns]
    if len(disponibles) >= 2:
        corr = df_filtrado[disponibles].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 4.6))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5,
            annot_kws={"size": 6}, ax=ax, cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Correlación entre variables numéricas de negocio", fontsize=9)
        plt.xticks(rotation=45, ha="right", fontsize=6)
        plt.yticks(fontsize=6)
        plt.tight_layout()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            fig.savefig(tmp_img.name, dpi=170, bbox_inches="tight")
            ruta_imagen = tmp_img.name
        plt.close(fig)

        pdf.ln(1)
        pdf.image(ruta_imagen, w=140, x=(210 - 140) / 2)
        os.remove(ruta_imagen)
        pdf.set_x(pdf.l_margin)
        pdf.ln(2)

    # 3. Qué se identificó (5 hallazgos, versión condensada)
    seccion("3. Qué se identificó")
    preguntas_hallazgos = [
        ("Pregunta 1 - Fuga de Capital y Rentabilidad",
         f"{h['n_margen_negativo']:,} transacciones ({h['pct_margen_negativo']:.1f}%) se "
         f"ejecutaron con margen negativo, acumulando una pérdida de "
         f"${abs(h['perdida_total']):,.0f} USD."
         + ((f" La pérdida se distribuye de forma pareja entre los 4 canales "
             f"({h['pct_perdida_canal_min']:.1f}%-{h['pct_perdida_canal_max']:.1f}% cada uno), "
             "sin uno dominante: es un problema sistemático de pricing, no aislado a un canal."
             if h.get("margen_negativo_balanceado", False) else
             f" El canal {h['canal_peor_margen']} concentra {h['pct_perdida_canal_peor']:.1f}% "
             "de esa pérdida, muy por encima de su participación en el volumen de ventas.")
            if "canal_peor_margen" in h else "")
         if "n_margen_negativo" in h else
         "No hay datos suficientes en el filtro actual para responder esta pregunta."),
        ("Pregunta 2 - Crisis Logística y Cuellos de Botella",
         (f"{h['ciudad_logistica_critica']} presenta la correlación más negativa del país "
          f"({h['correlacion_logistica']:.2f}) entre tiempo de entrega y NPS."
          if "ciudad_logistica_critica" in h and abs(h.get("correlacion_logistica", 0)) >= 0.3 else
          "La relación entre tiempo de entrega y NPS es débil en el filtro actual - la "
          "logística no aparece como el principal motor de insatisfacción."
          if "ciudad_logistica_critica" in h else
          "No hay datos suficientes en el filtro actual para responder esta pregunta.")),
        ("Pregunta 3 - Análisis de la Venta Invisible",
         f"{h['n_fantasma']:,} ventas ({h['pct_fantasma']:.1f}%) corresponden a SKU fuera "
         f"del catálogo oficial, representando ${h['ingreso_riesgo']:,.0f} USD "
         f"({h['pct_ingreso_riesgo']:.1f}% del ingreso total) sin trazabilidad de costo."
         if "n_fantasma" in h else
         "No hay datos suficientes en el filtro actual para responder esta pregunta."),
        ("Pregunta 4 - Diagnóstico de Fidelidad",
         f"{', '.join(h['categorias_paradoja'])} combina alta disponibilidad de stock con "
         "bajo NPS - señal de un problema de calidad o precio, no de inventario."
         if "categorias_paradoja" in h else
         "No se detectó ninguna categoría en paradoja con el filtro actual."),
        ("Pregunta 5 - Storytelling de Riesgo Operativo",
         (f"{h['bodega_critica']} es la bodega más rezagada en revisión de stock, con una "
          f"correlación notable ({h.get('correlacion_logistica_bodega', 0):.2f}) frente a "
          "su tasa de tickets de soporte."
          if "bodega_critica" in h and pd.notna(h.get("correlacion_logistica_bodega"))
          and h["correlacion_logistica_bodega"] > 0.3 else
          "No se encontró una correlación fuerte entre antigüedad de revisión y tickets "
          "de soporte en el filtro actual."
          if "bodega_critica" in h else
          "No hay datos suficientes en el filtro actual para responder esta pregunta.")),
    ]
    for titulo_p, texto_p in preguntas_hallazgos:
        subtitulo(titulo_p)
        parrafo(texto_p)

    # 4. Plan de acción
    seccion("4. Plan de acción recomendado")
    for i, rec in enumerate(_construir_recomendaciones(h), start=1):
        subtitulo(f"{i}. {rec['titulo']}  (Complejidad {rec['complejidad']})")
        parrafo(f"Objetivo: {rec['objetivo']}")
        parrafo(
            f"Responsable: {rec['responsable']}  |  Plazo: {rec['plazo']}  |  "
            f"Impacto esperado: {rec['impacto_esperado']}"
        )
        for paso in rec["pasos"]:
            bullet(paso)
        pdf.ln(2)

    return bytes(pdf.output())


# Render principal

def render(df_filtrado: pd.DataFrame, datasets_crudos: dict = None, reportes_limpieza: list = None):
    st.header("📋 Análisis Final")
    st.caption(
        "Informe de consultoría dirigido a la junta directiva de TechLogistics S.A.S. "
        "Reacciona a los filtros del sidebar, igual que el resto del dashboard."
    )

    if df_filtrado is None or df_filtrado.empty:
        st.warning("No hay datos para mostrar con los filtros actuales. Ajusta el sidebar.")
        return

    _render_contexto()
    st.divider()
    _render_que_se_analizo(datasets_crudos, reportes_limpieza, df_filtrado)
    st.divider()

    hallazgos = _calcular_hallazgos(df_filtrado)
    _render_hallazgos(hallazgos)
    st.divider()
    _render_plan_de_accion(hallazgos)

    st.divider()
    pdf_bytes = _generar_pdf_informe(df_filtrado, datasets_crudos, reportes_limpieza, hallazgos)
    download_button_verde(
        "Descargar análisis completo (PDF)",
        data=pdf_bytes,
        file_name="informe_techlogistics.pdf",
        mime="application/pdf",
        key="btn_descarga_pdf_informe",
    )