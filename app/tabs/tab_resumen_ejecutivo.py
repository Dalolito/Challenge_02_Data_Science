"""
tab_resumen_ejecutivo.py
--------------------------
Pestaña "Análisis Final": recopila los hallazgos de Operaciones y Cliente
en una narrativa de negocio única, dirigida a la junta directiva — no
explica código, explica por qué la empresa está perdiendo dinero y cómo
los datos lo demuestran. Cierra con un Plan de Acción de 3 recomendaciones
tácticas priorizadas por complejidad (Baja/Media/Alta), tal como pide el
checklist del informe de hallazgos.

Recalcula los mismos números que ya se ven en Operaciones y Cliente (no
importa esas tabs para no acoplar el orden de renderizado), a partir del
mismo df_filtrado — así el resumen siempre está sincronizado con lo que
el usuario ve en el resto del dashboard.
"""

import pandas as pd
import streamlit as st


def _calcular_hallazgos(df: pd.DataFrame) -> dict:
    """Recalcula los números clave de las 5 preguntas del reto, en un solo dict."""
    h = {"n_total": len(df)}

    # 1. Márgenes negativos
    if "Margen_Utilidad" in df.columns:
        df_margen = df[df["Margen_Utilidad"].notna()]
        df_negativo = df_margen[df_margen["Margen_Utilidad"] < 0]
        h["n_margen_negativo"] = len(df_negativo)
        h["pct_margen_negativo"] = 100 * len(df_negativo) / len(df_margen) if len(df_margen) else 0
        h["perdida_total"] = df_negativo["Margen_Utilidad"].sum() if not df_negativo.empty else 0
        if not df_negativo.empty and "SKU_ID" in df.columns:
            top_sku = df_negativo.groupby("SKU_ID")["Margen_Utilidad"].sum().idxmin()
            h["peor_sku"] = top_sku

    # 2. Logística
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

    # 3. SKU fantasma
    if "SKU_Fantasma" in df.columns and "Precio_Venta_Final" in df.columns:
        n_fantasma = int(df["SKU_Fantasma"].sum())
        h["n_fantasma"] = n_fantasma
        h["pct_fantasma"] = 100 * n_fantasma / len(df) if len(df) else 0
        ingreso_total = df["Precio_Venta_Final"].sum()
        ingreso_fantasma = df.loc[df["SKU_Fantasma"], "Precio_Venta_Final"].sum()
        h["ingreso_riesgo"] = ingreso_fantasma
        h["pct_ingreso_riesgo"] = 100 * ingreso_fantasma / ingreso_total if ingreso_total else 0

    # 4. Paradoja stock/NPS
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

    # 5. Riesgo operativo (bodegas)
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
            h["correlacion_riesgo_operativo"] = corr
            if pd.notna(corr) and corr > 0.3:
                h["bodega_critica"] = resumen_bod["Dias_Sin_Revision"].idxmax()

    return h


def _render_narrativa(h: dict):
    st.subheader("Diagnóstico general")

    parrafos = []

    if "n_margen_negativo" in h:
        parrafos.append(
            f"**Rentabilidad.** De las transacciones analizadas, **{h['n_margen_negativo']:,} "
            f"ventas ({h['pct_margen_negativo']:.1f}%)** se están haciendo con margen negativo, "
            f"acumulando una pérdida de **${abs(h['perdida_total']):,.0f} USD**. "
            + (f"El SKU **{h['peor_sku']}** es el que más está drenando capital de forma "
               "individual. " if "peor_sku" in h else "")
            + "Esto no es un problema marginal: la empresa está subsidiando una parte "
              "significativa de su catálogo en vez de generar utilidad con él."
        )

    if "n_fantasma" in h:
        parrafos.append(
            f"**Control de inventario.** **{h['n_fantasma']:,} ventas ({h['pct_fantasma']:.1f}%)** "
            f"corresponden a productos que no existen en el maestro de inventario — representan "
            f"**${h['ingreso_riesgo']:,.0f} USD** ({h['pct_ingreso_riesgo']:.1f}% del ingreso total) "
            "que la empresa no puede auditar ni costear con certeza. Mientras esto no se resuelva, "
            "cualquier cálculo de rentabilidad global sigue teniendo un margen de error importante."
        )

    if "ciudad_logistica_critica" in h:
        fuerza = "una relación real" if abs(h["correlacion_logistica"]) >= 0.3 else "una relación débil"
        parrafos.append(
            f"**Logística.** En **{h['ciudad_logistica_critica']}** se observa {fuerza} entre "
            f"tiempo de entrega y satisfacción del cliente (correlación de "
            f"{h['correlacion_logistica']:.2f}). "
            + ("Es la señal más clara del dataset de que la operación logística está costando "
               "clientes en una zona puntual." if abs(h["correlacion_logistica"]) >= 0.3 else
               "Sin embargo, la relación es débil incluso en el peor caso, lo que sugiere que la "
               "logística no es el principal motor de la insatisfacción del cliente en este momento.")
        )

    if "categorias_paradoja" in h:
        parrafos.append(
            f"**Producto vs. inventario.** La(s) categoría(s) **{', '.join(h['categorias_paradoja'])}** "
            "combinan alta disponibilidad de stock con baja satisfacción del cliente — no es un "
            "problema de que falte producto, sino de que el producto disponible no está "
            "convenciendo al cliente."
        )

    if "bodega_critica" in h:
        parrafos.append(
            f"**Riesgo operativo.** La bodega **{h['bodega_critica']}** muestra una correlación "
            f"positiva notable ({h['correlacion_riesgo_operativo']:.2f}) entre el tiempo sin "
            "revisar su inventario y su tasa de tickets de soporte — es evidencia de que operar "
            "sin auditorías frecuentes de stock se traduce directamente en más fricción para "
            "el cliente."
        )

    if not parrafos:
        st.info("No hay suficientes datos en el filtro actual para generar el diagnóstico.")
        return

    for p in parrafos:
        st.markdown(p)
        st.write("")


def _render_plan_de_accion(h: dict):
    st.subheader("Plan de Acción")
    st.caption("Tres recomendaciones tácticas, priorizadas por complejidad de implementación.")

    recomendaciones = []

    if "n_margen_negativo" in h and h.get("n_margen_negativo", 0) > 0:
        recomendaciones.append({
            "titulo": "Revisar precios de los SKU con margen negativo",
            "complejidad": "Baja",
            "detalle": (
                f"Ajustar o descatalogar los SKU identificados en la sección de Operaciones "
                f"({h['n_margen_negativo']:,} ventas afectadas, "
                f"${abs(h.get('perdida_total', 0)):,.0f} USD en pérdida acumulada). "
                "Es un cambio de precio o de catálogo, no requiere desarrollo técnico ni "
                "cambios de proceso — se puede ejecutar en días."
            ),
        })

    if "n_fantasma" in h and h.get("n_fantasma", 0) > 0:
        recomendaciones.append({
            "titulo": "Auditar y sincronizar el catálogo de inventario",
            "complejidad": "Media",
            "detalle": (
                f"Investigar el origen de los {h['n_fantasma']:,} SKU vendidos sin registro en "
                f"inventario (${h.get('ingreso_riesgo', 0):,.0f} USD en riesgo). Requiere "
                "coordinación entre el equipo de ventas y el de inventario para decidir, "
                "producto por producto, si son altas pendientes de catalogar o errores de "
                "digitación — no es solo un ajuste de sistema."
            ),
        })

    if "bodega_critica" in h:
        recomendaciones.append({
            "titulo": f"Implementar auditorías periódicas de stock en {h['bodega_critica']}",
            "complejidad": "Alta",
            "detalle": (
                "Establecer un proceso recurrente de revisión de inventario (ej. mensual) en "
                "la bodega más rezagada, con responsables y métricas de seguimiento. Es un "
                "cambio de proceso operativo que requiere gestión de personas y tiempo para "
                "consolidarse, no solo una corrección puntual."
            ),
        })
    elif "ciudad_logistica_critica" in h and abs(h.get("correlacion_logistica", 0)) >= 0.3:
        recomendaciones.append({
            "titulo": f"Evaluar cambio de operador logístico en {h['ciudad_logistica_critica']}",
            "complejidad": "Alta",
            "detalle": (
                "Renegociar SLA o cambiar de proveedor de transporte en la zona identificada. "
                "Implica costos de transición y validación de nuevos proveedores — no es una "
                "corrección inmediata."
            ),
        })

    if not recomendaciones:
        st.info("No hay suficientes hallazgos en el filtro actual para generar un plan de acción.")
        return

    color_por_complejidad = {"Baja": "🟢", "Media": "🟡", "Alta": "🔴"}
    for i, rec in enumerate(recomendaciones, start=1):
        with st.expander(f"{i}. {rec['titulo']}  —  {color_por_complejidad[rec['complejidad']]} Complejidad {rec['complejidad']}"):
            st.write(rec["detalle"])


def render(df_filtrado: pd.DataFrame):
    st.header("📋 Análisis Final")
    st.caption(
        "Síntesis de los hallazgos de Operaciones y Cliente, con lenguaje de negocio — "
        "para leer antes de la reunión con la junta directiva, no para leer código."
    )

    if df_filtrado is None or df_filtrado.empty:
        st.warning("No hay datos para mostrar con los filtros actuales. Ajusta el sidebar.")
        return

    hallazgos = _calcular_hallazgos(df_filtrado)

    _render_narrativa(hallazgos)
    st.divider()
    _render_plan_de_accion(hallazgos)