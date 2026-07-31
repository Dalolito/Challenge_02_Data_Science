"""
tab_cliente.py
----------------
Pestaña "Cliente": responde las Preguntas 4 y 5 del reto.

4. Diagnóstico de Fidelidad — categorías con alta disponibilidad (stock alto)
   pero sentimiento de cliente negativo (¿mala calidad o sobrecosto?).
5. Storytelling de Riesgo Operativo — relación entre antigüedad de la última
   revisión de stock y la tasa de tickets de soporte, por bodega.

Recibe el DataFrame maestro YA FILTRADO por el sidebar. Solo lee columnas
que ya vienen calculadas por feature_engineering.py / cleaning.py.
"""

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Pregunta 4 — Paradoja stock alto / sentimiento negativo
# ---------------------------------------------------------------------------

def _seccion_paradoja_stock(df: pd.DataFrame):
    st.subheader("4. Diagnóstico de Fidelidad — Paradoja Stock Alto / Sentimiento Negativo")
    st.caption(
        "¿Hay categorías con mucho stock disponible pero mala percepción del cliente? "
        "Eso apunta a un problema de calidad de producto, no de disponibilidad."
    )

    cols_necesarias = {"Categoria", "Stock_Actual", "Satisfaccion_NPS"}
    if not cols_necesarias.issubset(df.columns):
        st.warning("Faltan columnas necesarias para este análisis.")
        return

    df_validos = df.dropna(subset=["Categoria", "Stock_Actual", "Satisfaccion_NPS"])
    if df_validos.empty:
        st.warning("No hay datos suficientes en el filtro actual.")
        return

    resumen = (
        df_validos.groupby("Categoria")
        .agg(
            Stock_Promedio=("Stock_Actual", "mean"),
            NPS_Promedio=("Satisfaccion_NPS", "mean"),
            Rating_Producto_Promedio=("Rating_Producto", "mean"),
            Ratio_Soporte=("Ratio_Soporte_Categoria", "first"),
            N_Ventas=("Transaccion_ID", "count"),
        )
        .reset_index()
    )

    # Umbral: por encima de la mediana de stock y por debajo de la mediana de NPS
    stock_mediana = resumen["Stock_Promedio"].median()
    nps_mediana = resumen["NPS_Promedio"].median()
    resumen["Paradoja"] = (
        (resumen["Stock_Promedio"] > stock_mediana) & (resumen["NPS_Promedio"] < nps_mediana)
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stock promedio por categoría**")
        st.bar_chart(resumen.set_index("Categoria")["Stock_Promedio"])
    with c2:
        st.markdown("**NPS promedio por categoría**")
        st.bar_chart(resumen.set_index("Categoria")["NPS_Promedio"])

    st.markdown("**Cruce Stock vs NPS vs Rating de producto**")
    st.dataframe(
        resumen.sort_values("Paradoja", ascending=False),
        width="stretch", hide_index=True,
    )

    categorias_paradoja = resumen[resumen["Paradoja"]]["Categoria"].tolist()
    if categorias_paradoja:
        st.warning(
            f"⚠️ **Categorías en paradoja (stock alto + NPS bajo):** "
            f"{', '.join(categorias_paradoja)}. "
            "Compara su Rating_Producto_Promedio en la tabla: si también es bajo, "
            "apunta a un problema de calidad de producto; si el rating es aceptable "
            "pero el NPS es bajo, puede ser un tema de precio/sobrecosto percibido."
        )
    else:
        st.success("No se detectaron categorías en la zona de paradoja (stock alto + NPS bajo).")


# ---------------------------------------------------------------------------
# Pregunta 5 — Antigüedad de revisión de stock vs tickets de soporte
# ---------------------------------------------------------------------------

def _seccion_riesgo_operativo(df: pd.DataFrame):
    st.subheader("5. Storytelling de Riesgo Operativo")
    st.caption(
        "¿Las bodegas que llevan más tiempo sin revisar su inventario tienen más "
        "tickets de soporte? Eso indicaría que están 'operando a ciegas'."
    )

    cols_necesarias = {"Bodega_Origen", "Ultima_Revision", "Ticket_Soporte_Abierto"}
    if not cols_necesarias.issubset(df.columns):
        st.warning("Faltan columnas necesarias para este análisis.")
        return

    df_validos = df.dropna(subset=["Bodega_Origen", "Ultima_Revision"]).copy()
    if df_validos.empty:
        st.warning("No hay datos suficientes en el filtro actual.")
        return

    fecha_referencia = df_validos["Ultima_Revision"].max()
    df_validos["Dias_Sin_Revision"] = (fecha_referencia - df_validos["Ultima_Revision"]).dt.days

    resumen = (
        df_validos.groupby("Bodega_Origen")
        .agg(
            Dias_Sin_Revision_Promedio=("Dias_Sin_Revision", "mean"),
            Tasa_Ticket_Soporte=("Ticket_Soporte_Abierto", "mean"),
            N_Transacciones=("Transaccion_ID", "count"),
        )
        .reset_index()
    )
    resumen["Tasa_Ticket_Soporte"] = (resumen["Tasa_Ticket_Soporte"] * 100).round(1)
    resumen = resumen.sort_values("Dias_Sin_Revision_Promedio", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Días promedio sin revisión de stock, por bodega**")
        st.bar_chart(resumen.set_index("Bodega_Origen")["Dias_Sin_Revision_Promedio"])
    with c2:
        st.markdown("**Tasa de tickets de soporte (%), por bodega**")
        st.bar_chart(resumen.set_index("Bodega_Origen")["Tasa_Ticket_Soporte"])

    st.markdown("**Tabla comparativa**")
    st.dataframe(resumen, width="stretch", hide_index=True)

    correlacion = resumen["Dias_Sin_Revision_Promedio"].corr(resumen["Tasa_Ticket_Soporte"])
    st.metric("Correlación (días sin revisión ↔ tasa de tickets)", f"{correlacion:.2f}")

    if pd.notna(correlacion) and correlacion > 0.3:
        bodega_critica = resumen.iloc[0]["Bodega_Origen"]
        st.error(
            f"🚨 La correlación es positiva y notable: a más tiempo sin revisar stock, "
            f"más tickets de soporte. **{bodega_critica}** es la bodega más rezagada en "
            "revisión — es la que más urge auditar primero."
        )
    else:
        st.info(
            "La correlación entre antigüedad de revisión y tickets de soporte es débil "
            "en el filtro actual — no parece ser el principal factor explicativo de los "
            "tickets de soporte."
        )


# ---------------------------------------------------------------------------
# Render principal
# ---------------------------------------------------------------------------

def render(df_filtrado: pd.DataFrame):
    st.header("😊 Cliente")

    if df_filtrado is None or df_filtrado.empty:
        st.warning("No hay datos para mostrar con los filtros actuales. Ajusta el sidebar.")
        return

    _seccion_paradoja_stock(df_filtrado)
    st.divider()
    _seccion_riesgo_operativo(df_filtrado)