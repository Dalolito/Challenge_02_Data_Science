"""
Pestaña "Cliente": responde las Preguntas 4 y 5 del reto (paradoja
stock alto/sentimiento negativo y riesgo operativo por bodega).
Recibe el DataFrame maestro ya filtrado por el sidebar.
"""

import pandas as pd
import streamlit as st

from .ui_helpers import titulo_seccion, bar_chart_fijo, badge_inline, BADGE_INLINE_CSS, metric_box, render_html_block


# Pregunta 4 — Paradoja stock alto / sentimiento negativo

def _seccion_paradoja_stock(df: pd.DataFrame):
    titulo_seccion("4. Diagnóstico de Fidelidad — Paradoja Stock Alto / Sentimiento Negativo")
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

    # Umbral: stock sobre la mediana y NPS bajo la mediana
    stock_mediana = resumen["Stock_Promedio"].median()
    nps_mediana = resumen["NPS_Promedio"].median()
    resumen["Paradoja"] = (
        (resumen["Stock_Promedio"] > stock_mediana) & (resumen["NPS_Promedio"] < nps_mediana)
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stock promedio por categoría**")
        bar_chart_fijo(resumen.set_index("Categoria")["Stock_Promedio"])
    with c2:
        st.markdown("**NPS promedio por categoría**")
        bar_chart_fijo(resumen.set_index("Categoria")["NPS_Promedio"])

    categoria_mas_stock = resumen.loc[resumen["Stock_Promedio"].idxmax(), "Categoria"]
    categoria_peor_nps = resumen.loc[resumen["NPS_Promedio"].idxmin(), "Categoria"]
    st.markdown(
        f"De las gráficas anteriores observamos que **{categoria_mas_stock}** es la categoría "
        f"con más stock disponible, mientras que **{categoria_peor_nps}** es la que peor NPS "
        "reporta. Cuando ambas coinciden en la misma categoría, es una señal de que el problema "
        "no es de disponibilidad — hay producto de sobra — sino de algo que el cliente está "
        "percibiendo mal (calidad o precio)."
    )

    st.markdown("**Cruce Stock vs NPS vs Rating de producto**")
    st.dataframe(
        resumen.sort_values("Paradoja", ascending=False),
        width="stretch", hide_index=True,
    )

    categorias_paradoja = resumen[resumen["Paradoja"]]["Categoria"].tolist()
    if categorias_paradoja:
        detalles = []
        for cat in categorias_paradoja:
            fila = resumen[resumen["Categoria"] == cat].iloc[0]
            rating = fila["Rating_Producto_Promedio"]
            if pd.notna(rating) and rating < 3.5:
                diagnostico = "su Rating_Producto_Promedio también es bajo, lo que apunta a un problema de calidad de producto"
            else:
                diagnostico = "su Rating_Producto_Promedio es aceptable pero el NPS sigue bajo, lo que apunta más a un tema de precio o sobrecosto percibido"
            detalles.append(f"**{cat}** ({diagnostico})")

        st.markdown(
            f"De la tabla anterior observamos que {', '.join(detalles)}. "
            "Es la categoría que la empresa debería revisar primero para entender por qué, "
            "teniendo suficiente inventario, no está logrando la satisfacción del cliente."
        )
    else:
        st.markdown(
            "De la tabla anterior observamos que ninguna categoría combina stock alto con NPS "
            "bajo — no hay evidencia de la paradoja stock/sentimiento en el filtro actual."
        )


# Pregunta 5 — Antigüedad de revisión de stock vs tickets de soporte

def _seccion_riesgo_operativo(df: pd.DataFrame):
    titulo_seccion("5. Storytelling de Riesgo Operativo")
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
        bar_chart_fijo(resumen.set_index("Bodega_Origen")["Dias_Sin_Revision_Promedio"])
    with c2:
        st.markdown("**Tasa de tickets de soporte (%), por bodega**")
        bar_chart_fijo(resumen.set_index("Bodega_Origen")["Tasa_Ticket_Soporte"])

    bodega_mas_rezagada = resumen.iloc[0]["Bodega_Origen"]
    bodega_mas_tickets = resumen.loc[resumen["Tasa_Ticket_Soporte"].idxmax(), "Bodega_Origen"]
    st.markdown(
        f"De las gráficas anteriores observamos que **{bodega_mas_rezagada}** es la bodega que "
        f"más tiempo lleva sin revisar su inventario, y **{bodega_mas_tickets}** es la que más "
        "tickets de soporte genera" +
        (" — son la misma bodega, lo cual refuerza la hipótesis de que operar sin auditar el "
         "stock se traduce directamente en más problemas para el cliente."
         if bodega_mas_rezagada == bodega_mas_tickets else
         " — no son la misma bodega, así que conviene revisar la correlación real antes de "
         "asumir una relación causal entre ambas cosas.")
    )

    st.markdown("**Tabla comparativa**")
    st.dataframe(resumen, width="stretch", hide_index=True)

    correlacion = resumen["Dias_Sin_Revision_Promedio"].corr(resumen["Tasa_Ticket_Soporte"])
    _, col_corr, _ = st.columns([1, 1, 1])
    metric_box(col_corr, "Correlación (días sin revisión ↔ tasa de tickets)", f"{correlacion:.2f}")

    if pd.notna(correlacion) and correlacion > 0.3:
        bodega_critica = resumen.iloc[0]["Bodega_Origen"]
        st.markdown(
            f"De la tabla anterior observamos que la correlación es positiva y notable "
            f"({correlacion:.2f}): a más tiempo sin revisar stock, más tickets de soporte. "
            f"**{bodega_critica}** es la bodega más rezagada en revisión — es la que más urge "
            "auditar primero, antes de que siga generando más costo oculto en soporte al cliente."
        )
    else:
        st.markdown(
            f"De la tabla anterior observamos que la correlación entre antigüedad de revisión y "
            f"tickets de soporte es débil ({correlacion:.2f}) en el filtro actual — la falta de "
            "revisión de inventario no parece ser, por sí sola, el principal factor detrás de los "
            "tickets de soporte; vale la pena buscar la causa en otro lado."
        )


# Render principal

def render(df_filtrado: pd.DataFrame):
    st.header("😊 Cliente")

    render_html_block(
        BADGE_INLINE_CSS,
        f"""
        <div style="font-size:1rem; line-height:1.6;">
        Esta pestaña responde las {badge_inline("2 últimas preguntas de alta gerencia")} del
        reto, cruzando la voz del cliente (feedback) con datos operativos de inventario y
        logística:
        <ol style="margin-top:10px;">
            <li>{badge_inline("Diagnóstico de Fidelidad:")} si hay categorías con stock alto
                pero sentimiento de cliente negativo, y si eso apunta a un problema de calidad
                de producto o de precio.</li>
            <li>{badge_inline("Storytelling de Riesgo Operativo:")} si las bodegas que más
                tiempo llevan sin revisar su inventario son también las que más tickets de
                soporte generan.</li>
        </ol>
        </div>
        """,
    )

    if df_filtrado is None or df_filtrado.empty:
        st.warning("No hay datos para mostrar con los filtros actuales. Ajusta el sidebar.")
        return

    _seccion_paradoja_stock(df_filtrado)
    st.divider()
    _seccion_riesgo_operativo(df_filtrado)