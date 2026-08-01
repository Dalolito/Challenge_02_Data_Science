import altair as alt
import pandas as pd
import streamlit as st

from .ui_helpers import titulo_seccion, bar_chart_fijo, badge_inline, BADGE_INLINE_CSS, metric_box, render_html_block, download_button_verde, guardar_altair


# Pregunta 1 — Fuga de capital y rentabilidad

def _seccion_margenes(df: pd.DataFrame):
    titulo_seccion("1. Fuga de Capital y Rentabilidad")
    st.caption(
        "SKUs que se están vendiendo con margen negativo — cada venta de estos "
        "productos le cuesta dinero a la empresa en vez de generarlo."
    )

    df_con_margen = df[df["Margen_Utilidad"].notna()].copy()
    df_negativo = df_con_margen[df_con_margen["Margen_Utilidad"] < 0]

    if df_con_margen.empty:
        st.warning("No hay transacciones con margen calculable en el filtro actual.")
        return

    n_negativo = len(df_negativo)
    pct_negativo = 100 * n_negativo / len(df_con_margen)
    perdida_total = df_negativo["Margen_Utilidad"].sum()  # ya es negativo

    c1, c2, c3 = st.columns(3)
    metric_box(c1, "Ventas con margen negativo", f"{n_negativo:,}",
               delta=f"↑ {pct_negativo:.1f}% del total", delta_es_bueno=False)
    metric_box(c2, "Pérdida acumulada", f"{perdida_total:,.0f} USD")
    metric_box(c3, "Margen promedio general", f"{df_con_margen['Margen_Utilidad'].mean():,.2f} USD")

    if df_negativo.empty:
        st.success("No se encontraron ventas con margen negativo en el filtro actual.")
        return

    # Top 15 SKU con peor margen acumulado
    top_sku_negativo = (
        df_negativo.groupby("SKU_ID")["Margen_Utilidad"]
        .agg(Perdida_Total="sum", N_Ventas="count")
        .sort_values("Perdida_Total")
        .head(15)
    )
    st.markdown("**Top 15 SKU con mayor pérdida acumulada**")
    bar_chart_fijo(top_sku_negativo["Perdida_Total"], guardar="operaciones_top15_sku_margen_negativo")

    peor_sku = top_sku_negativo.index[0]
    peor_perdida = top_sku_negativo.iloc[0]["Perdida_Total"]
    st.markdown(
        f"De la gráfica anterior observamos que el SKU **{peor_sku}** concentra la mayor "
        f"pérdida individual, con **{abs(peor_perdida):,.0f} USD** acumulados en negativo. "
        f"Entre los 15 SKU más problemáticos suman **{abs(top_sku_negativo['Perdida_Total'].sum()):,.0f} USD** "
        "de pérdida — no es un caso aislado, es un grupo de productos que la empresa debería "
        "revisar de precio o descatalogar."
    )

    # Desglose por canal
    if "Canal_Venta" in df_negativo.columns:
        st.markdown("**¿Se concentra en algún canal de venta?**")
        por_canal = (
            df_negativo.groupby("Canal_Venta")["Margen_Utilidad"]
            .agg(Perdida_Total="sum", N_Ventas="count")
            .sort_values("Perdida_Total")
        )
        pastel_canal = por_canal.reset_index()
        pastel_canal["Perdida_Abs"] = pastel_canal["Perdida_Total"].abs()
        pastel_canal["Pct"] = 100 * pastel_canal["Perdida_Abs"] / pastel_canal["Perdida_Abs"].sum()

        grafico_pastel = (
            alt.Chart(pastel_canal)
            .mark_arc(innerRadius=0)
            .encode(
                theta=alt.Theta("Perdida_Abs:Q", stack=True),
                color=alt.Color("Canal_Venta:N", title="Canal",
                                 scale=alt.Scale(scheme="blues")),
                tooltip=["Canal_Venta", alt.Tooltip("Perdida_Abs:Q", title="Pérdida (USD)", format=",.0f"),
                         alt.Tooltip("Pct:Q", title="% del total", format=".1f")],
            )
            .properties(height=350, width=350)
        )
        texto_pastel = (
            alt.Chart(pastel_canal)
            .mark_text(radius=140, size=13, color="white", fontWeight="bold")
            .encode(
                theta=alt.Theta("Perdida_Abs:Q", stack=True),
                text=alt.Text("Pct:Q", format=".1f"),
            )
        )
        st.altair_chart(grafico_pastel + texto_pastel, width="content")
        guardar_altair(grafico_pastel + texto_pastel, "operaciones_perdida_por_canal")

        canal_peor = por_canal.index[0]
        pct_min = pastel_canal["Pct"].min()
        pct_max = pastel_canal["Pct"].max()

        if (pct_max - pct_min) <= 10:
            st.markdown(
                f"Aunque el canal **{canal_peor}** concentra técnicamente la mayor pérdida "
                f"({abs(por_canal.iloc[0]['Perdida_Total']):,.0f} USD en "
                f"{int(por_canal.iloc[0]['N_Ventas']):,} ventas), la diferencia con los demás "
                f"canales es mínima — todos se mueven entre {pct_min:.1f}% y {pct_max:.1f}% de "
                "la pérdida total. Esto indica que **el canal no es el problema real**: es un "
                "problema de pricing sistemático que afecta a todos por igual, no una falla "
                "puntual concentrada en uno solo."
            )
        else:
            st.markdown(
                f"De la gráfica anterior observamos que el canal **{canal_peor}** es donde más se "
                f"concentra la pérdida ({abs(por_canal.iloc[0]['Perdida_Total']):,.0f} USD en "
                f"{int(por_canal.iloc[0]['N_Ventas']):,} ventas). Esto sugiere que el problema no es "
                "solo de producto, sino también de cómo se está fijando el precio en ese canal "
                "específico frente a los demás."
            )

    with st.expander("**Ver tabla completa de SKUs con margen negativo**"):
        tabla = (
            df_negativo.groupby("SKU_ID")
            .agg(
                Categoria=("Categoria", "first"),
                Canal_Predominante=("Canal_Venta", lambda s: s.mode().iloc[0] if not s.mode().empty else "—"),
                N_Ventas=("Margen_Utilidad", "count"),
                Perdida_Total=("Margen_Utilidad", "sum"),
                Margen_Promedio=("Margen_Utilidad", "mean"),
            )
            .sort_values("Perdida_Total")
            .reset_index()
        )
        st.dataframe(tabla, width="stretch")
        download_button_verde(
            "Descargar tabla de márgenes negativos",
            data=tabla.to_csv(index=False).encode("utf-8-sig"),
            file_name="skus_margen_negativo.csv",
            mime="text/csv",
            key="btn_descarga_margenes",
        )


# Pregunta 2 — Crisis logística y cuellos de botella

def _seccion_logistica(df: pd.DataFrame):
    titulo_seccion("2. Crisis Logística y Cuellos de Botella")
    st.caption(
        "¿En qué ciudades/bodegas la correlación entre Tiempo de Entrega y NPS bajo "
        "es más fuerte? Esa es la zona que necesita un cambio de operador."
    )

    cols_necesarias = {"Ciudad_Destino", "Tiempo_Entrega_Real", "Satisfaccion_NPS"}
    if not cols_necesarias.issubset(df.columns):
        st.warning("Faltan columnas necesarias para este análisis.")
        return

    df_validos = df.dropna(subset=["Ciudad_Destino", "Tiempo_Entrega_Real", "Satisfaccion_NPS"])
    if df_validos.empty:
        st.warning("No hay datos suficientes en el filtro actual (revisa Ciudad_Invalida).")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tiempo de entrega promedio por ciudad**")
        te_por_ciudad = df_validos.groupby("Ciudad_Destino")["Tiempo_Entrega_Real"].mean().sort_values(ascending=False)
        bar_chart_fijo(te_por_ciudad, guardar="operaciones_tiempo_entrega_por_ciudad")
    with c2:
        st.markdown("**NPS promedio por ciudad**")
        nps_por_ciudad = df_validos.groupby("Ciudad_Destino")["Satisfaccion_NPS"].mean().sort_values()
        bar_chart_fijo(nps_por_ciudad, guardar="operaciones_nps_por_ciudad")

    ciudad_mas_lenta = te_por_ciudad.index[0]
    ciudad_peor_nps = nps_por_ciudad.index[0]
    if ciudad_mas_lenta == ciudad_peor_nps:
        st.markdown(
            f"De las gráficas anteriores observamos que **{ciudad_mas_lenta}** es al mismo "
            "tiempo la ciudad con el tiempo de entrega más largo y el NPS más bajo — es un "
            "primer indicio de que ahí el retraso logístico sí le está costando satisfacción "
            "a la empresa."
        )
    else:
        st.markdown(
            f"De las gráficas anteriores observamos que **{ciudad_mas_lenta}** tiene el mayor "
            f"tiempo de entrega, pero **{ciudad_peor_nps}** es la que peor NPS reporta — no son "
            "la misma ciudad, así que el tiempo de entrega no parece ser, por sí solo, el "
            "principal motivo de insatisfacción. Hay que revisar la correlación real antes de "
            "sacar una conclusión."
        )

    # Correlación Tiempo_Entrega vs NPS, por ciudad
    st.markdown("**Correlación Tiempo de Entrega ↔ NPS, por ciudad**")
    st.caption(
        "Un valor negativo fuerte significa: a mayor tiempo de entrega, peor NPS. "
        "Eso es lo que estamos buscando — dónde el retraso logístico está pegándole "
        "directamente a la satisfacción del cliente."
    )
    correlaciones = (
        df_validos.groupby("Ciudad_Destino")
        .apply(lambda g: g["Tiempo_Entrega_Real"].corr(g["Satisfaccion_NPS"]) if len(g) > 5 else None)
        .dropna()
        .sort_values()
    )
    if correlaciones.empty:
        st.info("No hay suficientes datos por ciudad para calcular correlaciones confiables.")
    else:
        tabla_corr = correlaciones.reset_index()
        tabla_corr.columns = ["Ciudad_Destino", "Correlacion_Tiempo_vs_NPS"]
        st.dataframe(tabla_corr, width="stretch", hide_index=True)

        zona_critica = tabla_corr.iloc[0]
        corr_valor = zona_critica["Correlacion_Tiempo_vs_NPS"]
        if abs(corr_valor) >= 0.3:
            st.markdown(
                f"De la tabla anterior observamos que **{zona_critica['Ciudad_Destino']}** tiene "
                f"la correlación más negativa ({corr_valor:.2f}) entre tiempo de entrega y NPS — "
                "es la ciudad donde el retraso logístico más le está costando satisfacción del "
                "cliente a la empresa, y la primera candidata para un cambio de operador logístico."
            )
        else:
            st.markdown(
                f"De la tabla anterior observamos que, incluso en la ciudad con la correlación "
                f"más marcada (**{zona_critica['Ciudad_Destino']}**, {corr_valor:.2f}), la relación "
                "entre tiempo de entrega y NPS es débil. Esto sugiere que, con los datos actuales, "
                "la logística no es el principal motor de la insatisfacción del cliente — vale la "
                "pena buscar la causa en otro factor, como la calidad del producto."
            )

    # Mismo análisis por bodega (si está disponible)
    if "Bodega_Origen" in df_validos.columns:
        with st.expander("**Ver el mismo análisis por Bodega_Origen**"):
            te_bodega = df_validos.groupby("Bodega_Origen")["Tiempo_Entrega_Real"].mean().sort_values(ascending=False)
            bar_chart_fijo(te_bodega, guardar="operaciones_tiempo_entrega_por_bodega")


# Pregunta 3 — Venta invisible (SKU fantasma)

def _seccion_sku_fantasma(df: pd.DataFrame):
    titulo_seccion("3. Análisis de la Venta Invisible")
    st.caption(
        "Ventas cuyo SKU no existe en el maestro de inventario — no se puede "
        "calcular su costo ni su margen real, así que representan un riesgo de "
        "control financiero, no solo un problema de catálogo."
    )

    if "SKU_Fantasma" not in df.columns:
        st.warning("La columna SKU_Fantasma no está disponible.")
        return

    n_fantasma = int(df["SKU_Fantasma"].sum())
    n_total = len(df)
    pct_fantasma = 100 * n_fantasma / n_total if n_total else 0

    ingreso_fantasma = df.loc[df["SKU_Fantasma"], "Precio_Venta_Final"].sum()
    ingreso_total = df["Precio_Venta_Final"].sum()
    pct_ingreso_riesgo = 100 * ingreso_fantasma / ingreso_total if ingreso_total else 0

    c1, c2, c3, c4 = st.columns(4)
    metric_box(c1, "Ventas fantasma", f"{n_fantasma:,}",
               delta=f"↑ {pct_fantasma:.1f}% de las ventas", delta_es_bueno=False)
    metric_box(c2, "SKUs distintos involucrados", f"{df.loc[df['SKU_Fantasma'], 'SKU_ID'].nunique():,}")
    metric_box(c3, "Ingreso en riesgo (USD)", f"{ingreso_fantasma:,.0f}")
    metric_box(c4, "% del ingreso total", f"{pct_ingreso_riesgo:.1f}%")

    if n_fantasma == 0:
        st.success("No hay ventas fantasma en el filtro actual.")
        return

    # ¿Se concentra en ciudades o canales específicos?
    c1, c2 = st.columns(2)
    canal_top, canal_segundo = None, None
    ciudad_top, ciudad_segundo = None, None
    with c1:
        if "Canal_Venta" in df.columns:
            st.markdown("**Ventas fantasma por canal**")
            por_canal = df[df["SKU_Fantasma"]]["Canal_Venta"].value_counts()
            bar_chart_fijo(por_canal, guardar="operaciones_ventas_fantasma_por_canal")
            if not por_canal.empty:
                canal_top = por_canal.index[0]
                if len(por_canal) >= 2 and (por_canal.iloc[0] - por_canal.iloc[1]) / por_canal.iloc[0] <= 0.15:
                    canal_segundo = por_canal.index[1]
    with c2:
        if "Ciudad_Destino" in df.columns:
            st.markdown("**Ventas fantasma por ciudad**")
            por_ciudad = df[df["SKU_Fantasma"]]["Ciudad_Destino"].value_counts()
            bar_chart_fijo(por_ciudad, guardar="operaciones_ventas_fantasma_por_ciudad")
            if not por_ciudad.empty:
                ciudad_top = por_ciudad.index[0]
                if len(por_ciudad) >= 2 and (por_ciudad.iloc[0] - por_ciudad.iloc[1]) / por_ciudad.iloc[0] <= 0.15:
                    ciudad_segundo = por_ciudad.index[1]

    if canal_top is not None:
        canal_txt = (
            f"el canal **{canal_top}** concentra la mayor cantidad de ventas fantasma "
            f"(muy cerca de **{canal_segundo}**, prácticamente empatados)"
            if canal_segundo else
            f"el canal **{canal_top}** concentra la mayor cantidad de ventas fantasma"
        )
        ciudad_txt = ""
        if ciudad_top:
            ciudad_txt = (
                f", y **{ciudad_top}** y **{ciudad_segundo}** son las ciudades con más "
                "casos, prácticamente empatadas"
                if ciudad_segundo else
                f", y **{ciudad_top}** la mayor cantidad por ciudad"
            )
        st.markdown(
            f"De las gráficas anteriores observamos que {canal_txt}{ciudad_txt}. Esto apunta "
            "a que el problema no está distribuido al azar en todo el negocio, sino asociado "
            "a un punto específico del proceso de venta — probablemente ahí es donde vale la "
            "pena revisar primero si el catálogo de productos está desactualizado."
        )

    with st.expander(f"**Ver los {n_fantasma:,} registros de venta fantasma**"):
        cols_mostrar = [c for c in [
            "Transaccion_ID", "SKU_ID", "Fecha_Venta", "Cantidad_Vendida",
            "Precio_Venta_Final", "Canal_Venta", "Ciudad_Destino",
        ] if c in df.columns]
        tabla_fantasma = df.loc[df["SKU_Fantasma"], cols_mostrar]
        st.dataframe(tabla_fantasma, width="stretch")
        download_button_verde(
            "Descargar ventas fantasma",
            data=tabla_fantasma.to_csv(index=False).encode("utf-8-sig"),
            file_name="ventas_sku_fantasma.csv",
            mime="text/csv",
            key="btn_descarga_fantasma",
        )


# Render principal

def render(df_filtrado: pd.DataFrame):
    st.header("📦 Operaciones")

    render_html_block(
        BADGE_INLINE_CSS,
        f"""
        <div style="font-size:1rem; line-height:1.6;">
        Esta pestaña responde las {badge_inline("3 primeras preguntas de alta gerencia")} del
        reto, todas con evidencia calculada directamente sobre el dataset maestro ya filtrado
        en el sidebar:
        <ol style="margin-top:10px;">
            <li>{badge_inline("Fuga de Capital y Rentabilidad:")} qué SKUs se venden con margen
                negativo y cuánto le está costando eso a la empresa.</li>
            <li>{badge_inline("Crisis Logística y Cuellos de Botella:")} dónde la correlación
                entre tiempo de entrega y NPS es más fuerte, para identificar la zona que
                necesita un cambio de operador.</li>
            <li>{badge_inline("Análisis de la Venta Invisible:")} el impacto financiero de las
                ventas cuyo SKU no existe en el inventario (SKU Fantasma).</li>
        </ol>
        </div>
        """,
    )

    if df_filtrado is None or df_filtrado.empty:
        st.warning("No hay datos para mostrar con los filtros actuales. Ajusta el sidebar.")
        return

    _seccion_margenes(df_filtrado)
    st.divider()
    _seccion_logistica(df_filtrado)
    st.divider()
    _seccion_sku_fantasma(df_filtrado)