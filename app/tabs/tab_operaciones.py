"""
tab_operaciones.py
--------------------
Pestaña "Operaciones": responde las Preguntas 1, 2 y 3 del reto.

1. Fuga de Capital y Rentabilidad — SKUs con margen negativo.
2. Crisis Logística — correlación Tiempo de Entrega vs NPS por ciudad/bodega.
3. Venta Invisible — impacto financiero de las ventas con SKU fantasma.

Recibe el DataFrame maestro YA FILTRADO por los controles del sidebar
(app.py se lo pasa). No vuelve a limpiar ni a integrar nada — todas las
columnas que usa (Margen_Utilidad, SKU_Fantasma, Brecha_Entrega, etc.)
ya vienen calculadas por src/feature_engineering.py.
"""

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Pregunta 1 — Fuga de capital y rentabilidad
# ---------------------------------------------------------------------------

def _seccion_margenes(df: pd.DataFrame):
    st.subheader("1. Fuga de Capital y Rentabilidad")
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
    c1.metric("Ventas con margen negativo", f"{n_negativo:,}", f"{pct_negativo:.1f}% del total")
    c2.metric("Pérdida acumulada", f"${perdida_total:,.0f} USD")
    c3.metric("Margen promedio general", f"${df_con_margen['Margen_Utilidad'].mean():,.2f} USD")

    if df_negativo.empty:
        st.success("No se encontraron ventas con margen negativo en el filtro actual.")
        return

    # --- Gráfico: top 15 SKU con peor margen acumulado ---
    top_sku_negativo = (
        df_negativo.groupby("SKU_ID")["Margen_Utilidad"]
        .agg(Perdida_Total="sum", N_Ventas="count")
        .sort_values("Perdida_Total")
        .head(15)
    )
    st.markdown("**Top 15 SKU con mayor pérdida acumulada**")
    st.bar_chart(top_sku_negativo["Perdida_Total"])

    # --- Desglose por canal: ¿es un problema puntual o generalizado? ---
    if "Canal_Venta" in df_negativo.columns:
        st.markdown("**¿Se concentra en algún canal de venta?**")
        por_canal = (
            df_negativo.groupby("Canal_Venta")["Margen_Utilidad"]
            .agg(Perdida_Total="sum", N_Ventas="count")
            .sort_values("Perdida_Total")
        )
        st.bar_chart(por_canal["Perdida_Total"])

    with st.expander("Ver tabla completa de SKUs con margen negativo"):
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
        st.download_button(
            "⬇️ Descargar tabla de márgenes negativos",
            data=tabla.to_csv(index=False).encode("utf-8-sig"),
            file_name="skus_margen_negativo.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Pregunta 2 — Crisis logística y cuellos de botella
# ---------------------------------------------------------------------------

def _seccion_logistica(df: pd.DataFrame):
    st.subheader("2. Crisis Logística y Cuellos de Botella")
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
        st.bar_chart(te_por_ciudad)
    with c2:
        st.markdown("**NPS promedio por ciudad**")
        nps_por_ciudad = df_validos.groupby("Ciudad_Destino")["Satisfaccion_NPS"].mean().sort_values()
        st.bar_chart(nps_por_ciudad)

    # --- Correlación Tiempo_Entrega vs NPS, calculada POR ciudad ---
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
        st.error(
            f"🚨 **Zona crítica: {zona_critica['Ciudad_Destino']}** — correlación de "
            f"{zona_critica['Correlacion_Tiempo_vs_NPS']:.2f}. Es la ciudad donde el "
            "tiempo de entrega afecta más fuertemente la satisfacción del cliente."
        )

    # --- Lo mismo pero por bodega, si está disponible ---
    if "Bodega_Origen" in df_validos.columns:
        with st.expander("Ver el mismo análisis por Bodega_Origen"):
            te_bodega = df_validos.groupby("Bodega_Origen")["Tiempo_Entrega_Real"].mean().sort_values(ascending=False)
            st.bar_chart(te_bodega)


# ---------------------------------------------------------------------------
# Pregunta 3 — Venta invisible (SKU fantasma)
# ---------------------------------------------------------------------------

def _seccion_sku_fantasma(df: pd.DataFrame):
    st.subheader("3. Análisis de la Venta Invisible")
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
    c1.metric("Ventas fantasma", f"{n_fantasma:,}", f"{pct_fantasma:.1f}% de las ventas")
    c2.metric("SKUs distintos involucrados", f"{df.loc[df['SKU_Fantasma'], 'SKU_ID'].nunique():,}")
    c3.metric("Ingreso en riesgo (USD)", f"${ingreso_fantasma:,.0f}")
    c4.metric("% del ingreso total", f"{pct_ingreso_riesgo:.1f}%")

    if n_fantasma == 0:
        st.success("No hay ventas fantasma en el filtro actual.")
        return

    # --- ¿Se concentra en ciudades, canales o fechas específicas? ---
    c1, c2 = st.columns(2)
    with c1:
        if "Canal_Venta" in df.columns:
            st.markdown("**Ventas fantasma por canal**")
            por_canal = df[df["SKU_Fantasma"]]["Canal_Venta"].value_counts()
            st.bar_chart(por_canal)
    with c2:
        if "Ciudad_Destino" in df.columns:
            st.markdown("**Ventas fantasma por ciudad**")
            por_ciudad = df[df["SKU_Fantasma"]]["Ciudad_Destino"].value_counts()
            st.bar_chart(por_ciudad)

    with st.expander(f"Ver los {n_fantasma:,} registros de venta fantasma"):
        cols_mostrar = [c for c in [
            "Transaccion_ID", "SKU_ID", "Fecha_Venta", "Cantidad_Vendida",
            "Precio_Venta_Final", "Canal_Venta", "Ciudad_Destino",
        ] if c in df.columns]
        tabla_fantasma = df.loc[df["SKU_Fantasma"], cols_mostrar]
        st.dataframe(tabla_fantasma, width="stretch")
        st.download_button(
            "⬇️ Descargar ventas fantasma",
            data=tabla_fantasma.to_csv(index=False).encode("utf-8-sig"),
            file_name="ventas_sku_fantasma.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Render principal
# ---------------------------------------------------------------------------

def render(df_filtrado: pd.DataFrame):
    st.header("📦 Operaciones")

    if df_filtrado is None or df_filtrado.empty:
        st.warning("No hay datos para mostrar con los filtros actuales. Ajusta el sidebar.")
        return

    _seccion_margenes(df_filtrado)
    st.divider()
    _seccion_logistica(df_filtrado)
    st.divider()
    _seccion_sku_fantasma(df_filtrado)