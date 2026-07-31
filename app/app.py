"""
Punto de entrada de la app Streamlit: orquesta la carga/limpieza (src/),
los filtros globales del sidebar y las 4 pestañas. Sin lógica de negocio.

Ejecutar con: streamlit run app/app.py (desde la raíz del repo)
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_all_datasets
from cleaning import clean_all_datasets
from feature_engineering import build_master_dataset

from tabs import tab_auditoria, tab_operaciones, tab_cliente, tab_ia_insights, tab_resumen_ejecutivo


st.set_page_config(page_title="TechLogistics S.A. | DSS", page_icon="📦", layout="wide")


@st.cache_data(show_spinner="Cargando y limpiando datos crudos...")
def cargar_y_limpiar():
    raw = load_all_datasets()
    limpios, reportes = clean_all_datasets(raw)
    return raw, limpios, reportes


@st.cache_data(show_spinner="Integrando datasets y calculando variables derivadas...")
def construir_maestro(_datasets_limpios: dict) -> pd.DataFrame:
    return build_master_dataset(
        _datasets_limpios["inventario"],
        _datasets_limpios["transacciones"],
        _datasets_limpios["feedback"],
    )


def aplicar_filtros(df, fecha_rango, categorias, bodegas, ciudades):
    """Aplica los filtros del sidebar sobre el dataset maestro. Solo filtra, no transforma."""
    df_filtrado = df.copy()

    if fecha_rango and len(fecha_rango) == 2 and "Fecha_Venta" in df_filtrado.columns:
        inicio, fin = pd.Timestamp(fecha_rango[0]), pd.Timestamp(fecha_rango[1])
        df_filtrado = df_filtrado[
            df_filtrado["Fecha_Venta"].between(inicio, fin) | df_filtrado["Fecha_Venta"].isna()
        ]

    if categorias and "Categoria" in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado["Categoria"].isin(categorias)]

    if bodegas and "Bodega_Origen" in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado["Bodega_Origen"].isin(bodegas)]

    if ciudades and "Ciudad_Destino" in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado["Ciudad_Destino"].isin(ciudades)]

    return df_filtrado


# Carga inicial
datasets_crudos, datasets_limpios, reportes_limpieza = cargar_y_limpiar()
df_maestro = construir_maestro(datasets_limpios)


# Sidebar: filtros globales + botón de refrescar
with st.sidebar:
    st.title("📦 TechLogistics S.A.")
    st.caption("Sistema de Soporte a la Decisión — Consultoría Senior")

    if st.button("🔄 Refrescar Análisis", width='stretch'):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("Filtros globales")

    fecha_min = df_maestro["Fecha_Venta"].min()
    fecha_max = df_maestro["Fecha_Venta"].max()
    fecha_rango = st.date_input(
        "Rango de fechas", value=(fecha_min, fecha_max),
        min_value=fecha_min, max_value=fecha_max,
    )

    categorias_sel = st.multiselect(
        "Categoría", options=sorted(df_maestro["Categoria"].dropna().unique()),
    )
    bodegas_sel = st.multiselect(
        "Bodega de origen", options=sorted(df_maestro["Bodega_Origen"].dropna().unique()),
    )
    ciudades_sel = st.multiselect(
        "Ciudad destino", options=sorted(df_maestro["Ciudad_Destino"].dropna().unique()),
    )

    st.divider()
    st.caption(f"Dataset maestro: {len(df_maestro):,} transacciones totales")


df_filtrado = aplicar_filtros(df_maestro, fecha_rango, categorias_sel, bodegas_sel, ciudades_sel)


# Encabezado + tabs
st.title("TechLogistics S.A. — Sistema de Soporte a la Decisión")
st.caption(
    f"Mostrando **{len(df_filtrado):,}** de **{len(df_maestro):,}** transacciones "
    "según los filtros aplicados."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Auditoría", "📦 Operaciones", "😊 Cliente", "🤖 Insights de IA", "📋 Análisis Final",
])

with tab1:
    tab_auditoria.render(datasets_crudos, datasets_limpios, reportes_limpieza)

with tab2:
    tab_operaciones.render(df_filtrado)

with tab3:
    tab_cliente.render(df_filtrado)

with tab4:
    tab_ia_insights.render(df_filtrado)

with tab5:
    tab_resumen_ejecutivo.render(df_filtrado, datasets_crudos, reportes_limpieza)