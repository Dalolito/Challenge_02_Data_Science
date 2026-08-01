"""
ui_helpers.py
-------------
Componentes visuales compartidos entre las pestañas del dashboard, para no
repetir el mismo CSS/lógica en cada archivo de app/tabs/.
"""

import textwrap

import altair as alt
import pandas as pd
import streamlit as st


def render_html_block(*partes: str):
    """
    Renderiza uno o más fragmentos HTML/CSS como un solo st.markdown().

    Streamlit le aplica textwrap.dedent() a TODO el string final antes de
    mandarlo al navegador, pero dedent solo puede quitar la indentación
    mínima COMÚN a todas las líneas. Si se concatenan fragmentos con
    distinta indentación de origen (ej. una constante CSS definida sin
    indentar + un f-string escrito con 8 espacios dentro de una función),
    el mínimo común baja a 0 y el fragmento indentado queda con espacios
    de sobra -> Markdown lo interpreta como bloque de código.

    Por eso cada parte se dedenta POR SEPARADO aquí, antes de unirlas.
    """
    html = "\n".join(textwrap.dedent(p).strip("\n") for p in partes)
    st.markdown(html, unsafe_allow_html=True)


def titulo_seccion(texto: str):
    """Renderiza un título de sección como badge azul claro y translúcido."""
    st.markdown(
        f"""
        <div style="
            background-color:rgba(96, 165, 250, 0.15);
            border:1px solid rgba(96, 165, 250, 0.35);
            display:inline-block;
            padding:8px 18px;
            border-radius:8px;
            margin:18px 0 10px 0;
        ">
            <span style="color:#93c5fd; font-weight:700; font-size:1.7rem; text-decoration:none;">{texto}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


BADGE_INLINE_CSS = """<style>
.badge-inline {
    background-color:rgba(96, 165, 250, 0.15);
    border:1px solid rgba(96, 165, 250, 0.35);
    color:#93c5fd;
    font-weight:700;
    padding:1px 8px;
    border-radius:6px;
    text-decoration:none;
    white-space:nowrap;
}
</style>"""


def badge_inline(texto: str) -> str:
    """Devuelve un <span> con el mismo estilo de badge azul, para usar
    inline dentro de un bloque de texto (en vez de <b>)."""
    return f'<span class="badge-inline">{texto}</span>'


def metric_box(col, etiqueta: str, valor: str, delta: str = None, delta_es_bueno: bool = None):
    """
    Renderiza una métrica dentro de un cuadro amarillo claro y translúcido,
    en vez del st.metric() nativo.

    delta_es_bueno: True -> delta en verde, False -> delta en rojo,
    None -> delta en gris neutro (informativo, sin implicar bien/mal).
    """
    delta_html = ""
    if delta is not None:
        if delta_es_bueno is True:
            color_delta = "#4ade80"
        elif delta_es_bueno is False:
            color_delta = "#f87171"
        else:
            color_delta = "#cbd5e1"
        delta_html = (
            f'<div style="color:{color_delta}; font-size:0.95rem; margin-top:4px;">{delta}</div>'
        )
    with col:
        st.markdown(
            f"""
            <div style="
                background-color:rgba(250, 220, 96, 0.15);
                border:1px solid rgba(250, 220, 96, 0.35);
                border-radius:8px;
                padding:12px 18px;
                margin-bottom:8px;
            ">
                <div style="color:#fde68a; font-size:0.9rem;">{etiqueta}</div>
                <div style="color:#ffffff; font-weight:700; font-size:1.8rem;">{valor}</div>
                {delta_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
def download_button_verde(label: str, data, file_name: str, mime: str, key: str):
    """Botón de descarga centrado, en verde claro translúcido, sin emoji."""
    st.markdown(
        f"""
        <style>
        div.st-key-{key} button {{
            background-color: rgba(74, 222, 128, 0.25);
            border: 1px solid rgba(74, 222, 128, 0.5);
        }}
        div.st-key-{key} button p {{
            color: #ffffff;
            font-weight: 700;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    _, col_boton, _ = st.columns([1, 1, 1])
    with col_boton:
        st.download_button(
            label, data=data, file_name=file_name, mime=mime,
            width="stretch", key=key,
        )
def bar_chart_fijo(serie: pd.Series, height: int = 350, color: str = "#60a5fa"):
    """
    Grafico de barras a partir de una Serie (index=categoría, values=número),
    construido con Altair SIN .interactive() — queda completamente fijo,
    sin zoom ni paneo con la rueda del mouse (a diferencia de st.bar_chart,
    que trae esa interacción activada por defecto).

    Respeta el orden en el que vengan las categorías en la Serie (si ya
    viene ordenada con .sort_values(), el gráfico mantiene ese orden en
    vez de reordenar alfabéticamente).
    """
    df = serie.reset_index()
    df.columns = ["Categoria", "Valor"]
    grafico = (
        alt.Chart(df)
        .mark_bar(color=color)
        .encode(
            x=alt.X("Categoria:N", sort=None, title=serie.index.name or "Categoría"),
            y=alt.Y("Valor:Q", title=serie.name or "Valor"),
            tooltip=["Categoria", "Valor"],
        )
        .properties(height=height)
    )
    st.altair_chart(grafico, width="stretch")