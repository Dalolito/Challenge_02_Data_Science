import os
import textwrap

import altair as alt
import pandas as pd
import streamlit as st

# results/figuras/ vive 2 niveles arriba de este archivo (app/tabs/ -> raíz -> results/figuras)
_RUTA_FIGURAS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "figuras",
)


def _ruta_figura(nombre: str) -> str:
    os.makedirs(_RUTA_FIGURAS, exist_ok=True)
    return os.path.join(_RUTA_FIGURAS, f"{nombre}.png")


def guardar_altair(chart: "alt.Chart", nombre: str):
    """Exporta un gráfico de Altair a results/figuras/{nombre}.png (requiere vl-convert-python).
    Si falla (ej. no está instalado el paquete), no rompe la app — solo avisa una vez."""
    try:
        chart.save(_ruta_figura(nombre), scale_factor=2)
    except Exception as e:
        st.caption(f"⚠️ No se pudo guardar la figura '{nombre}.png': {e}")


def guardar_matplotlib(fig, nombre: str):
    """Exporta una figura de Matplotlib/Seaborn a results/figuras/{nombre}.png."""
    try:
        fig.savefig(_ruta_figura(nombre), dpi=170, bbox_inches="tight")
    except Exception as e:
        st.caption(f"⚠️ No se pudo guardar la figura '{nombre}.png': {e}")


def render_html_block(*partes: str):
    """Renderiza fragmentos HTML/CSS en un solo st.markdown(), dedentando cada
    parte por separado (si no, Streamlit puede interpretar el HTML indentado
    como bloque de código)."""
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
    """Métrica en cuadro amarillo translúcido (reemplaza st.metric()).
    delta_es_bueno: True=verde, False=rojo, None=gris neutro."""
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


def bar_chart_fijo(serie: pd.Series, height: int = 350, color: str = "#60a5fa", guardar: str = None):
    """Barras con Altair sin .interactive() (no hace zoom/paneo con el mouse,
    a diferencia de st.bar_chart). Respeta el orden de la Serie recibida.
    Si se pasa `guardar`, también exporta a results/figuras/{guardar}.png."""
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
    if guardar:
        guardar_altair(grafico, guardar)