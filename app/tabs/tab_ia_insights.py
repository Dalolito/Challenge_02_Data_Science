"""
Pestaña "Insights de IA": dispara src/ai_insights.py con el resumen de los
datos filtrados. La API key se lee de st.secrets["GROQ_API_KEY"], nunca hardcodeada.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from ai_insights import build_summary_stats, get_ai_recommendation


def render(df_filtrado):
    st.header("🤖 Insights de IA")
    st.caption(
        "Genera una recomendación estratégica con Llama-3 (Groq), basada "
        "ÚNICAMENTE en los datos que están filtrados ahora mismo en el sidebar."
    )

    if df_filtrado is None or df_filtrado.empty:
        st.warning("No hay datos para analizar con los filtros actuales. Ajusta el sidebar.")
        return

    resumen = build_summary_stats(df_filtrado)

    with st.expander("Ver el resumen estadístico que se le enviaría al modelo"):
        st.json(resumen)

    # .get() lanza error si secrets.toml no existe, por eso el try/except
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        api_key = None

    if not api_key or api_key == "tu_api_key_aqui":
        st.error(
            "No se encontró una GROQ_API_KEY válida en `.streamlit/secrets.toml`. "
            "Agrega tu key real para poder usar esta pestaña (no se sube al repositorio, "
            "ese archivo está en `.gitignore`)."
        )
        return

    if st.button("✨ Generar recomendación estratégica", type="primary"):
        with st.spinner("Consultando a Llama-3..."):
            try:
                recomendacion = get_ai_recommendation(resumen, api_key)
                st.markdown("### Recomendación")
                st.markdown(recomendacion)
                st.caption(
                    f"Generado sobre {resumen.get('n_transacciones', 0):,} transacciones "
                    "filtradas actualmente."
                )
            except Exception as e:
                st.error(f"No se pudo generar la recomendación: {e}")
                st.caption(
                    "Verifica que la GROQ_API_KEY sea válida y que el paquete `groq` "
                    "esté instalado (`pip install groq`)."
                )