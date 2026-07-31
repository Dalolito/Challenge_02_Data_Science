"""
tab_auditoria.py
-----------------
Pestaña "Auditoría": Módulo de Transparencia del dashboard.

Muestra, para cada dataset:
1. Health Score antes/después de limpieza.
2. Qué errores se encontraron y cómo se corrigieron (usa los reportes
   que devuelve src/cleaning.py — no recalcula nada).
3. Comparación ANTES vs DESPUÉS con ejemplos reales de filas.
4. Registros marcados/excluidos (flags: SKU_Fantasma, Ciudad_Invalida, etc.)
   con opción de verlos y descargarlos.

Se llama desde app.py así:
    from app.tabs import tab_auditoria
    tab_auditoria.render(datasets_crudos, datasets_limpios, reportes)
"""

import io
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _health_score(df: pd.DataFrame) -> float:
    """Mismo cálculo usado en el notebook de exploración: completitud + unicidad."""
    if df is None or len(df) == 0:
        return 0.0
    completitud = 1 - df.isna().mean().mean()
    unicidad = 1 - (df.duplicated().sum() / len(df))
    return round(100 * (completitud + unicidad) / 2, 1)


def _resumen_health_scores(datasets_crudos: dict, datasets_limpios: dict) -> pd.DataFrame:
    filas = []
    for nombre in datasets_crudos:
        filas.append({
            "Dataset": nombre.capitalize(),
            "Health Score (antes)": _health_score(datasets_crudos[nombre]),
            "Health Score (después)": _health_score(datasets_limpios[nombre]),
        })
    df = pd.DataFrame(filas)
    df["Mejora (pts)"] = (df["Health Score (después)"] - df["Health Score (antes)"]).round(1)
    return df


def _descargar_reporte_csv(reportes: list) -> bytes:
    """Aplana la lista de reportes de cleaning.py a un CSV descargable."""
    filas = []
    for rep in reportes:
        for cambio in rep.get("cambios", []):
            filas.append({
                "dataset": rep["dataset"],
                "columna": cambio["columna"],
                "identificacion": cambio["identificacion"],
                "decision": cambio["decision"],
                "justificacion": cambio["justificacion"],
            })
    df = pd.DataFrame(filas)
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig")


# ---------------------------------------------------------------------------
# Render principal
# ---------------------------------------------------------------------------

def render(datasets_crudos: dict, datasets_limpios: dict, reportes: list):
    st.header("🔍 Auditoría de Calidad — Módulo de Transparencia")
    st.caption(
        "Todo lo que se muestra aquí sale directamente de `src/cleaning.py`. "
        "No se recalcula nada en la interfaz: si algo cambia en la lógica de limpieza, "
        "esta pestaña se actualiza sola."
    )

    # -----------------------------------------------------------------
    # 1. Health Score antes/después
    # -----------------------------------------------------------------
    st.subheader("1. Health Score por dataset")
    resumen = _resumen_health_scores(datasets_crudos, datasets_limpios)

    col_tabla, col_grafico = st.columns([1, 1.4])
    with col_tabla:
        st.dataframe(resumen, hide_index=True, width='stretch')
    with col_grafico:
        chart_df = resumen.melt(
            id_vars="Dataset",
            value_vars=["Health Score (antes)", "Health Score (después)"],
            var_name="Momento", value_name="Score",
        )
        st.bar_chart(chart_df, x="Dataset", y="Score", color="Momento", stack=False)

    st.divider()

    # -----------------------------------------------------------------
    # 2. Qué se encontró y cómo se corrigió, por dataset
    # -----------------------------------------------------------------
    st.subheader("2. Errores detectados y corrección aplicada")

    nombres_bonitos = {"inventario": "📦 Inventario", "transacciones": "🚚 Transacciones", "feedback": "😊 Feedback"}

    tabs_dataset = st.tabs([nombres_bonitos.get(r["dataset"], r["dataset"]) for r in reportes])

    for tab, rep in zip(tabs_dataset, reportes):
        with tab:
            nulos_antes = sum(rep["nulos_antes"].values())
            nulos_despues = sum(rep["nulos_despues"].values())
            c1, c2, c3 = st.columns(3)
            c1.metric("Nulos totales — antes", f"{nulos_antes:,}")
            c2.metric("Nulos totales — después", f"{nulos_despues:,}",
                       delta=int(nulos_despues - nulos_antes), delta_color="inverse")
            c3.metric("Correcciones aplicadas", len(rep["cambios"]))

            if not rep["cambios"]:
                st.info("No se registraron cambios para este dataset.")
                continue

            for cambio in rep["cambios"]:
                with st.expander(f"**{cambio['columna']}**"):
                    st.markdown(f"🔎 **Cómo se identificó**")
                    st.write(cambio["identificacion"])
                    st.markdown(f"🛠️ **Qué se decidió**")
                    st.write(cambio["decision"])
                    st.markdown(f"💡 **Por qué**")
                    st.write(cambio["justificacion"])

    st.divider()

    # -----------------------------------------------------------------
    # 3. Antes vs Después — muestra de filas reales
    # -----------------------------------------------------------------
    st.subheader("3. Antes vs Después — ejemplo de filas")
    st.caption("Selecciona un dataset y una columna para comparar los valores crudos contra los ya limpios.")

    dataset_sel = st.selectbox(
        "Dataset", options=list(datasets_crudos.keys()),
        format_func=lambda k: nombres_bonitos.get(k, k), key="auditoria_dataset_sel",
    )
    df_crudo = datasets_crudos[dataset_sel]
    df_limpio = datasets_limpios[dataset_sel]

    columnas_comunes = [c for c in df_crudo.columns if c in df_limpio.columns]
    columna_sel = st.selectbox("Columna", options=columnas_comunes, key="auditoria_columna_sel")

    n_muestra = min(10, len(df_crudo))
    idx_muestra = df_crudo.sample(n=n_muestra, random_state=42).index if len(df_crudo) > 0 else []

    comparacion = pd.DataFrame({
        "Antes (crudo)": df_crudo.loc[idx_muestra, columna_sel].astype(str).values,
        "Después (limpio)": df_limpio.loc[idx_muestra, columna_sel].astype(str).values
        if idx_muestra is not None and len(idx_muestra) > 0 and all(i in df_limpio.index for i in idx_muestra)
        else ["—"] * len(idx_muestra),
    })
    st.dataframe(comparacion, width='stretch')

    st.divider()

    # -----------------------------------------------------------------
    # 4. Registros marcados / excluidos (flags de transparencia)
    # -----------------------------------------------------------------
    st.subheader("4. Ver registros excluidos / marcados")
    st.caption(
        "Estas columnas de flag las crea `cleaning.py` para casos que NO se pueden "
        "corregir automáticamente (requieren una decisión de negocio)."
    )

    flags_conocidos = {
        "transacciones": {
            "SKU_Fantasma": "Ventas cuyo SKU no existe en el inventario",
            "Ciudad_Invalida": "Ciudad_Destino contaminada con datos de Canal_Venta",
            "Cantidad_Corregida": "Cantidad_Vendida que era negativa y se corrigió",
        },
        "feedback": {},
        "inventario": {},
    }

    hubo_flags = False
    for nombre_ds, flags in flags_conocidos.items():
        df_ds = datasets_limpios.get(nombre_ds)
        if df_ds is None:
            continue
        for col_flag, descripcion in flags.items():
            if col_flag not in df_ds.columns:
                continue
            hubo_flags = True
            n_marcados = int(df_ds[col_flag].sum())
            pct = 100 * n_marcados / len(df_ds) if len(df_ds) else 0
            with st.expander(f"{nombres_bonitos.get(nombre_ds, nombre_ds)} — **{col_flag}**: "
                              f"{n_marcados:,} registros ({pct:.1f}%) — {descripcion}"):
                st.write(descripcion)
                df_marcados = df_ds[df_ds[col_flag]]
                st.dataframe(df_marcados.head(50), width='stretch')
                st.download_button(
                    f"⬇️ Descargar los {n_marcados:,} registros marcados",
                    data=df_marcados.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{nombre_ds}_{col_flag}.csv",
                    mime="text/csv",
                    key=f"download_{nombre_ds}_{col_flag}",
                )

    if not hubo_flags:
        st.info("No hay columnas de flag disponibles todavía en los datasets limpios.")

    st.divider()

    # -----------------------------------------------------------------
    # 5. Descarga del reporte completo de limpieza
    # -----------------------------------------------------------------
    st.subheader("5. Reporte completo de limpieza")
    st.download_button(
        "⬇️ Descargar reporte de limpieza (CSV)",
        data=_descargar_reporte_csv(reportes),
        file_name="reporte_limpieza.csv",
        mime="text/csv",
    )