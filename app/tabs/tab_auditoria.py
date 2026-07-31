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
5. Descarga del reporte completo de limpieza.

La exploración multivariable (matriz de correlación) vive en la pestaña
Análisis Final, no aquí — Auditoría es estrictamente sobre calidad y
trazabilidad de la limpieza, no sobre análisis de negocio.

Se llama desde app.py así:
    from app.tabs import tab_auditoria
    tab_auditoria.render(datasets_crudos, datasets_limpios, reportes)
"""

import io
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from quality_metrics import resumen_calidad_completo


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _tabla_4_dimensiones(datasets_crudos: dict, datasets_limpios: dict) -> pd.DataFrame:
    """
    Arma la tabla comparativa antes/después para las 4 dimensiones de
    calidad. Consistencia no aplica al dataset crudo (las columnas flag
    solo existen tras limpiar), así que queda vacía ahí — no se inventa
    un número.
    """
    filas = []
    for nombre in datasets_crudos:
        antes = resumen_calidad_completo(datasets_crudos[nombre], nombre)
        despues = resumen_calidad_completo(datasets_limpios.get(nombre), nombre)
        filas.append({
            "Dataset": nombre.capitalize(),
            "Momento": "Antes",
            "Completitud": antes["completitud"],
            "Unicidad": antes["unicidad"],
            "Validez": antes["validez"],
            "Consistencia": None,
        })
        filas.append({
            "Dataset": nombre.capitalize(),
            "Momento": "Después",
            "Completitud": despues["completitud"],
            "Unicidad": despues["unicidad"],
            "Validez": despues["validez"],
            "Consistencia": despues["consistencia"],
        })
    return pd.DataFrame(filas)


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
    # 1. Calidad en 4 dimensiones (no solo nulos)
    # -----------------------------------------------------------------
    st.subheader("1. Calidad de datos — 4 dimensiones")
    st.caption(
        "Medir solo '% de nulos' esconde problemas: un Rating_Producto=99 o una "
        "Ciudad_Destino='Ventas_Web' no son nulos, son datos presentes pero inválidos "
        "o inconsistentes. Por eso se miden 4 dimensiones por separado."
    )

    with st.expander("¿Qué mide cada dimensión?"):
        st.markdown("""
- **Completitud**: % de celdas que SÍ tienen un valor (lo contrario de nulos).
- **Unicidad**: % de filas que NO son un duplicado exacto de otra.
- **Validez**: % de valores que caen dentro del rango de negocio esperado
  (ej. un Rating entre 1 y 5, una Edad entre 0 y 100). Un dato puede estar
  presente y aun así ser inválido — eso NO lo detecta la completitud.
- **Consistencia**: % de filas sin problemas de integridad referencial ni
  contaminación cruzada entre columnas (SKU sin inventario asociado, ciudad
  contaminada con datos de otra columna). Solo se puede calcular sobre el
  dataset ya limpio, porque ahí es donde existen las columnas de flag.
        """)

    tabla_dimensiones = _tabla_4_dimensiones(datasets_crudos, datasets_limpios)

    nombres_bonitos = {"inventario": "📦 Inventario", "transacciones": "🚚 Transacciones", "feedback": "😊 Feedback"}

    tabs_dimensiones = st.tabs([nombres_bonitos.get(k, k) for k in datasets_crudos.keys()])
    for tab, nombre_ds in zip(tabs_dimensiones, datasets_crudos.keys()):
        with tab:
            sub = tabla_dimensiones[tabla_dimensiones["Dataset"] == nombre_ds.capitalize()]

            col_tabla, col_grafico = st.columns([1, 1.3])
            with col_tabla:
                st.dataframe(
                    sub.drop(columns="Dataset").set_index("Momento"),
                    width="stretch",
                )
            with col_grafico:
                chart_df = sub.melt(
                    id_vars="Momento",
                    value_vars=["Completitud", "Unicidad", "Validez", "Consistencia"],
                    var_name="Dimensión", value_name="Score",
                ).dropna(subset=["Score"])
                if not chart_df.empty:
                    st.bar_chart(chart_df, x="Dimensión", y="Score", color="Momento", stack=False)

            # Detalle de validez por columna, si aplica
            despues = resumen_calidad_completo(datasets_limpios.get(nombre_ds), nombre_ds)
            if despues["consistencia_detalle"]:
                st.markdown("**Detalle de consistencia (dataset limpio):**")
                for col_flag, info in despues["consistencia_detalle"].items():
                    st.write(f"- `{col_flag}`: {info['n_marcados']:,} registros marcados "
                             f"({info['pct_marcados']}%)")

    st.divider()

    # -----------------------------------------------------------------
    # 2. Qué se encontró y cómo se corrigió, por dataset
    # -----------------------------------------------------------------
    st.subheader("2. Errores detectados y corrección aplicada")

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