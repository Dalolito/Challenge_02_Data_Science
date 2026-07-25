"""
feature_engineering.py
------------------------
Integración (merge) de los 3 datasets limpios y creación de variables
derivadas (Fase 2 del challenge).

Supuestos de negocio documentados explícitamente:

1. SLA_DIAS_PROMETIDO = 15
   No existe una columna de "fecha prometida de entrega". Se usa como
   referencia la mediana real de Tiempo_Entrega_Real observada en los
   datos limpios (~15 días) como proxy del SLA esperado. Es un parámetro
   configurable.

2. Margen_Utilidad = NaN para ventas con SKU_Fantasma=True (no hay
   Costo_Unitario_USD porque el producto no existe en inventario). No se
   imputa ni se excluye la fila — es información en sí misma para la
   Pregunta 3 del reto (impacto financiero de la venta invisible).

3. Feedback se AGREGA a nivel de Transaccion_ID antes del merge.
   877 transacciones tienen entre 2 y 4 registros de feedback distintos
   asociados (no son duplicados, son opiniones distintas de clientes).
   Si se hiciera un merge directo 1-a-muchos, esas ventas se contarían
   2, 3 o 4 veces en cualquier suma de ingresos/margen — corrompiendo
   la Pregunta 1 (fuga de capital). Por eso el dataset maestro queda a
   nivel de TRANSACCIÓN (una fila = una venta, 10,000 filas), y el
   feedback múltiple se resume con la columna 'Feedback_Multiple_Count'
   para no perder la señal de que hubo más de una opinión.
"""

import numpy as np
import pandas as pd

SLA_DIAS_PROMETIDO = 15  # ver supuesto #1 arriba


# ---------------------------------------------------------------------------
# Agregación de feedback (evita fan-out en el merge — ver supuesto #3)
# ---------------------------------------------------------------------------

def _aggregate_feedback_por_transaccion(df_feedback: pd.DataFrame) -> pd.DataFrame:
    """
    Colapsa el feedback a UNA fila por Transaccion_ID.

    Reglas de agregación:
    - Rating_Producto, Rating_Logistica, Satisfaccion_NPS, Edad_Cliente: promedio.
    - Ticket_Soporte_Abierto: máximo (si CUALQUIER feedback de esa transacción
      abrió un ticket, se marca 1 — es peor caso, más conservador para el
      negocio que promediar).
    - Recomienda_Marca: la moda (valor más frecuente); si hay empate, el primero.
    - Comentario_Texto: se concatenan los comentarios no nulos con ' | '.
    - Feedback_Multiple_Count: cuántos registros de feedback se agregaron
      (1 = caso normal, >1 = flag de transparencia).
    """
    if df_feedback is None or len(df_feedback) == 0:
        return df_feedback

    def _moda_segura(serie):
        serie_valida = serie.dropna()
        if serie_valida.empty:
            return np.nan
        return serie_valida.mode().iloc[0]

    def _concat_comentarios(serie):
        comentarios = serie.dropna().astype(str)
        comentarios = comentarios[comentarios.str.strip() != ""]
        return " | ".join(comentarios) if len(comentarios) > 0 else np.nan

    agg = df_feedback.groupby("Transaccion_ID").agg(
        Rating_Producto=("Rating_Producto", "mean"),
        Rating_Logistica=("Rating_Logistica", "mean"),
        Satisfaccion_NPS=("Satisfaccion_NPS", "mean"),
        Edad_Cliente=("Edad_Cliente", "mean"),
        Ticket_Soporte_Abierto=("Ticket_Soporte_Abierto", "max"),
        Recomienda_Marca=("Recomienda_Marca", _moda_segura),
        Comentario_Texto=("Comentario_Texto", _concat_comentarios),
        Feedback_Multiple_Count=("Transaccion_ID", "size"),
    ).reset_index()

    return agg


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_datasets(
    df_inventario: pd.DataFrame,
    df_transacciones: pd.DataFrame,
    df_feedback: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une los 3 datasets limpios en una sola tabla maestra A NIVEL DE TRANSACCIÓN
    (una fila = una venta, sin duplicar filas por múltiples feedback).

    - transacciones <- inventario           por SKU_ID (left join: se
      conservan TODAS las ventas, incluidas las de SKU fantasma).
    - resultado      <- feedback agregado    por Transaccion_ID (left join).

    Parameters
    ----------
    df_inventario, df_transacciones, df_feedback : DataFrames YA LIMPIOS
        (salida de cleaning.clean_all_datasets()).

    Returns
    -------
    pd.DataFrame con exactamente len(df_transacciones) filas.
    """
    n_transacciones_esperadas = len(df_transacciones)

    df = df_transacciones.merge(
        df_inventario, on="SKU_ID", how="left", suffixes=("", "_inv"),
    )

    feedback_agregado = _aggregate_feedback_por_transaccion(df_feedback)
    df = df.merge(
        feedback_agregado, on="Transaccion_ID", how="left", suffixes=("", "_fb"),
    )
    df["Feedback_Multiple_Count"] = df["Feedback_Multiple_Count"].fillna(0).astype(int)

    assert len(df) == n_transacciones_esperadas, (
        f"El merge cambió el número de filas: {n_transacciones_esperadas} transacciones "
        f"esperadas, {len(df)} obtenidas. Revisar duplicados en las llaves de join."
    )

    return df


# ---------------------------------------------------------------------------
# Variables derivadas
# ---------------------------------------------------------------------------

def _add_margen_utilidad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Margen_Utilidad = Precio_Venta_Final - Costo_Unitario_USD - Costo_Envio
    Margen_Utilidad_Pct = Margen_Utilidad / Precio_Venta_Final * 100
    NaN para ventas con SKU_Fantasma=True (no hay costo de referencia).
    """
    tiene_costo = df["Costo_Unitario_USD"].notna()

    df["Margen_Utilidad"] = np.where(
        tiene_costo,
        df["Precio_Venta_Final"] - df["Costo_Unitario_USD"] - df["Costo_Envio"].fillna(0),
        np.nan,
    )
    df["Margen_Utilidad_Pct"] = np.where(
        tiene_costo & (df["Precio_Venta_Final"] != 0),
        100 * df["Margen_Utilidad"] / df["Precio_Venta_Final"],
        np.nan,
    )
    return df


def _add_brecha_entrega(df: pd.DataFrame, sla_dias: int = SLA_DIAS_PROMETIDO) -> pd.DataFrame:
    """
    Brecha_Entrega = Tiempo_Entrega_Real - SLA_dias_prometido.
    Positivo = entrega tardía respecto al SLA. Negativo = entrega adelantada.
    """
    df["Brecha_Entrega"] = df["Tiempo_Entrega_Real"] - sla_dias
    df["Entrega_Fuera_SLA"] = df["Brecha_Entrega"] > 0
    return df


def _add_ratio_soporte_categoria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio_Soporte_Categoria = % de tickets de soporte abiertos dentro de la
    misma Categoria de producto (tasa agregada por categoría, no por fila
    individual — sirve para comparar categorías entre sí).
    Filas sin Categoria (SKU fantasma) quedan con NaN.
    """
    if "Ticket_Soporte_Abierto" not in df.columns or "Categoria" not in df.columns:
        return df
    ratio_por_categoria = df.groupby("Categoria")["Ticket_Soporte_Abierto"].mean() * 100
    df["Ratio_Soporte_Categoria"] = df["Categoria"].map(ratio_por_categoria)
    return df


def add_derived_features(df: pd.DataFrame, sla_dias: int = SLA_DIAS_PROMETIDO) -> pd.DataFrame:
    """Aplica las 3 variables derivadas requeridas por el reto."""
    df = df.copy()
    df = _add_margen_utilidad(df)
    df = _add_brecha_entrega(df, sla_dias=sla_dias)
    df = _add_ratio_soporte_categoria(df)
    return df


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def build_master_dataset(
    df_inventario: pd.DataFrame,
    df_transacciones: pd.DataFrame,
    df_feedback: pd.DataFrame,
    sla_dias: int = SLA_DIAS_PROMETIDO,
) -> pd.DataFrame:
    """Punto de entrada único: merge (sin fan-out) + variables derivadas."""
    df_master = merge_datasets(df_inventario, df_transacciones, df_feedback)
    df_master = add_derived_features(df_master, sla_dias=sla_dias)
    return df_master


if __name__ == "__main__":
    from data_loader import load_all_datasets
    from cleaning import clean_all_datasets

    raw = load_all_datasets()
    cleaned, _ = clean_all_datasets(raw)

    df_master = build_master_dataset(
        cleaned["inventario"], cleaned["transacciones"], cleaned["feedback"]
    )

    print(f"Dataset maestro: {df_master.shape[0]:,} filas x {df_master.shape[1]} columnas")
    print(f"(debe ser igual al número de transacciones: {len(cleaned['transacciones']):,})")

    print(f"\nTransacciones con más de 1 feedback asociado: "
          f"{(df_master['Feedback_Multiple_Count'] > 1).sum():,}")

    print(f"\nMargen_Utilidad -> NaN (SKU fantasma): {df_master['Margen_Utilidad'].isna().sum():,} "
          f"de {len(df_master):,} filas")
    print(f"Margen_Utilidad promedio (solo válidos): {df_master['Margen_Utilidad'].mean():.2f} USD")
    print(f"Ventas con margen NEGATIVO: {(df_master['Margen_Utilidad'] < 0).sum():,}")

    print(f"\nEntregas fuera de SLA ({SLA_DIAS_PROMETIDO} días): "
          f"{df_master['Entrega_Fuera_SLA'].sum():,} ({100*df_master['Entrega_Fuera_SLA'].mean():.1f}%)")

    print("\nRatio_Soporte_Categoria por categoría:")
    print(df_master.groupby("Categoria")["Ratio_Soporte_Categoria"].first().sort_values(ascending=False))