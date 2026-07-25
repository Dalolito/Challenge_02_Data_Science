"""Merge + variables derivadas (Fase 2)."""

import numpy as np
import pandas as pd

SLA_DIAS_PROMETIDO = 15


def _aggregate_feedback_por_transaccion(df_feedback: pd.DataFrame) -> pd.DataFrame:
    """Colapsa feedback a 1 fila por Transaccion_ID."""
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


def merge_datasets(
    df_inventario: pd.DataFrame,
    df_transacciones: pd.DataFrame,
    df_feedback: pd.DataFrame,
) -> pd.DataFrame:
    """Left join: transacciones -> inventario -> feedback."""
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


# Variables derivadas

def _add_margen_utilidad(df: pd.DataFrame) -> pd.DataFrame:
    """Margen = Precio - Costo - Envío. NaN para SKU fantasma."""
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
    """Brecha = real - SLA; positivo = tarde."""
    df["Brecha_Entrega"] = df["Tiempo_Entrega_Real"] - sla_dias
    df["Entrega_Fuera_SLA"] = df["Brecha_Entrega"] > 0
    return df


def _add_ratio_soporte_categoria(df: pd.DataFrame) -> pd.DataFrame:
    """% tickets soporte por categoría."""
    if "Ticket_Soporte_Abierto" not in df.columns or "Categoria" not in df.columns:
        return df
    ratio_por_categoria = df.groupby("Categoria")["Ticket_Soporte_Abierto"].mean() * 100
    df["Ratio_Soporte_Categoria"] = df["Categoria"].map(ratio_por_categoria)
    return df


def add_derived_features(df: pd.DataFrame, sla_dias: int = SLA_DIAS_PROMETIDO) -> pd.DataFrame:
    """Aplica las 3 variables derivadas del reto."""
    df = df.copy()
    df = _add_margen_utilidad(df)
    df = _add_brecha_entrega(df, sla_dias=sla_dias)
    df = _add_ratio_soporte_categoria(df)
    return df


def build_master_dataset(
    df_inventario: pd.DataFrame,
    df_transacciones: pd.DataFrame,
    df_feedback: pd.DataFrame,
    sla_dias: int = SLA_DIAS_PROMETIDO,
) -> pd.DataFrame:
    """Punto de entrada único: merge + variables derivadas."""
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
