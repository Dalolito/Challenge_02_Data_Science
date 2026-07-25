"""
quality_metrics.py
---------------------
Calcula el Health Score y métricas de calidad de un dataset, ANTES y
DESPUÉS de la limpieza. Es la única fuente de verdad para este cálculo —
el notebook de exploración y app/tabs/tab_auditoria.py importan de aquí,
en vez de recalcular el score cada uno por su lado (eso fue justo el
problema que teníamos: el mismo número calculado en 2 sitios distintos).
"""

import pandas as pd


def health_score(df: pd.DataFrame) -> float:
    """
    Health Score de un dataset, en escala 0-100.

    Combina dos componentes con el mismo peso:
    - Completitud: 1 - (% promedio de celdas nulas)
    - Unicidad:    1 - (% de filas duplicadas)

    Es una métrica simple y auditable a propósito: cualquiera en el equipo
    puede recalcularla a mano con dos líneas de pandas. No pretende ser un
    índice sofisticado, sino un número fácil de explicar en el informe de
    hallazgos.
    """
    if df is None or len(df) == 0:
        return 0.0

    completitud = 1 - df.isna().mean().mean()
    unicidad = 1 - (df.duplicated().sum() / len(df))
    return round(100 * (completitud + unicidad) / 2, 1)


def nulidad_por_columna(df: pd.DataFrame) -> pd.Series:
    """% de nulos por columna, ordenado de mayor a menor."""
    return (df.isna().mean() * 100).round(2).sort_values(ascending=False)


def resumen_calidad(df: pd.DataFrame, nombre: str) -> dict:
    """
    Resumen de una sola fila con las métricas clave de un dataset,
    pensado para construir tablas comparativas (ej. antes vs después).
    """
    return {
        "dataset": nombre,
        "filas": len(df),
        "columnas": df.shape[1],
        "completitud_%": round(100 * (1 - df.isna().mean().mean()), 1),
        "duplicados": int(df.duplicated().sum()),
        "health_score": health_score(df),
    }


def comparar_antes_despues(
    datasets_crudos: dict, datasets_limpios: dict
) -> pd.DataFrame:
    """
    Tabla comparativa de Health Score antes/después para varios datasets.

    Parameters
    ----------
    datasets_crudos, datasets_limpios : dict[str, pd.DataFrame]
        Mismas claves en ambos diccionarios (ej. 'inventario', 'transacciones', 'feedback').

    Returns
    -------
    DataFrame con columnas: dataset, health_score_antes, health_score_despues, mejora_pts
    """
    filas = []
    for nombre in datasets_crudos:
        antes = health_score(datasets_crudos[nombre])
        despues = health_score(datasets_limpios.get(nombre))
        filas.append({
            "dataset": nombre,
            "health_score_antes": antes,
            "health_score_despues": despues,
            "mejora_pts": round(despues - antes, 1),
        })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    from data_loader import load_all_datasets
    from cleaning import clean_all_datasets

    raw = load_all_datasets()
    cleaned, _ = clean_all_datasets(raw)

    print("=== Health Score ANTES de limpieza ===")
    for nombre, df in raw.items():
        print(f"  {nombre}: {health_score(df)}")

    print("\n=== Health Score DESPUÉS de limpieza ===")
    for nombre, df in cleaned.items():
        print(f"  {nombre}: {health_score(df)}")

    print("\n=== Tabla comparativa ===")
    print(comparar_antes_despues(raw, cleaned).to_string(index=False))