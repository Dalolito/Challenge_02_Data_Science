"""Health Score y métricas de calidad pre/post limpieza."""

import pandas as pd


def health_score(df: pd.DataFrame) -> float:
    """Score 0-100 basado en completitud y unicidad."""
    if df is None or len(df) == 0:
        return 0.0

    completitud = 1 - df.isna().mean().mean()
    unicidad = 1 - (df.duplicated().sum() / len(df))
    return round(100 * (completitud + unicidad) / 2, 1)


def nulidad_por_columna(df: pd.DataFrame) -> pd.Series:
    """% nulos por columna, descendente."""
    return (df.isna().mean() * 100).round(2).sort_values(ascending=False)


def resumen_calidad(df: pd.DataFrame, nombre: str) -> dict:
    """Resumen de métricas clave de un dataset."""
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
    """Tabla comparativa Health Score antes/después."""
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
