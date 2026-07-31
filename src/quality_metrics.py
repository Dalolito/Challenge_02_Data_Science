"""
quality_metrics.py
---------------------
Calcula la calidad de un dataset en 4 dimensiones separadas, no un solo
número combinado. La razón: un dataset puede tener 0% de nulos y seguir
lleno de basura (ej. Rating_Producto=99 no es un nulo, Ciudad_Destino=
'Ventas_Web' no es un nulo) — medir solo completitud esconde esos casos.

Dimensiones:
- Completitud: 1 - (% promedio de celdas nulas)
- Unicidad:    1 - (% de filas duplicadas)
- Validez:     % de valores que caen DENTRO del rango de negocio esperado
               (ej. Rating 1-5, Edad 0-100, Stock >= 0) — un dato presente
               pero inválido no lo detecta la completitud.
- Consistencia: % de filas SIN problemas de integridad referencial o
               contaminación cruzada entre columnas (SKU_Fantasma,
               Ciudad_Invalida, etc. — usa las columnas flag que ya
               genera cleaning.py).

Este módulo es la única fuente de verdad para estos cálculos: el notebook
y app/tabs/tab_auditoria.py importan de aquí.
"""

import pandas as pd

# Reglas de validez por dataset: columna -> (valor_min, valor_max) o
# función de validación. Se aplican sobre el dataset CRUDO (antes de
# limpiar) para medir qué tan "sucio" venía, y también se pueden aplicar
# sobre el limpio para confirmar que la corrección funcionó.
REGLAS_VALIDEZ = {
    "inventario": {
        "Stock_Actual": (0, None),           # no puede ser negativo
        "Costo_Unitario_USD": (0.01, 100_000),  # rango de negocio razonable
    },
    "transacciones": {
        "Cantidad_Vendida": (0, None),
        "Tiempo_Entrega_Real": (0, 365),     # 999 y outliers absurdos quedan fuera
    },
    "feedback": {
        "Rating_Producto": (1, 5),
        "Rating_Logistica": (1, 5),
        "Edad_Cliente": (0, 100),
    },
}

# Columnas flag de consistencia que genera cleaning.py, por dataset.
FLAGS_CONSISTENCIA = {
    "transacciones": ["SKU_Fantasma", "Ciudad_Invalida"],
}


# ---------------------------------------------------------------------------
# Completitud y unicidad (igual que antes)
# ---------------------------------------------------------------------------

def score_completitud(df: pd.DataFrame) -> float:
    """% de celdas NO nulas, promedio sobre todas las columnas. 0-100."""
    if df is None or len(df) == 0:
        return 0.0
    return round(100 * (1 - df.isna().mean().mean()), 1)


def score_unicidad(df: pd.DataFrame) -> float:
    """% de filas que NO son duplicado exacto de otra. 0-100."""
    if df is None or len(df) == 0:
        return 0.0
    return round(100 * (1 - df.duplicated().sum() / len(df)), 1)


# ---------------------------------------------------------------------------
# Validez
# ---------------------------------------------------------------------------

def score_validez(df: pd.DataFrame, nombre_dataset: str) -> dict:
    """
    % de valores dentro del rango de negocio esperado, por columna y
    promedio general. Un valor NaN no cuenta como inválido aquí (eso ya
    lo mide completitud) — solo cuenta un valor PRESENTE pero fuera de rango.

    Returns
    -------
    dict con 'score_validez' (0-100, promedio) y 'detalle' (por columna).
    """
    reglas = REGLAS_VALIDEZ.get(nombre_dataset, {})
    if not reglas or df is None or len(df) == 0:
        return {"score_validez": None, "detalle": {}}

    detalle = {}
    for columna, (minimo, maximo) in reglas.items():
        if columna not in df.columns:
            continue
        serie = pd.to_numeric(df[columna], errors="coerce")
        serie_presente = serie.dropna()
        if serie_presente.empty:
            continue

        valido = serie_presente.between(
            minimo if minimo is not None else -float("inf"),
            maximo if maximo is not None else float("inf"),
        )
        pct_valido = round(100 * valido.mean(), 1)
        detalle[columna] = {
            "pct_valido": pct_valido,
            "n_invalidos": int((~valido).sum()),
            "n_evaluados": len(serie_presente),
        }

    if not detalle:
        return {"score_validez": None, "detalle": {}}

    score_promedio = round(sum(d["pct_valido"] for d in detalle.values()) / len(detalle), 1)
    return {"score_validez": score_promedio, "detalle": detalle}


# ---------------------------------------------------------------------------
# Consistencia
# ---------------------------------------------------------------------------

def score_consistencia(df: pd.DataFrame, nombre_dataset: str) -> dict:
    """
    % de filas SIN ningún problema de integridad referencial o
    contaminación cruzada detectado (usa las columnas flag que crea
    cleaning.py: SKU_Fantasma, Ciudad_Invalida, etc.)

    Solo tiene sentido calcularlo sobre el dataset YA LIMPIO, porque las
    columnas flag no existen en el crudo.
    """
    columnas_flag = FLAGS_CONSISTENCIA.get(nombre_dataset, [])
    columnas_presentes = [c for c in columnas_flag if c in df.columns]

    if not columnas_presentes or df is None or len(df) == 0:
        return {"score_consistencia": None, "detalle": {}}

    detalle = {}
    fila_tiene_problema = pd.Series(False, index=df.index)
    for col in columnas_presentes:
        n_marcados = int(df[col].sum())
        pct_marcados = round(100 * n_marcados / len(df), 1)
        detalle[col] = {"n_marcados": n_marcados, "pct_marcados": pct_marcados}
        fila_tiene_problema = fila_tiene_problema | df[col].fillna(False)

    score = round(100 * (1 - fila_tiene_problema.mean()), 1)
    return {"score_consistencia": score, "detalle": detalle}


# ---------------------------------------------------------------------------
# Resumen combinado — las 4 dimensiones juntas
# ---------------------------------------------------------------------------

def resumen_calidad_completo(df: pd.DataFrame, nombre_dataset: str) -> dict:
    """
    Calcula las 4 dimensiones para un dataset y las devuelve en un solo dict,
    listo para tabular. Si una dimensión no aplica (ej. validez sin reglas
    definidas, o consistencia sin columnas flag), queda como None — la tab
    debe manejar ese caso, no inventar un número.
    """
    validez = score_validez(df, nombre_dataset)
    consistencia = score_consistencia(df, nombre_dataset)

    return {
        "dataset": nombre_dataset,
        "filas": len(df) if df is not None else 0,
        "completitud": score_completitud(df),
        "unicidad": score_unicidad(df),
        "validez": validez["score_validez"],
        "validez_detalle": validez["detalle"],
        "consistencia": consistencia["score_consistencia"],
        "consistencia_detalle": consistencia["detalle"],
    }


def comparar_antes_despues_completo(datasets_crudos: dict, datasets_limpios: dict) -> pd.DataFrame:
    """
    Tabla comparativa con las 4 dimensiones antes/después, para varios datasets.
    La consistencia solo se puede calcular en el dataset limpio (necesita las
    columnas flag), así que 'antes' queda como None para esa dimensión.
    """
    filas = []
    for nombre in datasets_crudos:
        antes = resumen_calidad_completo(datasets_crudos[nombre], nombre)
        despues = resumen_calidad_completo(datasets_limpios.get(nombre), nombre)
        filas.append({
            "dataset": nombre,
            "completitud_antes": antes["completitud"], "completitud_despues": despues["completitud"],
            "unicidad_antes": antes["unicidad"], "unicidad_despues": despues["unicidad"],
            "validez_antes": antes["validez"], "validez_despues": despues["validez"],
            "consistencia_antes": None, "consistencia_despues": despues["consistencia"],
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Compatibilidad con código existente (notebook, versiones previas de la tab)
# ---------------------------------------------------------------------------

def health_score(df: pd.DataFrame) -> float:
    """
    Health Score combinado (completitud + unicidad), en escala 0-100.
    Se mantiene por compatibilidad con el notebook de exploración —
    para el detalle de las 4 dimensiones, usar resumen_calidad_completo().
    """
    if df is None or len(df) == 0:
        return 0.0
    completitud = 1 - df.isna().mean().mean()
    unicidad = 1 - (df.duplicated().sum() / len(df))
    return round(100 * (completitud + unicidad) / 2, 1)


def comparar_antes_despues(datasets_crudos: dict, datasets_limpios: dict) -> pd.DataFrame:
    """Versión simple (solo Health Score combinado), usada por el notebook."""
    filas = []
    for nombre in datasets_crudos:
        antes = health_score(datasets_crudos[nombre])
        despues = health_score(datasets_limpios.get(nombre))
        filas.append({
            "dataset": nombre, "health_score_antes": antes,
            "health_score_despues": despues, "mejora_pts": round(despues - antes, 1),
        })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    from data_loader import load_all_datasets
    from cleaning import clean_all_datasets

    raw = load_all_datasets()
    cleaned, _ = clean_all_datasets(raw)

    print("=== Comparativo de 4 dimensiones (antes -> después) ===\n")
    tabla = comparar_antes_despues_completo(raw, cleaned)
    print(tabla.to_string(index=False))

    print("\n=== Detalle de validez por columna (feedback, ANTES) ===")
    print(score_validez(raw["feedback"], "feedback")["detalle"])

    print("\n=== Detalle de consistencia (transacciones, DESPUÉS) ===")
    print(score_consistencia(cleaned["transacciones"], "transacciones")["detalle"])