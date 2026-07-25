"""
cleaning.py

Módulo de limpieza y preprocesamiento de los tres datasets del proyecto.
Transforma los DataFrames crudos cargados por data_loader.py en datos
listos para análisis, generando un reporte de trazabilidad con cada
decisión aplicada.
"""

import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Diccionarios de normalización
# ---------------------------------------------------------------------------

CIUDAD_MAP = {
    "MED": "Medellín",
    "med": "Medellín",
    "Medellin": "Medellín",
    "BOG": "Bogotá",
    "bog": "Bogotá",
    "Bogota": "Bogotá",
    "CAL": "Cali",
    "cal": "Cali",
    "Barranquilla": "Barranquilla",
    "BAQ": "Barranquilla",
    "CART": "Cartagena",
    "Cartagena": "Cartagena",
}

CANAL_MAP = {
    "Online": "Online",
    "Físico": "Físico",
    "Fisico": "Físico",
    "WhatsApp": "WhatsApp",
}

ESTADO_ENVIO_MAP = {
    "Entregado": "Entregado",
    "Perdido": "Perdido",
    "En tránsito": "En tránsito",
    "En tr\u00e1nsito": "En tránsito",
    "Devuelto": "Devuelto",
}

TICKET_MAP = {
    "Sí": 1,
    "Si": 1,
    "sí": 1,
    "si": 1,
    "S": 1,
    "s": 1,
    "No": 0,
    "no": 0,
    "N": 0,
    "n": 0,
    "0": 0,
    "1": 1,
    0: 0,
    1: 1,
}

RECOMIENDA_MAP = {
    "Sí": "Sí",
    "Si": "Sí",
    "sí": "Sí",
    "si": "Sí",
    "S": "Sí",
    "No": "No",
    "no": "No",
    "N": "No",
    "Maybe": "Maybe",
    "Tal vez": "Maybe",
}


def _parse_lead_time(val):
    """
    Extrae un valor numérico de Lead_Time_Dias.
    - Si ya es número, lo devuelve tal cual.
    - Si es un rango tipo '25-30 días', devuelve el promedio.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).strip().lower()
    val = val.replace("días", "").replace("dias", "").replace("dãas", "").strip()

    match = re.match(r"(\d+)\s*[-–]\s*(\d+)", val)
    if match:
        return (float(match.group(1)) + float(match.group(2))) / 2

    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def _cap_iqr(series, factor=1.5):
    """
    Aplica Winsorizing (capping) basado en rango intercuartílico.
    No elimina registros, sólo limita valores extremos al bigote inferior/superior.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return series.clip(lower, upper)


def _impute_median_by_group(series, group_series):
    """Imputa nulos de `series` con la mediana por grupo definido en `group_series`."""
    medians = series.groupby(group_series).median()
    result = series.fillna(series.map(medians))
    remaining = result.isna().sum()
    if remaining > 0:
        result = result.fillna(series.median())
    return result


# ---------------------------------------------------------------------------
# Limpieza de Inventario
# ---------------------------------------------------------------------------

def clean_inventario(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Limpia el dataset de inventario central.
    """
    report = {"dataset": "inventario", "cambios": [], "nulos_antes": {}, "nulos_despues": {}}
    df = df.copy()

    nulos_antes = df.isna().sum().to_dict()

    # --- Costo_Unitario_USD: capping IQR ---
    if "Costo_Unitario_USD" in df.columns:
        col = "Costo_Unitario_USD"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        antes = df[col].isna().sum()
        df[col] = _cap_iqr(df[col])
        report["cambios"].append({
            "columna": col,
            "accion": "Winsorizing IQR",
            "detalle": "Valores extremos limitados al bigote del rango intercuartílico",
        })

    # --- Lead_Time_Dias: parsear texto a número ---
    if "Lead_Time_Dias" in df.columns:
        col = "Lead_Time_Dias"
        texto_count = df[col].apply(
            lambda x: isinstance(x, str) and bool(re.search(r"[a-zA-Z]", str(x)))
        ).sum()
        df[col] = df[col].apply(_parse_lead_time)
        report["cambios"].append({
            "columna": col,
            "accion": "Parseo de rangos textuales",
            "detalle": f"{texto_count} valores textuales (ej. '25-30 días') convertidos a numérico",
        })

    # --- Stock_Actual: negativos -> NaN, luego imputar ---
    if "Stock_Actual" in df.columns:
        col = "Stock_Actual"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        negativos = (df[col] < 0).sum()
        df.loc[df[col] < 0, col] = np.nan
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)
        report["cambios"].append({
            "columna": col,
            "accion": "Negativos a NaN + imputación con mediana",
            "detalle": f"{negativos} stocks negativos corregidos. Mediana imputada: {mediana:.1f}",
        })

    # --- Ultima_Revision: asegurar datetime ---
    if "Ultima_Revision" in df.columns:
        df["Ultima_Revision"] = pd.to_datetime(df["Ultima_Revision"], errors="coerce")

    # --- Categoria: estandarizar mayúsculas ---
    if "Categoria" in df.columns:
        df["Categoria"] = (
            df["Categoria"]
            .str.strip()
            .str.replace(r"[^\w\sáéíóúñ]", "", regex=True)
            .str.title()
        )

    # --- Punto_Reorden: asegurar numérico ---
    if "Punto_Reorden" in df.columns:
        df["Punto_Reorden"] = pd.to_numeric(df["Punto_Reorden"], errors="coerce")

    nulos_despues = df.isna().sum().to_dict()
    report["nulos_antes"] = {k: int(v) for k, v in nulos_antes.items()}
    report["nulos_despues"] = {k: int(v) for k, v in nulos_despues.items()}

    return df, report


# ---------------------------------------------------------------------------
# Limpieza de Transacciones
# ---------------------------------------------------------------------------

def clean_transacciones(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Limpia el dataset de transacciones logísticas.
    """
    report = {"dataset": "transacciones", "cambios": [], "nulos_antes": {}, "nulos_despues": {}}
    df = df.copy()

    nulos_antes = df.isna().sum().to_dict()

    # --- Cantidad_Vendida: negativos -> absoluto (posible error de signo) ---
    if "Cantidad_Vendida" in df.columns:
        col = "Cantidad_Vendida"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        negativos = (df[col] < 0).sum()
        df[col] = df[col].abs()
        report["cambios"].append({
            "columna": col,
            "accion": "Valor absoluto a negativos",
            "detalle": f"{negativos} cantidades negativas convertidas a positivas (asumiendo error de signo)",
        })

    # --- Precio_Venta_Final: asegurar numérico ---
    if "Precio_Venta_Final" in df.columns:
        df["Precio_Venta_Final"] = pd.to_numeric(df["Precio_Venta_Final"], errors="coerce")

    # --- Costo_Envio: imputar nulos con mediana por ciudad ---
    if "Costo_Envio" in df.columns and "Ciudad_Destino" in df.columns:
        col = "Costo_Envio"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        nulos_ce = df[col].isna().sum()
        df[col] = _impute_median_by_group(df[col], df["Ciudad_Destino"])
        report["cambios"].append({
            "columna": col,
            "accion": "Imputación con mediana por ciudad",
            "detalle": f"{nulos_ce} nulos imputados usando mediana del costo de envío por ciudad destino",
        })

    # --- Tiempo_Entrega_Real: outlier 999 -> NaN, luego capping IQR ---
    if "Tiempo_Entrega_Real" in df.columns:
        col = "Tiempo_Entrega_Real"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        outlier_999 = (df[col] == 999).sum()
        df.loc[df[col] == 999, col] = np.nan
        mediana_te = df[col].median()
        df[col] = df[col].fillna(mediana_te)
        df[col] = _cap_iqr(df[col])
        report["cambios"].append({
            "columna": col,
            "accion": "Outlier 999 -> NaN + imputación mediana + Winsorizing IQR",
            "detalle": f"{outlier_999} registros con 999 días tratados como centinela. Mediana imputada: {mediana_te:.1f}",
        })

    # --- Ciudad_Destino: normalizar ---
    if "Ciudad_Destino" in df.columns:
        col = "Ciudad_Destino"
        valores_originales = df[col].value_counts().to_dict()
        df[col] = df[col].str.strip().map(CIUDAD_MAP).fillna(df[col].str.strip().str.title())
        report["cambios"].append({
            "columna": col,
            "accion": "Normalización de nombres",
            "detalle": "Abreviaciones (MED, BOG, CAL) normalizadas a nombre completo de ciudad",
        })

    # --- Canal_Venta: normalizar ---
    if "Canal_Venta" in df.columns:
        col = "Canal_Venta"
        df[col] = df[col].str.strip().map(CANAL_MAP).fillna(df[col].str.strip())
        df.loc[df[col].str.contains(r"[^\w\sáéíóúñ]", na=False, regex=True), col] = "Otro"

    # --- Estado_Envio: normalizar ---
    if "Estado_Envio" in df.columns:
        col = "Estado_Envio"
        df[col] = df[col].str.strip().map(ESTADO_ENVIO_MAP).fillna(df[col].str.strip())

    # --- Fecha_Venta: parsear y filtrar futuras ---
    if "Fecha_Venta" in df.columns:
        col = "Fecha_Venta"
        no_parse = pd.to_datetime(df[col], errors="coerce", dayfirst=True).isna().sum()
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        futuras = (df[col] > pd.Timestamp.now()).sum()
        df.loc[df[col] > pd.Timestamp.now(), col] = np.nan
        report["cambios"].append({
            "columna": col,
            "accion": "Parseo de fechas + limpieza de futuras",
            "detalle": f"{no_parse} fechas no parseables, {futuras} fechas futuras invalidadas",
        })

    nulos_despues = df.isna().sum().to_dict()
    report["nulos_antes"] = {k: int(v) for k, v in nulos_antes.items()}
    report["nulos_despues"] = {k: int(v) for k, v in nulos_despues.items()}

    return df, report


# ---------------------------------------------------------------------------
# Limpieza de Feedback
# ---------------------------------------------------------------------------

def clean_feedback(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Limpia el dataset de feedback de clientes.
    """
    report = {"dataset": "feedback", "cambios": [], "nulos_antes": {}, "nulos_despues": {}}
    df = df.copy()

    nulos_antes = df.isna().sum().to_dict()

    # --- Duplicados exactos ---
    duplicados = df.duplicated(keep="first").sum()
    if duplicados > 0:
        df = df.drop_duplicates(keep="first")
        report["cambios"].append({
            "columna": "Todas",
            "accion": "Eliminación de duplicados exactos",
            "detalle": f"{duplicados} filas duplicadas eliminadas (keep=first)",
        })

    # --- Duplicados por Feedback_ID ---
    if "Feedback_ID" in df.columns:
        dup_id = df["Feedback_ID"].duplicated(keep="first").sum()
        if dup_id > 0:
            df = df.drop_duplicates(subset=["Feedback_ID"], keep="first")
            report["cambios"].append({
                "columna": "Feedback_ID",
                "accion": "Eliminación de duplicados por ID",
                "detalle": f"{dup_id} Feedback_ID duplicados eliminados (keep=first)",
            })

    # --- Rating_Producto y Rating_Logistica: forzar rango [1-5] ---
    for col in ["Rating_Producto", "Rating_Logistica"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            fuera_rango = ((df[col] < 1) | (df[col] > 5)).sum()
            df.loc[(df[col] < 1) | (df[col] > 5), col] = np.nan
            mediana_r = df[col].median()
            df[col] = df[col].fillna(mediana_r)
            report["cambios"].append({
                "columna": col,
                "accion": "Outliers de escala [1-5] -> NaN + imputación mediana",
                "detalle": f"{fuera_rango} valores fuera de rango corregidos. Mediana imputada: {mediana_r:.1f}",
            })

    # --- Edad_Cliente: imposibles -> NaN, imputar ---
    if "Edad_Cliente" in df.columns:
        col = "Edad_Cliente"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        invalidas = ((df[col] < 0) | (df[col] > 100)).sum()
        df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan
        mediana_e = df[col].median()
        df[col] = df[col].fillna(mediana_e).astype(int)
        report["cambios"].append({
            "columna": col,
            "accion": "Edades imposibles (>100 o <0) -> NaN + imputación mediana",
            "detalle": f"{invalidas} edades inválidas corregidas. Mediana imputada: {mediana_e:.0f}",
        })

    # --- Satisfaccion_NPS: asegurar numérico (NPS puede ser negativo, es válido) ---
    if "Satisfaccion_NPS" in df.columns:
        df["Satisfaccion_NPS"] = pd.to_numeric(df["Satisfaccion_NPS"], errors="coerce")
        df["Satisfaccion_NPS"] = df["Satisfaccion_NPS"].fillna(df["Satisfaccion_NPS"].median())

    # --- Ticket_Soporte_Abierto: estandarizar a 0/1 ---
    if "Ticket_Soporte_Abierto" in df.columns:
        col = "Ticket_Soporte_Abierto"
        df[col] = df[col].astype(str).str.strip().map(TICKET_MAP)
        df[col] = df[col].fillna(0).astype(int)

    # --- Recomienda_Marca: estandarizar ---
    if "Recomienda_Marca" in df.columns:
        col = "Recomienda_Marca"
        df[col] = df[col].astype(str).str.strip().map(RECOMIENDA_MAP).fillna(df[col])

    # --- Comentario_Texto: placeholders -> NaN ---
    if "Comentario_Texto" in df.columns:
        col = "Comentario_Texto"
        placeholders = df[col].isin(["N/A", "n/a", "NA", "---", "", "-"]).sum()
        df.loc[df[col].isin(["N/A", "n/a", "NA", "---", "", "-"]), col] = np.nan
        report["cambios"].append({
            "columna": col,
            "accion": "Placeholders a NaN",
            "detalle": f"{placeholders} comentarios placeholder (N/A, ---, vacío) convertidos a NaN",
        })

    nulos_despues = df.isna().sum().to_dict()
    report["nulos_antes"] = {k: int(v) for k, v in nulos_antes.items()}
    report["nulos_despues"] = {k: int(v) for k, v in nulos_despues.items()}

    return df, report


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def clean_all_datasets(
    datasets: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """
    Aplica limpieza a los tres datasets y retorna los DataFrames limpios
    junto con una lista de reportes por dataset.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Diccionario con claves 'inventario', 'transacciones', 'feedback'.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], list[dict]]
        - DataFrames limpios (mismas claves)
        - Lista de reportes con trazabilidad de cambios
    """
    reportes = []

    df_inv, rep_inv = clean_inventario(datasets["inventario"])
    reportes.append(rep_inv)

    df_trx, rep_trx = clean_transacciones(datasets["transacciones"])
    reportes.append(rep_trx)

    df_fb, rep_fb = clean_feedback(datasets["feedback"])
    reportes.append(rep_fb)

    datasets_limpios = {
        "inventario": df_inv,
        "transacciones": df_trx,
        "feedback": df_fb,
    }

    return datasets_limpios, reportes


if __name__ == "__main__":
    from data_loader import load_all_datasets

    raw = load_all_datasets()
    cleaned, reports = clean_all_datasets(raw)

    for rep in reports:
        print(f"\n=== {rep['dataset'].upper()} ===")
        for c in rep["cambios"]:
            print(f"  [{c['columna']}] {c['accion']}: {c['detalle']}")
        print(f"  Nulos antes: {sum(rep['nulos_antes'].values())} -> después: {sum(rep['nulos_despues'].values())}")
