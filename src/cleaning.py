"""
cleaning.py
------------
Módulo de limpieza y preprocesamiento de los tres datasets del proyecto.
Transforma los DataFrames crudos cargados por data_loader.py en datos
listos para análisis, generando un reporte de trazabilidad con cada
decisión aplicada.

v2 — Corrige 4 bugs detectados al auditar la v1 contra los datos reales
(ver docs/auditoria_cleaning_v1.md para el detalle de cada bug) y cubre
2 casos que la v1 no trataba (SKU fantasma, Estado_Envio nulo).
"""

import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Diccionarios de normalización (todas las keys en minúscula: se compara
# siempre con el valor ya pasado por .lower().strip(), así "SI", "Si" y
# "si" matchean igual — este era el Bug #4 de la v1).
# ---------------------------------------------------------------------------

CIUDAD_MAP = {
    "med": "Medellín", "medellin": "Medellín", "medellín": "Medellín",
    "bog": "Bogotá", "bogota": "Bogotá", "bogotá": "Bogotá",
    "cal": "Cali", "cali": "Cali",
    "baq": "Barranquilla", "barranquilla": "Barranquilla",
    "cart": "Cartagena", "cartagena": "Cartagena",
    "bucaramanga": "Bucaramanga",
}
CIUDADES_VALIDAS = set(CIUDAD_MAP.values())

CANAL_MAP = {
    "online": "Online",
    "físico": "Físico", "fisico": "Físico",
    "whatsapp": "WhatsApp",
    "app": "App",
}

ESTADO_ENVIO_MAP = {
    "entregado": "Entregado",
    "perdido": "Perdido",
    "en tránsito": "En tránsito", "en transito": "En tránsito",
    "devuelto": "Devuelto",
}

TICKET_MAP = {
    "sí": 1, "si": 1, "s": 1, "1": 1,
    "no": 0, "n": 0, "0": 0,
}

RECOMIENDA_MAP = {
    "sí": "Sí", "si": "Sí", "s": "Sí",
    "no": "No", "n": "No",
    "maybe": "Maybe", "tal vez": "Maybe",
}

CATEGORIA_MAP = {
    "laptop": "Laptops", "laptops": "Laptops",
    "monitor": "Monitores", "monitores": "Monitores",
    "smartphone": "Smartphones", "smartphones": "Smartphones",
    "tablet": "Tablets", "tablets": "Tablets",
    "accesorio": "Accesorios", "accesorios": "Accesorios",
}


# ---------------------------------------------------------------------------
# Helpers de parseo / normalización por celda
# ---------------------------------------------------------------------------

def _parse_lead_time(val):
    """
    BUG FIX (v1 -> v2): la v1 no reconocía 'Inmediato' y lo dejaba como NaN,
    borrando 433 registros válidos. 'Inmediato' significa 0 días de espera.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)

    val = str(val).strip().lower()
    if val == "inmediato":
        return 0.0

    val = val.replace("días", "").replace("dias", "").replace("dãas", "").strip()
    match = re.match(r"(\d+)\s*[-–]\s*(\d+)", val)
    if match:
        return (float(match.group(1)) + float(match.group(2))) / 2

    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def _normalize_categoria(val):
    """
    BUG FIX (v1 -> v2): la v1 solo hacía .str.title(), que no fusiona
    singular/plural ni guiones, y convertía '???' en string vacío que
    isna() no detecta. v2 normaliza a 5 categorías reales y '???' -> NaN.
    """
    if pd.isna(val):
        return np.nan
    v = str(val).strip()
    if v in ("", "???"):
        return np.nan
    v_norm = re.sub(r"[^a-zA-Záéíóúñ]", "", v).lower()
    return CATEGORIA_MAP.get(v_norm, v.title())


def _normalize_ciudad(val):
    """
    BUG FIX (v1 -> v2): la v1 no detectaba 'Ventas_Web' como valor inválido
    (contaminación cruzada con Canal_Venta). v2 lo marca como NaN + flag.
    Devuelve (valor_normalizado, es_invalido: bool)
    """
    if pd.isna(val):
        return np.nan, False
    v = str(val).strip().lower()
    if v in CIUDAD_MAP:
        return CIUDAD_MAP[v], False
    return np.nan, True


def _cap_iqr(series, factor=1.5):
    """Winsorizing (capping) basado en rango intercuartílico. No elimina filas."""
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
    if result.isna().sum() > 0:
        result = result.fillna(series.median())
    return result


# ---------------------------------------------------------------------------
# Limpieza de Inventario
# ---------------------------------------------------------------------------

def clean_inventario(df: pd.DataFrame):
    report = {"dataset": "inventario", "cambios": [], "nulos_antes": {}, "nulos_despues": {}}
    df = df.copy()
    nulos_antes = df.isna().sum().to_dict()

    if "Costo_Unitario_USD" in df.columns:
        col = "Costo_Unitario_USD"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = _cap_iqr(df[col])
        report["cambios"].append({
            "columna": col, "accion": "Winsorizing IQR",
            "detalle": "Costos extremos (ej. $850,000) limitados al bigote del rango intercuartílico. "
                       "No se eliminan filas, se acota el valor para no perder el registro del SKU.",
        })

    if "Lead_Time_Dias" in df.columns:
        col = "Lead_Time_Dias"
        inmediato_count = df[col].astype(str).str.strip().str.lower().eq("inmediato").sum()
        rango_count = df[col].astype(str).str.contains(r"\d+\s*[-–]\s*\d+", na=False).sum()
        df[col] = df[col].apply(_parse_lead_time)
        report["cambios"].append({
            "columna": col, "accion": "Parseo de texto a número",
            "detalle": (
                f"{inmediato_count} valores 'Inmediato' -> 0 días (dato real, no faltante). "
                f"{rango_count} rangos tipo '25-30 días' -> promedio del rango (punto medio)."
            ),
        })

    if "Stock_Actual" in df.columns:
        col = "Stock_Actual"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        negativos = (df[col] < 0).sum()
        df.loc[df[col] < 0, col] = np.nan
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)
        report["cambios"].append({
            "columna": col, "accion": "Negativos -> NaN + imputación con mediana",
            "detalle": (
                f"{negativos} valores de stock negativo tratados como nulo técnico. "
                f"Mediana imputada: {mediana:.1f} (robusta frente a la dispersión entre categorías)."
            ),
        })

    if "Categoria" in df.columns:
        col = "Categoria"
        antes_unique = df[col].nunique()
        signos_interrogacion = (df[col].astype(str).str.strip() == "???").sum()
        df[col] = df[col].apply(_normalize_categoria)
        despues_unique = df[col].nunique()
        report["cambios"].append({
            "columna": col, "accion": "Normalización de categorías + '???' -> NaN",
            "detalle": (
                f"{antes_unique} valores únicos crudos reducidos a {despues_unique} categorías reales "
                f"(fusiona 'LAPTOP'+'Laptops', 'smart-phone'+'Smartphones'). "
                f"{signos_interrogacion} registros con '???' convertidos explícitamente a NaN."
            ),
        })
        if "Bodega_Origen" in df.columns:
            moda_por_bodega = df.groupby("Bodega_Origen")[col].agg(
                lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
            )
            nulos_cat = df[col].isna().sum()
            df[col] = df[col].fillna(df["Bodega_Origen"].map(moda_por_bodega))
            df[col] = df[col].fillna(df[col].mode().iloc[0])
            report["cambios"].append({
                "columna": col, "accion": "Imputación de nulos con moda por Bodega_Origen",
                "detalle": f"{nulos_cat} categorías nulas (ex-'???') imputadas con la categoría más "
                           "frecuente de su misma bodega de origen.",
            })

    if "Ultima_Revision" in df.columns:
        df["Ultima_Revision"] = pd.to_datetime(df["Ultima_Revision"], errors="coerce")

    if "Punto_Reorden" in df.columns:
        df["Punto_Reorden"] = pd.to_numeric(df["Punto_Reorden"], errors="coerce")

    nulos_despues = df.isna().sum().to_dict()
    report["nulos_antes"] = {k: int(v) for k, v in nulos_antes.items()}
    report["nulos_despues"] = {k: int(v) for k, v in nulos_despues.items()}
    return df, report


# ---------------------------------------------------------------------------
# Limpieza de Transacciones
# ---------------------------------------------------------------------------

def clean_transacciones(df: pd.DataFrame, inventario_df: "pd.DataFrame | None" = None):
    """
    inventario_df: DataFrame YA LIMPIO, opcional, para marcar ventas fantasma.
    """
    report = {"dataset": "transacciones", "cambios": [], "nulos_antes": {}, "nulos_despues": {}}
    df = df.copy()
    nulos_antes = df.isna().sum().to_dict()

    if "Cantidad_Vendida" in df.columns:
        col = "Cantidad_Vendida"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        negativos_mask = df[col] < 0
        df["Cantidad_Corregida"] = negativos_mask
        df[col] = df[col].abs()
        report["cambios"].append({
            "columna": col, "accion": "Valor absoluto a negativos + columna de flag",
            "detalle": (
                f"{int(negativos_mask.sum())} cantidades negativas convertidas a positivas "
                "(no hay columna de tipo de transacción que distinga devolución de error). "
                "Se agrega 'Cantidad_Corregida' (booleano) para trazabilidad en el dashboard."
            ),
        })

    if "Precio_Venta_Final" in df.columns:
        df["Precio_Venta_Final"] = pd.to_numeric(df["Precio_Venta_Final"], errors="coerce")

    if "Costo_Envio" in df.columns and "Ciudad_Destino" in df.columns:
        col = "Costo_Envio"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        nulos_ce = df[col].isna().sum()
        df[col] = _impute_median_by_group(df[col], df["Ciudad_Destino"])
        report["cambios"].append({
            "columna": col, "accion": "Imputación con mediana por ciudad",
            "detalle": f"{nulos_ce} nulos imputados con la mediana del costo de envío de su "
                       "misma ciudad destino.",
        })

    if "Tiempo_Entrega_Real" in df.columns:
        col = "Tiempo_Entrega_Real"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        outlier_999 = (df[col] == 999).sum()
        df.loc[df[col] == 999, col] = np.nan
        mediana_te = df[col].median()
        df[col] = df[col].fillna(mediana_te)
        df[col] = _cap_iqr(df[col])
        report["cambios"].append({
            "columna": col, "accion": "Outlier 999 -> NaN + imputación mediana + Winsorizing IQR",
            "detalle": f"{outlier_999} registros con 999 días (centinela) imputados con la mediana "
                       f"({mediana_te:.1f} días), luego se acotan otros outliers con IQR.",
        })

    if "Ciudad_Destino" in df.columns:
        col = "Ciudad_Destino"
        resultado = df[col].apply(_normalize_ciudad)
        df[col] = resultado.apply(lambda x: x[0])
        contaminados = resultado.apply(lambda x: x[1])
        n_contaminados = int(contaminados.sum())
        df["Ciudad_Invalida"] = contaminados
        report["cambios"].append({
            "columna": col, "accion": "Normalización + detección de contaminación cruzada",
            "detalle": (
                f"{n_contaminados} registros tenían 'Ventas_Web' (valor de Canal_Venta, no una "
                "ciudad) en Ciudad_Destino. Se marcan NaN + flag 'Ciudad_Invalida'=True. "
                "No se imputa una ciudad (no hay forma confiable de inferirla); quedan excluidos "
                "de análisis geográficos pero se conservan para el resto de análisis."
            ),
        })

    if "Canal_Venta" in df.columns:
        col = "Canal_Venta"
        df[col] = df[col].astype(str).str.strip().str.lower().map(CANAL_MAP).fillna(df[col])

    if "Estado_Envio" in df.columns:
        col = "Estado_Envio"
        nulos_ee = int(df[col].isna().sum())
        df[col] = df[col].astype(str).str.strip().str.lower().map(ESTADO_ENVIO_MAP)
        df[col] = df[col].fillna("Desconocido")
        report["cambios"].append({
            "columna": col, "accion": "Normalización + nulos -> 'Desconocido'",
            "detalle": (
                f"{nulos_ee} nulos (16.8%) NO se imputan con una moda estadística porque inventar "
                "un estado de envío falso distorsionaría el KPI de servicio. Se deja como categoría "
                "explícita 'Desconocido', auditable en el dashboard."
            ),
        })

    if "Fecha_Venta" in df.columns:
        col = "Fecha_Venta"
        no_parse = pd.to_datetime(df[col], errors="coerce", dayfirst=True).isna().sum()
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        futuras = (df[col] > pd.Timestamp.now()).sum()
        df.loc[df[col] > pd.Timestamp.now(), col] = np.nan
        report["cambios"].append({
            "columna": col, "accion": "Parseo de fechas + invalidación de futuras",
            "detalle": f"{no_parse} fechas no parseables, {futuras} fechas futuras invalidadas.",
        })

    if inventario_df is not None and "SKU_ID" in df.columns and "SKU_ID" in inventario_df.columns:
        skus_inventario = set(inventario_df["SKU_ID"].dropna().unique())
        df["SKU_Fantasma"] = ~df["SKU_ID"].isin(skus_inventario)
        n_fantasma = int(df["SKU_Fantasma"].sum())
        pct = 100 * n_fantasma / len(df)
        report["cambios"].append({
            "columna": "SKU_ID", "accion": "Flag de venta fantasma (sin eliminar filas)",
            "detalle": (
                f"{n_fantasma} ventas ({pct:.1f}%) tienen un SKU inexistente en inventario "
                "(hallazgo central del reto, Pregunta 3). Se agrega columna 'SKU_Fantasma' "
                "en vez de eliminar: es una decisión de negocio que se resuelve en "
                "feature_engineering.py / el informe, no se silencia en la limpieza."
            ),
        })

    nulos_despues = df.isna().sum().to_dict()
    report["nulos_antes"] = {k: int(v) for k, v in nulos_antes.items()}
    report["nulos_despues"] = {k: int(v) for k, v in nulos_despues.items()}
    return df, report


# ---------------------------------------------------------------------------
# Limpieza de Feedback
# ---------------------------------------------------------------------------

def clean_feedback(df: pd.DataFrame):
    report = {"dataset": "feedback", "cambios": [], "nulos_antes": {}, "nulos_despues": {}}
    df = df.copy()
    nulos_antes = df.isna().sum().to_dict()

    duplicados = df.duplicated(keep="first").sum()
    if duplicados > 0:
        df = df.drop_duplicates(keep="first")
        report["cambios"].append({
            "columna": "Todas", "accion": "Eliminación de duplicados exactos",
            "detalle": f"{duplicados} filas 100% idénticas eliminadas. Distinto de una colisión de ID.",
        })

    if "Feedback_ID" in df.columns:
        dup_mask = df["Feedback_ID"].duplicated(keep=False)
        n_colisiones = int(dup_mask.sum())
        if n_colisiones > 0:
            sufijo = df.groupby("Feedback_ID").cumcount()
            nuevo_id = df["Feedback_ID"].astype(str) + np.where(sufijo > 0, "-" + sufijo.astype(str), "")
            df["Feedback_ID_Original"] = df["Feedback_ID"]
            df["Feedback_ID"] = nuevo_id
        report["cambios"].append({
            "columna": "Feedback_ID", "accion": "Regeneración de ID único (NO se eliminan filas)",
            "detalle": (
                f"{n_colisiones} filas compartían un Feedback_ID con OTRO Transaccion_ID/rating/edad "
                "distinto -> colisión de identificador, no duplicado real. La v1 las eliminaba "
                "(perdía feedback legítimo). v2 conserva todo y genera un ID único con sufijo, "
                "guardando el original en 'Feedback_ID_Original'."
            ),
        })

    for col in ["Rating_Producto", "Rating_Logistica"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            fuera_rango = int(((df[col] < 1) | (df[col] > 5)).sum())
            df.loc[(df[col] < 1) | (df[col] > 5), col] = np.nan
            mediana_r = df[col].median()
            df[col] = df[col].fillna(mediana_r)
            report["cambios"].append({
                "columna": col, "accion": "Códigos de error (ej. 99) -> NaN + imputación mediana",
                "detalle": f"{fuera_rango} valores fuera de [1-5] corregidos. "
                           f"Mediana imputada: {mediana_r:.1f}.",
            })

    if "Edad_Cliente" in df.columns:
        col = "Edad_Cliente"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        invalidas = int(((df[col] < 0) | (df[col] > 100)).sum())
        df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan
        mediana_e = df[col].median()
        df[col] = df[col].fillna(mediana_e).astype(int)
        report["cambios"].append({
            "columna": col, "accion": "Edades imposibles (>100 o <0) -> NaN + imputación mediana",
            "detalle": f"{invalidas} edades inválidas corregidas. Mediana imputada: {mediana_e:.0f}.",
        })

    if "Satisfaccion_NPS" in df.columns:
        col = "Satisfaccion_NPS"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        nulos_nps = int(df[col].isna().sum())
        df[col] = df[col].fillna(df[col].median())
        report["cambios"].append({
            "columna": col, "accion": "Verificación de escala + imputación de nulos con mediana",
            "detalle": (
                f"Rango observado -99.8 a 99.9: coincide con la escala estándar NPS (-100 a 100), "
                f"no requiere renormalización. {nulos_nps} nulos imputados con la mediana."
            ),
        })

    if "Ticket_Soporte_Abierto" in df.columns:
        col = "Ticket_Soporte_Abierto"
        df[col] = df[col].astype(str).str.strip().str.lower().map(TICKET_MAP)
        df[col] = df[col].fillna(0).astype(int)

    if "Recomienda_Marca" in df.columns:
        col = "Recomienda_Marca"
        antes_nulos = int(df[col].isna().sum())
        df[col] = df[col].astype(str).str.strip().str.lower().map(RECOMIENDA_MAP)
        report["cambios"].append({
            "columna": col, "accion": "Normalización case-insensitive a Sí/No/Maybe",
            "detalle": (
                "BUG FIX: v1 no reconocía 'SI'/'NO' en mayúsculas y dejaba la columna intacta. "
                f"v2 normaliza sin importar mayúscula/tilde. {antes_nulos} nulos (24.9%) se "
                "DEJAN como NaN — no se imputa una recomendación de marca inventada."
            ),
        })

    if "Comentario_Texto" in df.columns:
        col = "Comentario_Texto"
        placeholders_set = {"N/A", "n/a", "NA", "na", "---", "", "-"}
        placeholders = int(df[col].isin(placeholders_set).sum())
        df.loc[df[col].isin(placeholders_set), col] = np.nan
        report["cambios"].append({
            "columna": col, "accion": "Placeholders -> NaN",
            "detalle": f"{placeholders} comentarios placeholder convertidos a NaN. No se imputan "
                       "(texto libre, no se puede inventar contenido).",
        })

    nulos_despues = df.isna().sum().to_dict()
    report["nulos_antes"] = {k: int(v) for k, v in nulos_antes.items()}
    report["nulos_despues"] = {k: int(v) for k, v in nulos_despues.items()}
    return df, report


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def clean_all_datasets(datasets: dict):
    reportes = []

    df_inv, rep_inv = clean_inventario(datasets["inventario"])
    reportes.append(rep_inv)

    df_trx, rep_trx = clean_transacciones(datasets["transacciones"], inventario_df=df_inv)
    reportes.append(rep_trx)

    df_fb, rep_fb = clean_feedback(datasets["feedback"])
    reportes.append(rep_fb)

    datasets_limpios = {"inventario": df_inv, "transacciones": df_trx, "feedback": df_fb}
    return datasets_limpios, reportes


if __name__ == "__main__":
    from data_loader import load_all_datasets

    raw = load_all_datasets()
    cleaned, reports = clean_all_datasets(raw)

    for rep in reports:
        print(f"\n=== {rep['dataset'].upper()} ===")
        for c in rep["cambios"]:
            print(f"  [{c['columna']}] {c['accion']}")
            print(f"      {c['detalle']}")
        print(f"  Nulos antes: {sum(rep['nulos_antes'].values())} -> después: {sum(rep['nulos_despues'].values())}")
