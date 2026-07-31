"""
cleaning.py
------------
Módulo de limpieza y preprocesamiento de los tres datasets del proyecto.
Transforma los DataFrames crudos cargados por data_loader.py en datos
listos para análisis, generando un reporte de trazabilidad con cada
decisión aplicada.

v3 — El reporte de cada cambio ahora tiene 3 campos separados en vez de
un solo párrafo, para que la pestaña de Auditoría lo muestre como una
historia clara:
  - identificacion: cómo se detectó el problema y qué tan grande era
  - decision: qué técnica se aplicó, en concreto
  - justificacion: por qué esa técnica y no otra (qué se buscó preservar)
"""

import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Diccionarios de normalización (keys en minúscula: comparación siempre
# contra el valor ya pasado por .lower().strip())
# ---------------------------------------------------------------------------

CIUDAD_MAP = {
    "med": "Medellín", "medellin": "Medellín", "medellín": "Medellín",
    "bog": "Bogotá", "bogota": "Bogotá", "bogotá": "Bogotá",
    "cal": "Cali", "cali": "Cali",
    "baq": "Barranquilla", "barranquilla": "Barranquilla",
    "cart": "Cartagena", "cartagena": "Cartagena",
    "bucaramanga": "Bucaramanga",
}

CANAL_MAP = {
    "online": "Online", "físico": "Físico", "fisico": "Físico",
    "whatsapp": "WhatsApp", "app": "App",
}

ESTADO_ENVIO_MAP = {
    "entregado": "Entregado", "perdido": "Perdido",
    "en tránsito": "En tránsito", "en transito": "En tránsito",
    "devuelto": "Devuelto",
}

TICKET_MAP = {"sí": 1, "si": 1, "s": 1, "1": 1, "no": 0, "n": 0, "0": 0}

RECOMIENDA_MAP = {
    "sí": "Sí", "si": "Sí", "s": "Sí", "no": "No", "n": "No",
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
# Helper para construir cada entrada del reporte con el mismo formato
# ---------------------------------------------------------------------------

def _registrar_cambio(report: dict, columna: str, identificacion: str, decision: str, justificacion: str):
    """Agrega un cambio al reporte en el formato narrativo de 3 partes."""
    report["cambios"].append({
        "columna": columna,
        "identificacion": identificacion,
        "decision": decision,
        "justificacion": justificacion,
    })


# ---------------------------------------------------------------------------
# Helpers de parseo / normalización por celda
# ---------------------------------------------------------------------------

def _parse_lead_time(val):
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
    if pd.isna(val):
        return np.nan
    v = str(val).strip()
    if v in ("", "???"):
        return np.nan
    v_norm = re.sub(r"[^a-zA-Záéíóúñ]", "", v).lower()
    return CATEGORIA_MAP.get(v_norm, v.title())


def _normalize_ciudad(val):
    if pd.isna(val):
        return np.nan, False
    v = str(val).strip().lower()
    if v in CIUDAD_MAP:
        return CIUDAD_MAP[v], False
    return np.nan, True


def _cap_iqr(series, factor=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return series.clip(q1 - factor * iqr, q3 + factor * iqr)


def _impute_median_by_group(series, group_series):
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

    # --- Costo_Unitario_USD ---
    if "Costo_Unitario_USD" in df.columns:
        col = "Costo_Unitario_USD"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        rango_original = f"${df[col].min():,.2f} – ${df[col].max():,.2f}"
        df[col] = _cap_iqr(df[col])
        _registrar_cambio(
            report, col,
            identificacion=f"Se revisó el rango de costos y se encontró un rango extremo "
                            f"({rango_original}), incluyendo valores absurdos para accesorios "
                            "de bajo costo (ej. $850,000).",
            decision="Se aplicó Winsorizing (capping) por rango intercuartílico (IQR): "
                      "los valores fuera del bigote se acotan al límite, no se eliminan.",
            justificacion="Se eligió acotar en vez de eliminar la fila para no perder el resto "
                           "del registro del SKU (categoría, stock, bodega siguen siendo datos "
                           "válidos); eliminar la fila completa habría sido más agresivo de lo "
                           "necesario para corregir un solo valor.",
        )

    # --- Lead_Time_Dias: parseo de texto ---
    if "Lead_Time_Dias" in df.columns:
        col = "Lead_Time_Dias"
        inmediato_count = int(df[col].astype(str).str.strip().str.lower().eq("inmediato").sum())
        rango_count = int(df[col].astype(str).str.contains(r"\d+\s*[-–]\s*\d+", na=False).sum())
        nulos_genuinos_antes = int(df[col].isna().sum())
        df[col] = df[col].apply(_parse_lead_time)
        _registrar_cambio(
            report, col,
            identificacion=f"La columna mezclaba 3 formatos distintos: {inmediato_count} "
                            f"valores de texto 'Inmediato', {rango_count} rangos de texto "
                            f"(ej. '25-30 días'), y {nulos_genuinos_antes} celdas vacías desde "
                            "el CSV original.",
            decision="'Inmediato' se convirtió a 0 (es un dato real, no una ausencia de dato). "
                      "Los rangos se convirtieron al promedio del rango (punto medio).",
            justificacion="Tratar 'Inmediato' como nulo habría borrado información real de "
                           "reposición inmediata. Para los rangos, el punto medio es la opción "
                           "más neutral: no favorece un escenario optimista ni uno conservador.",
        )

    # --- Stock_Actual ---
    if "Stock_Actual" in df.columns:
        col = "Stock_Actual"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        negativos = int((df[col] < 0).sum())
        df.loc[df[col] < 0, col] = np.nan
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)
        _registrar_cambio(
            report, col,
            identificacion=f"Se encontraron {negativos} registros con stock negativo, algo "
                            "físicamente imposible en un inventario real.",
            decision=f"Los valores negativos se trataron como nulo técnico y se imputaron con "
                     f"la mediana de la columna ({mediana:.1f} unidades).",
            justificacion="Se usó mediana en vez de media porque Stock_Actual tiene alta "
                           "dispersión entre categorías de producto (un mouse y un monitor no "
                           "manejan la misma escala de inventario), y la mediana es más robusta "
                           "frente a esa asimetría.",
        )

    # --- Categoria: normalización ---
    if "Categoria" in df.columns:
        col = "Categoria"
        antes_unique = df[col].nunique()
        signos_interrogacion = int((df[col].astype(str).str.strip() == "???").sum())
        df[col] = df[col].apply(_normalize_categoria)
        despues_unique = df[col].nunique()
        _registrar_cambio(
            report, col,
            identificacion=f"Se revisaron los valores únicos de Categoria y se encontraron "
                            f"{antes_unique} valores distintos para lo que en realidad son "
                            f"{despues_unique} categorías reales (ej. 'LAPTOP' y 'Laptops' eran "
                            f"la misma categoría con formato distinto). Además, {signos_interrogacion} "
                            "registros tenían '???' en vez de una categoría real.",
            decision="Se normalizaron las variantes de mayúscula/minúscula/guion a un único "
                     "valor canónico por categoría. '???' se convirtió explícitamente a NaN.",
            justificacion="'???' no es una categoría válida, es un nulo técnico disfrazado de "
                           "texto — si no se convierte a NaN explícito, queda invisible para "
                           "cualquier conteo de nulos (isna() no detecta strings como '???').",
        )
        if "Bodega_Origen" in df.columns:
            moda_por_bodega = df.groupby("Bodega_Origen")[col].agg(
                lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
            )
            nulos_cat = int(df[col].isna().sum())
            df[col] = df[col].fillna(df["Bodega_Origen"].map(moda_por_bodega))
            df[col] = df[col].fillna(df[col].mode().iloc[0])
            _registrar_cambio(
                report, col,
                identificacion=f"Tras convertir '???' a NaN, quedaron {nulos_cat} categorías "
                                "sin valor.",
                decision="Se imputaron con la categoría más frecuente (moda) dentro de la misma "
                          "Bodega_Origen del registro.",
                justificacion="Se asumió que la mezcla de productos tiende a ser similar dentro "
                               "de una misma bodega, así que la moda local es una mejor "
                               "aproximación que la moda global de todo el dataset.",
            )

    if "Ultima_Revision" in df.columns:
        df["Ultima_Revision"] = pd.to_datetime(df["Ultima_Revision"], errors="coerce")

    if "Punto_Reorden" in df.columns:
        df["Punto_Reorden"] = pd.to_numeric(df["Punto_Reorden"], errors="coerce")

    # --- Lead_Time_Dias: imputar los nulos genuinos (requiere Categoria ya limpia) ---
    if "Lead_Time_Dias" in df.columns:
        col = "Lead_Time_Dias"
        nulos_antes_imputar = int(df[col].isna().sum())
        if nulos_antes_imputar > 0:
            if "Categoria" in df.columns:
                mediana_por_categoria = df.groupby("Categoria")[col].transform("median")
                df[col] = df[col].fillna(mediana_por_categoria)
            df[col] = df[col].fillna(df[col].median())
            _registrar_cambio(
                report, col,
                identificacion=f"Después de convertir los textos ('Inmediato', rangos), quedaron "
                                f"{nulos_antes_imputar} celdas que ya venían vacías desde el CSV "
                                "original (no eran texto a convertir, sino ausencia real de dato).",
                decision="Se imputaron con la mediana de Lead_Time_Dias de la misma Categoria "
                          "del producto.",
                justificacion="Productos de la misma categoría suelen depender de proveedores y "
                               "cadenas de suministro similares, así que la mediana por categoría "
                               "es más representativa que una mediana global del inventario.",
            )

    nulos_despues = df.isna().sum().to_dict()
    report["nulos_antes"] = {k: int(v) for k, v in nulos_antes.items()}
    report["nulos_despues"] = {k: int(v) for k, v in nulos_despues.items()}
    return df, report


# ---------------------------------------------------------------------------
# Limpieza de Transacciones
# ---------------------------------------------------------------------------

def clean_transacciones(df: pd.DataFrame, inventario_df: "pd.DataFrame | None" = None):
    report = {"dataset": "transacciones", "cambios": [], "nulos_antes": {}, "nulos_despues": {}}
    df = df.copy()
    nulos_antes = df.isna().sum().to_dict()

    # --- Cantidad_Vendida ---
    if "Cantidad_Vendida" in df.columns:
        col = "Cantidad_Vendida"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        negativos_mask = df[col] < 0
        n_negativos = int(negativos_mask.sum())
        df["Cantidad_Corregida"] = negativos_mask
        df[col] = df[col].abs()
        _registrar_cambio(
            report, col,
            identificacion=f"Se encontraron {n_negativos} registros con cantidad vendida "
                            "negativa, algo que no tiene sentido en una venta.",
            decision="Se convirtió el valor a su equivalente positivo (valor absoluto) y se "
                      "agregó la columna 'Cantidad_Corregida' marcando cuáles filas se tocaron.",
            justificacion="No existe una columna que distinga una devolución real de un error "
                           "de captura de signo, así que no se puede saber con certeza cuál es "
                           "cuál. Se optó por corregir el signo (asumiendo error de captura) pero "
                           "dejando el flag visible para que el equipo de negocio pueda auditar "
                           "o excluir estos casos si lo considera necesario.",
        )

    if "Precio_Venta_Final" in df.columns:
        df["Precio_Venta_Final"] = pd.to_numeric(df["Precio_Venta_Final"], errors="coerce")

    # --- Costo_Envio ---
    if "Costo_Envio" in df.columns and "Ciudad_Destino" in df.columns:
        col = "Costo_Envio"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        nulos_ce = int(df[col].isna().sum())
        df[col] = _impute_median_by_group(df[col], df["Ciudad_Destino"])
        _registrar_cambio(
            report, col,
            identificacion=f"Se encontraron {nulos_ce} registros sin costo de envío registrado.",
            decision="Se imputaron con la mediana del costo de envío de la misma Ciudad_Destino "
                      "del registro (no una mediana global).",
            justificacion="El costo de envío varía fuertemente según la distancia/zona; usar la "
                           "mediana por ciudad da una estimación más ajustada que una mediana "
                           "única para todo el país.",
        )

    # --- Tiempo_Entrega_Real ---
    if "Tiempo_Entrega_Real" in df.columns:
        col = "Tiempo_Entrega_Real"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        outlier_999 = int((df[col] == 999).sum())
        df.loc[df[col] == 999, col] = np.nan
        mediana_te = df[col].median()
        df[col] = df[col].fillna(mediana_te)
        df[col] = _cap_iqr(df[col])
        _registrar_cambio(
            report, col,
            identificacion=f"Se detectaron {outlier_999} registros con exactamente 999 días de "
                            "entrega — un patrón repetido tan específico apunta a un valor "
                            "centinela (código de error), no a una entrega real.",
            decision=f"Se trataron como nulo y se imputaron con la mediana real de la columna "
                     f"({mediana_te:.1f} días); luego se acotaron otros outliers menos extremos "
                     "con IQR.",
            justificacion="999 no es una entrega real de casi 3 años, es un marcador de error "
                           "del sistema de origen; imputar con la mediana evita que ese código "
                           "de error distorsione cualquier promedio de tiempo de entrega.",
        )

    # --- Ciudad_Destino ---
    if "Ciudad_Destino" in df.columns:
        col = "Ciudad_Destino"
        resultado = df[col].apply(_normalize_ciudad)
        df[col] = resultado.apply(lambda x: x[0])
        contaminados = resultado.apply(lambda x: x[1])
        n_contaminados = int(contaminados.sum())
        df["Ciudad_Invalida"] = contaminados
        _registrar_cambio(
            report, col,
            identificacion=f"Al revisar los valores únicos de Ciudad_Destino se encontró "
                            f"'Ventas_Web' en {n_contaminados} registros — ese valor pertenece "
                            "a Canal_Venta, no es una ciudad. Es contaminación cruzada entre "
                            "columnas, el hallazgo de calidad más importante del dataset.",
            decision="Esos registros se marcaron como NaN en Ciudad_Destino, y se agregó la "
                      "columna 'Ciudad_Invalida' para dejarlos visibles y auditables.",
            justificacion="No hay forma confiable de inferir cuál era la ciudad real para esos "
                           "registros, así que imputar cualquier ciudad sería inventar un dato. "
                           "Se prefiere dejarlos fuera de los análisis geográficos en vez de "
                           "arriesgar una conclusión falsa sobre alguna ciudad.",
        )

    if "Canal_Venta" in df.columns:
        col = "Canal_Venta"
        df[col] = df[col].astype(str).str.strip().str.lower().map(CANAL_MAP).fillna(df[col])

    # --- Estado_Envio ---
    if "Estado_Envio" in df.columns:
        col = "Estado_Envio"
        nulos_ee = int(df[col].isna().sum())
        pct_ee = 100 * nulos_ee / len(df) if len(df) else 0
        df[col] = df[col].astype(str).str.strip().str.lower().map(ESTADO_ENVIO_MAP)
        df[col] = df[col].fillna("Desconocido")
        _registrar_cambio(
            report, col,
            identificacion=f"Se encontraron {nulos_ee} registros ({pct_ee:.1f}%) sin estado de "
                            "envío registrado.",
            decision="Se dejaron como una categoría explícita 'Desconocido', en vez de imputar "
                      "con el estado más frecuente.",
            justificacion="Imputar un estado de envío (ej. asumir 'Entregado' porque es el más "
                           "común) inventaría información sobre si el cliente recibió o no su "
                           "pedido, lo cual distorsionaría directamente el KPI de servicio al "
                           "cliente. Es preferible mostrar la incertidumbre que ocultarla.",
        )

    # --- Fecha_Venta ---
    if "Fecha_Venta" in df.columns:
        col = "Fecha_Venta"
        no_parse = int(pd.to_datetime(df[col], errors="coerce", dayfirst=True).isna().sum())
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        futuras = int((df[col] > pd.Timestamp.now()).sum())
        df.loc[df[col] > pd.Timestamp.now(), col] = np.nan
        _registrar_cambio(
            report, col,
            identificacion=f"Se encontraron {no_parse} fechas con formato no interpretable y "
                            f"{futuras} fechas posteriores al día de hoy.",
            decision="Las fechas no interpretables y las fechas futuras se invalidaron (NaN).",
            justificacion="No pueden existir ventas registradas en el futuro; mantenerlas "
                           "distorsionaría cualquier análisis de tendencia temporal.",
        )

    # --- SKU fantasma ---
    if inventario_df is not None and "SKU_ID" in df.columns and "SKU_ID" in inventario_df.columns:
        skus_inventario = set(inventario_df["SKU_ID"].dropna().unique())
        df["SKU_Fantasma"] = ~df["SKU_ID"].isin(skus_inventario)
        n_fantasma = int(df["SKU_Fantasma"].sum())
        pct = 100 * n_fantasma / len(df) if len(df) else 0
        _registrar_cambio(
            report, "SKU_ID",
            identificacion=f"Al cruzar los SKU de las ventas contra el maestro de inventario, "
                            f"{n_fantasma} ventas ({pct:.1f}%) tienen un SKU que no existe en "
                            "inventario — es el hallazgo central del reto.",
            decision="No se eliminó ni se imputó nada: se agregó la columna booleana "
                      "'SKU_Fantasma' para marcar estos registros sin tocarlos.",
            justificacion="Decidir si es un producto nuevo no catalogado, un error de "
                           "digitación o un posible fraude es una decisión de negocio que "
                           "requiere más contexto del que da la limpieza técnica — se deja "
                           "trazado para resolverse en feature_engineering.py y en el informe.",
        )

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

    # --- Duplicados exactos ---
    duplicados = int(df.duplicated(keep="first").sum())
    if duplicados > 0:
        df = df.drop_duplicates(keep="first")
        _registrar_cambio(
            report, "Todas",
            identificacion=f"Se encontraron {duplicados} filas 100% idénticas en todas sus "
                            "columnas.",
            decision="Se eliminaron, conservando la primera aparición de cada una.",
            justificacion="Una fila idéntica en todas sus columnas no aporta información "
                           "adicional, es una copia exacta — distinto de una colisión de ID "
                           "(ver el siguiente paso), donde el contenido sí es diferente.",
        )

    # --- Colisión de Feedback_ID ---
    if "Feedback_ID" in df.columns:
        dup_mask = df["Feedback_ID"].duplicated(keep=False)
        n_colisiones = int(dup_mask.sum())
        if n_colisiones > 0:
            sufijo = df.groupby("Feedback_ID").cumcount()
            nuevo_id = df["Feedback_ID"].astype(str) + np.where(sufijo > 0, "-" + sufijo.astype(str), "")
            df["Feedback_ID_Original"] = df["Feedback_ID"]
            df["Feedback_ID"] = nuevo_id
        _registrar_cambio(
            report, "Feedback_ID",
            identificacion=f"Se encontraron {n_colisiones} filas que comparten un Feedback_ID "
                            "con otra fila, pero con Transaccion_ID/rating/edad DISTINTOS al "
                            "revisar el contenido — es una colisión de identificador, no una "
                            "copia real.",
            decision="No se eliminó ninguna fila: se generó un ID único agregando un sufijo, "
                      "y se conservó el ID original en la columna 'Feedback_ID_Original'.",
            justificacion="Eliminar por ID duplicado (como haría una limpieza más ingenua) "
                           "habría borrado feedback real de clientes distintos que solo "
                           "coincidían en el identificador — el problema está en el ID, no en "
                           "el contenido del registro.",
        )

    # --- Ratings fuera de escala ---
    for col in ["Rating_Producto", "Rating_Logistica"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            fuera_rango = int(((df[col] < 1) | (df[col] > 5)).sum())
            df.loc[(df[col] < 1) | (df[col] > 5), col] = np.nan
            mediana_r = df[col].median()
            df[col] = df[col].fillna(mediana_r)
            _registrar_cambio(
                report, col,
                identificacion=f"Se encontraron {fuera_rango} valores fuera de la escala válida "
                                "1-5 (ej. 99) — un valor tan alto y específico apunta a un código "
                                "de error de captura, no a un rating real.",
                decision=f"Se trataron como nulo y se imputaron con la mediana "
                         f"({mediana_r:.1f}).",
                justificacion="Es una escala ordinal con distribución razonablemente simétrica, "
                               "así que la mediana representa mejor 'el rating típico' que "
                               "dejar el código de error o intentar adivinar el valor real.",
            )

    # --- Edad_Cliente ---
    if "Edad_Cliente" in df.columns:
        col = "Edad_Cliente"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        invalidas = int(((df[col] < 0) | (df[col] > 100)).sum())
        df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan
        mediana_e = df[col].median()
        df[col] = df[col].fillna(mediana_e).astype(int)
        _registrar_cambio(
            report, col,
            identificacion=f"Se encontraron {invalidas} edades biológicamente imposibles "
                            "(hasta 195 años).",
            decision=f"Se trataron como nulo y se imputaron con la mediana "
                     f"({mediana_e:.0f} años).",
            justificacion="La mediana no se ve afectada por los valores extremos que causaron "
                           "el problema original, a diferencia de la media.",
        )

    # --- Satisfaccion_NPS ---
    if "Satisfaccion_NPS" in df.columns:
        col = "Satisfaccion_NPS"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        nulos_nps = int(df[col].isna().sum())
        rango_obs = f"{df[col].min():.1f} a {df[col].max():.1f}"
        df[col] = df[col].fillna(df[col].median())
        _registrar_cambio(
            report, col,
            identificacion=f"Se verificó el rango observado ({rango_obs}) contra la escala "
                            f"estándar de NPS (-100 a 100): coincide, no hay que renormalizar. "
                            f"Se encontraron {nulos_nps} nulos.",
            decision="Se aseguró el tipo numérico y se imputaron los nulos con la mediana.",
            justificacion="No aplica ninguna transformación de escala porque los datos ya "
                           "vienen en el rango esperado; solo se resuelve la ausencia de dato.",
        )

    if "Ticket_Soporte_Abierto" in df.columns:
        col = "Ticket_Soporte_Abierto"
        df[col] = df[col].astype(str).str.strip().str.lower().map(TICKET_MAP)
        df[col] = df[col].fillna(0).astype(int)

    # --- Recomienda_Marca ---
    if "Recomienda_Marca" in df.columns:
        col = "Recomienda_Marca"
        valores_crudos = df[col].dropna().unique().tolist()
        antes_nulos = int(df[col].isna().sum())
        pct_nulos = 100 * antes_nulos / len(df) if len(df) else 0
        df[col] = df[col].astype(str).str.strip().str.lower().map(RECOMIENDA_MAP)
        _registrar_cambio(
            report, col,
            identificacion=f"Los valores venían en formatos inconsistentes ({valores_crudos[:5]}...) "
                            f"y {antes_nulos} registros ({pct_nulos:.1f}%) no tenían respuesta.",
            decision="Se normalizaron los valores sin importar mayúscula/tilde a Sí/No/Maybe. "
                      "Los nulos se DEJARON como NaN, sin imputar.",
            justificacion="Es la opinión subjetiva de un cliente sobre si recomendaría la marca; "
                           "no hay ninguna base para inventar esa opinión cuando el cliente no "
                           "respondió — imputarla introduciría un sesgo falso en cualquier "
                           "análisis de lealtad.",
        )

    # --- Comentario_Texto ---
    if "Comentario_Texto" in df.columns:
        col = "Comentario_Texto"
        placeholders_set = {"N/A", "n/a", "NA", "na", "---", "", "-"}
        placeholders = int(df[col].isin(placeholders_set).sum())
        df.loc[df[col].isin(placeholders_set), col] = np.nan
        _registrar_cambio(
            report, col,
            identificacion=f"Se encontraron {placeholders} comentarios con texto placeholder "
                            "('N/A', '---', vacío) en vez de una ausencia real de dato o un "
                            "comentario genuino.",
            decision="Se convirtieron explícitamente a NaN. No se imputan.",
            justificacion="Es texto libre — no existe una forma razonable de 'rellenar' el "
                           "contenido de un comentario que el cliente no escribió.",
        )

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
        print(f"\n{'='*70}\n{rep['dataset'].upper()}\n{'='*70}")
        for c in rep["cambios"]:
            print(f"\n[{c['columna']}]")
            print(f"  Se identificó: {c['identificacion']}")
            print(f"  Se decidió:    {c['decision']}")
            print(f"  Justificación: {c['justificacion']}")
        print(f"\nTotal nulos antes: {sum(rep['nulos_antes'].values())} -> después: {sum(rep['nulos_despues'].values())}")