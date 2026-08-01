import pandas as pd


def build_summary_stats(df: pd.DataFrame) -> dict:
    """Comprime el DataFrame filtrado a un dict compacto de estadísticos,
    para no enviar el dataset completo al modelo."""
    if df is None or df.empty:
        return {"n_transacciones": 0}

    summary = {"n_transacciones": len(df)}

    if "Margen_Utilidad" in df.columns:
        df_margen = df[df["Margen_Utilidad"].notna()]
        summary["margen_promedio_usd"] = round(df_margen["Margen_Utilidad"].mean(), 2) if not df_margen.empty else None
        summary["pct_ventas_margen_negativo"] = round(
            100 * (df_margen["Margen_Utilidad"] < 0).mean(), 1
        ) if not df_margen.empty else None

    if "SKU_Fantasma" in df.columns:
        summary["pct_ventas_sku_fantasma"] = round(100 * df["SKU_Fantasma"].mean(), 1)
        if "Precio_Venta_Final" in df.columns and df["Precio_Venta_Final"].sum() > 0:
            ingreso_fantasma = df.loc[df["SKU_Fantasma"], "Precio_Venta_Final"].sum()
            summary["ingreso_en_riesgo_usd"] = round(ingreso_fantasma, 2)
            summary["pct_ingreso_en_riesgo"] = round(
                100 * ingreso_fantasma / df["Precio_Venta_Final"].sum(), 1
            )

    if "Satisfaccion_NPS" in df.columns:
        summary["nps_promedio"] = round(df["Satisfaccion_NPS"].mean(), 1)

    if "Tiempo_Entrega_Real" in df.columns:
        summary["tiempo_entrega_promedio_dias"] = round(df["Tiempo_Entrega_Real"].mean(), 1)

    if "Ticket_Soporte_Abierto" in df.columns:
        summary["tasa_tickets_soporte_pct"] = round(100 * df["Ticket_Soporte_Abierto"].mean(), 1)

    if "Categoria" in df.columns:
        top_categoria = df["Categoria"].value_counts()
        if not top_categoria.empty:
            summary["categoria_mas_vendida"] = top_categoria.index[0]

    if "Ciudad_Destino" in df.columns:
        top_ciudad = df["Ciudad_Destino"].value_counts()
        if not top_ciudad.empty:
            summary["ciudad_con_mas_ventas"] = top_ciudad.index[0]

    return summary


def build_prompt(summary_stats: dict) -> str:
    """Arma el prompt en español para el modelo, a partir del resumen estadístico."""
    lineas = [f"- {clave}: {valor}" for clave, valor in summary_stats.items()]
    resumen_texto = "\n".join(lineas)

    return f"""Eres un consultor senior de datos analizando TechLogistics S.A.S.,
una empresa de retail tecnológico con problemas de rentabilidad y satisfacción
de clientes. A continuación tienes el resumen estadístico de las transacciones
actualmente filtradas en el dashboard:

{resumen_texto}

Con base ÚNICAMENTE en estos datos (no inventes cifras que no aparezcan arriba),
escribe exactamente 3 párrafos cortos de recomendación estratégica para la junta
directiva:

1. Diagnóstico: qué es lo más preocupante de estos números.
2. Causa probable: por qué podría estar pasando esto.
3. Recomendación accionable: qué debería hacer la empresa primero.

Sé directo, cuantitativo (usa las cifras dadas) y evita generalidades vacías."""


def get_ai_recommendation(summary_stats: dict, api_key: str) -> str:
    """Llama a Groq con el resumen y devuelve el texto. Lanza excepción si falla."""
    from groq import Groq

    if summary_stats.get("n_transacciones", 0) == 0:
        return "No hay transacciones en el filtro actual para analizar."

    client = Groq(api_key=api_key)
    prompt = build_prompt(summary_stats)

    respuesta = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=600,
    )
    return respuesta.choices[0].message.content