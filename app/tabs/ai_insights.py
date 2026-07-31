"""
Integración con Groq (Llama-3): genera recomendaciones estratégicas a partir
de un resumen estadístico detallado (igual al de Análisis Final), por pregunta
de negocio. La API key se lee de st.secrets["GROQ_API_KEY"], nunca hardcodeada.
"""

import pandas as pd


# Resumen estadístico detallado

def build_summary_stats(df: pd.DataFrame) -> dict:
    """Comprime el DataFrame a un dict estructurado y detallado (agregaciones,
    no el dataset completo) para que el modelo analice por pregunta de negocio."""
    if df is None or df.empty:
        return {"n_transacciones": 0}

    resumen = {"n_transacciones": len(df)}

    # Rentabilidad (Pregunta 1)
    if "Margen_Utilidad" in df.columns:
        df_margen = df[df["Margen_Utilidad"].notna()]
        if not df_margen.empty:
            df_neg = df_margen[df_margen["Margen_Utilidad"] < 0]
            resumen["rentabilidad"] = {
                "margen_promedio_usd": round(df_margen["Margen_Utilidad"].mean(), 2),
                "n_ventas_margen_negativo": int(len(df_neg)),
                "pct_ventas_margen_negativo": round(100 * len(df_neg) / len(df_margen), 1),
                "perdida_total_usd": round(df_neg["Margen_Utilidad"].sum(), 2) if not df_neg.empty else 0,
            }
            if not df_neg.empty and "SKU_ID" in df.columns:
                top_sku = (
                    df_neg.groupby("SKU_ID")["Margen_Utilidad"].sum()
                    .sort_values().head(5)
                )
                resumen["rentabilidad"]["top_5_sku_peor_margen"] = {
                    sku: round(valor, 2) for sku, valor in top_sku.items()
                }
            if not df_neg.empty and "Canal_Venta" in df.columns:
                por_canal = df_neg.groupby("Canal_Venta")["Margen_Utilidad"].sum().sort_values()
                resumen["rentabilidad"]["perdida_por_canal_usd"] = {
                    canal: round(valor, 2) for canal, valor in por_canal.items()
                }

    # Logística (Pregunta 2)
    if {"Ciudad_Destino", "Tiempo_Entrega_Real", "Satisfaccion_NPS"}.issubset(df.columns):
        df_val = df.dropna(subset=["Ciudad_Destino", "Tiempo_Entrega_Real", "Satisfaccion_NPS"])
        if not df_val.empty:
            corr_por_ciudad = (
                df_val.groupby("Ciudad_Destino")
                .apply(lambda g: g["Tiempo_Entrega_Real"].corr(g["Satisfaccion_NPS"]) if len(g) > 5 else None)
                .dropna()
                .sort_values()
            )
            tiempo_por_ciudad = df_val.groupby("Ciudad_Destino")["Tiempo_Entrega_Real"].mean().round(1)
            nps_por_ciudad = df_val.groupby("Ciudad_Destino")["Satisfaccion_NPS"].mean().round(1)
            resumen["logistica"] = {
                "tiempo_entrega_promedio_dias_por_ciudad": tiempo_por_ciudad.to_dict(),
                "nps_promedio_por_ciudad": nps_por_ciudad.to_dict(),
                "correlacion_tiempo_vs_nps_por_ciudad": {
                    c: round(v, 3) for c, v in corr_por_ciudad.items()
                },
            }

    # Venta invisible / SKU fantasma (Pregunta 3)
    if "SKU_Fantasma" in df.columns and "Precio_Venta_Final" in df.columns:
        n_fantasma = int(df["SKU_Fantasma"].sum())
        ingreso_total = df["Precio_Venta_Final"].sum()
        ingreso_fantasma = df.loc[df["SKU_Fantasma"], "Precio_Venta_Final"].sum()
        resumen["venta_invisible"] = {
            "n_ventas_sku_fantasma": n_fantasma,
            "pct_ventas_sku_fantasma": round(100 * n_fantasma / len(df), 1) if len(df) else 0,
            "ingreso_en_riesgo_usd": round(ingreso_fantasma, 2),
            "pct_ingreso_en_riesgo": round(100 * ingreso_fantasma / ingreso_total, 1) if ingreso_total else 0,
        }
        if n_fantasma > 0 and "Canal_Venta" in df.columns:
            por_canal = df[df["SKU_Fantasma"]]["Canal_Venta"].value_counts()
            resumen["venta_invisible"]["distribucion_por_canal"] = por_canal.to_dict()

    # Paradoja stock / NPS por categoría (Pregunta 4)
    if {"Categoria", "Stock_Actual", "Satisfaccion_NPS"}.issubset(df.columns):
        df_val = df.dropna(subset=["Categoria", "Stock_Actual", "Satisfaccion_NPS"])
        if not df_val.empty:
            resumen_cat = df_val.groupby("Categoria").agg(
                Stock_Promedio=("Stock_Actual", "mean"),
                NPS_Promedio=("Satisfaccion_NPS", "mean"),
                Rating_Promedio=("Rating_Producto", "mean"),
            ).round(2)
            resumen["fidelidad_por_categoria"] = resumen_cat.to_dict(orient="index")

    # Riesgo operativo por bodega (Pregunta 5)
    if {"Bodega_Origen", "Ultima_Revision", "Ticket_Soporte_Abierto"}.issubset(df.columns):
        df_val = df.dropna(subset=["Bodega_Origen", "Ultima_Revision"]).copy()
        if not df_val.empty:
            fecha_ref = df_val["Ultima_Revision"].max()
            df_val["Dias_Sin_Revision"] = (fecha_ref - df_val["Ultima_Revision"]).dt.days
            resumen_bod = df_val.groupby("Bodega_Origen").agg(
                Dias_Sin_Revision_Promedio=("Dias_Sin_Revision", "mean"),
                Tasa_Ticket_Soporte_Pct=("Ticket_Soporte_Abierto", "mean"),
            ).round(2)
            resumen_bod["Tasa_Ticket_Soporte_Pct"] = (resumen_bod["Tasa_Ticket_Soporte_Pct"] * 100).round(1)
            resumen["riesgo_operativo_por_bodega"] = resumen_bod.to_dict(orient="index")

    return resumen


# Prompt de consultoría completa

def build_prompt(summary_stats: dict) -> str:
    """Arma el prompt en español pidiendo un análisis de consultoría completo,
    estructurado por pregunta de negocio."""
    import json
    resumen_json = json.dumps(summary_stats, ensure_ascii=False, indent=2, default=str)

    return f"""Eres un consultor senior de datos analizando TechLogistics S.A.S., una empresa
de retail tecnológico que detectó erosión de margen y caída de lealtad de clientes.
A continuación tienes el resumen estadístico DETALLADO de las transacciones
actualmente filtradas en el dashboard (agregaciones reales, no inventes cifras
que no aparezcan aquí):

```json
{resumen_json}
```

Con base ÚNICAMENTE en estos datos, escribe un análisis de consultoría completo
para la junta directiva, con esta estructura exacta:

## Diagnóstico General
2-3 frases resumiendo el estado general del negocio según los datos.

## Hallazgos por Pregunta Estratégica
Para cada bloque de datos presente en el resumen (rentabilidad, logística, venta
invisible, fidelidad por categoría, riesgo operativo), escribe un hallazgo corto
y cuantitativo (cita las cifras exactas del resumen). Si algún bloque no está
presente en los datos, sáltalo — no inventes el hallazgo.

## Plan de Acción Priorizado
3 recomendaciones concretas y accionables, cada una con:
- Qué hacer (una frase clara)
- Por qué (basado en el dato específico que lo justifica)
- Complejidad estimada (Baja / Media / Alta)

Sé directo, cuantitativo, y evita generalidades vacías tipo "mejorar la comunicación
con el cliente" — cada recomendación debe estar atada a una cifra concreta del
resumen de datos."""


# Llamada a Groq

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
        temperature=0.3,
        max_tokens=1800,
    )
    return respuesta.choices[0].message.content