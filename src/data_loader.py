"""
data_loader.py

Módulo encargado de cargar los datasets crudos del proyecto (inventario,
transacciones y feedback) de forma robusta, sin importar desde dónde se
ejecute el script (notebook, `streamlit run`, o consola).
"""

import os
import pandas as pd

# Ruta base del proyecto: sube un nivel desde src/ hasta la raíz del repo
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW_DATA_DIR = os.path.join(_BASE_DIR, "data", "raw")

# Nombres de archivo esperados
_ARCHIVOS = {
    "inventario": "inventario_central_v2.csv",
    "transacciones": "transacciones_logistica_v2.csv",
    "feedback": "feedback_clientes_v2.csv",
}


def _cargar_csv(nombre_archivo: str) -> pd.DataFrame:
    """
    Carga un CSV desde data/raw/ con manejo de excepciones.

    Parameters
    ----------
    nombre_archivo : str
        Nombre del archivo CSV dentro de data/raw/.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta esperada.
    ValueError
        Si el archivo existe pero está vacío o no se puede parsear.
    """
    ruta = os.path.join(_RAW_DATA_DIR, nombre_archivo)

    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el archivo esperado en: {ruta}\n"
            f"Verifica que '{nombre_archivo}' esté dentro de data/raw/."
        )

    try:
        df = pd.read_csv(ruta)
    except pd.errors.EmptyDataError:
        raise ValueError(f"El archivo '{nombre_archivo}' está vacío.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error al parsear '{nombre_archivo}': {e}")

    return df


def load_inventario() -> pd.DataFrame:
    """Carga el dataset de inventario central (2,500 registros esperados)."""
    return _cargar_csv(_ARCHIVOS["inventario"])


def load_transacciones() -> pd.DataFrame:
    """Carga el dataset de transacciones logísticas (10,000 registros esperados)."""
    return _cargar_csv(_ARCHIVOS["transacciones"])


def load_feedback() -> pd.DataFrame:
    """Carga el dataset de feedback de clientes (4,500 registros esperados)."""
    return _cargar_csv(_ARCHIVOS["feedback"])


def load_all_datasets() -> dict[str, pd.DataFrame]:
    """
    Carga los 3 datasets del proyecto en un solo diccionario.

    Returns
    -------
    dict[str, pd.DataFrame]
        Claves: 'inventario', 'transacciones', 'feedback'.
    """
    return {
        "inventario": load_inventario(),
        "transacciones": load_transacciones(),
        "feedback": load_feedback(),
    }


if __name__ == "__main__":
    # Prueba rápida de carga al ejecutar el módulo directamente
    datasets = load_all_datasets()
    for nombre, df in datasets.items():
        print(f"{nombre}: {df.shape[0]} filas, {df.shape[1]} columnas")