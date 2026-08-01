import os
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(_BASE_DIR, "..", "data", "raw")

INVENTARIO_FILE = "inventario_central_v2.csv"
TRANSACCIONES_FILE = "transacciones_logistica_v2.csv"
FEEDBACK_FILE = "feedback_clientes_v2.csv"


def _read_csv_con_encoding(path: str, nombre: str) -> pd.DataFrame:
    """Prueba varios encodings al leer CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró '{nombre}' en {path}. "
            "Verifica que el archivo esté en data/raw/."
        )

    ultimo_error = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as e:
            ultimo_error = e
            continue

    raise UnicodeDecodeError(
        f"No se pudo leer '{nombre}' con ninguno de los encodings probados "
        f"(utf-8, latin-1, cp1252). Último error: {ultimo_error}"
    )


def load_inventario() -> pd.DataFrame:
    """Carga inventario_central_v2.csv crudo."""
    path = os.path.join(RAW_DIR, INVENTARIO_FILE)
    return _read_csv_con_encoding(path, "inventario")


def load_transacciones() -> pd.DataFrame:
    """Carga transacciones_logistica_v2.csv crudo."""
    path = os.path.join(RAW_DIR, TRANSACCIONES_FILE)
    return _read_csv_con_encoding(path, "transacciones")


def load_feedback() -> pd.DataFrame:
    """Carga feedback_clientes_v2.csv crudo."""
    path = os.path.join(RAW_DIR, FEEDBACK_FILE)
    return _read_csv_con_encoding(path, "feedback")


def load_all_datasets() -> dict:
    """Carga los 3 datasets crudos."""
    return {
        "inventario": load_inventario(),
        "transacciones": load_transacciones(),
        "feedback": load_feedback(),
    }


if __name__ == "__main__":
    datasets = load_all_datasets()
    for nombre, df in datasets.items():
        print(f"{nombre}: {df.shape[0]:,} filas x {df.shape[1]} columnas")
