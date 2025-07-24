import pandas as pd
from agent.registry import store_data

_loaded_df = None

def load_dataset(path: str) -> str:
    global _loaded_df
    try:
        _loaded_df = pd.read_csv(path)

        # Stocker le dataset complet
        store_data("loaded_dataset", _loaded_df)

        # Stocker chaque colonne individuellement
        for col in _loaded_df.columns:
            store_data(col, _loaded_df[col])

        rows, cols = _loaded_df.shape
        column_info = "\n".join([f"- {col} ({_loaded_df[col].dtype})" for col in _loaded_df.columns])
        return (
            f"✅ Dataset chargé avec succès : {rows} lignes, {cols} colonnes.\n"
            f"Colonnes:\n{column_info}"
        )
    except Exception as e:
        return f"❌ Erreur lors du chargement du fichier : {str(e)}"

def get_loaded_df():
    return _loaded_df