import pandas as pd

# Stockage global simple (peut être amélioré plus tard)
_loaded_df = None

def load_dataset(path: str) -> str:
    """
    Charge un fichier CSV dans un DataFrame global.
    :param path: Chemin vers le fichier CSV.
    :return: Résumé textuel pour le LLM.
    """
    global _loaded_df
    try:
        _loaded_df = pd.read_csv(path)
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