import os
import pandas as pd

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

def load_prompt(filename: str) -> str:
    """Charge un prompt depuis le dossier prompts."""
    path = os.path.join(PROMPT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def summarize_dataframe(df: pd.DataFrame) -> str:
    """Crée un résumé simple du DataFrame pour donner un contexte textuel au LLM."""
    lines = []
    lines.append(f"Le dataset contient {df.shape[0]} lignes et {df.shape[1]} colonnes.")
    lines.append("Les colonnes sont :")
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = df[col].isna().sum()
        lines.append(f"- {col} (type: {dtype}, valeurs manquantes: {missing})")
    return "\n".join(lines)
