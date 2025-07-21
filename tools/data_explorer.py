from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.utils import load_prompt, summarize_dataframe
import pandas as pd
from config.llm_config import get_llm




# 🔁 Variable globale contenant le dernier dataset chargé
from tools.data_loader import get_loaded_df  # Assure-toi que cette ligne pointe vers le bon module

def detect_issues_and_suggest_actions(input_str: str) -> str:
    loaded_df = get_loaded_df()
    if loaded_df is None:
        return "❌ Aucun dataset chargé. Veuillez d'abord utiliser `load_dataset()`."

    prompt_text = load_prompt("data_exploration_prompt.txt")
    
    # Préparer les variables attendues par le prompt
    colonnes = list(loaded_df.columns)
    types_colonnes = loaded_df.dtypes.to_dict()
    stats_sommaires = loaded_df.describe().to_string()

    prompt = PromptTemplate(
        input_variables=["colonnes", "stats_sommaires", "types_colonnes"],
        template=prompt_text
    )
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(
        colonnes=colonnes,
        stats_sommaires=stats_sommaires, 
        types_colonnes=types_colonnes
    )

def identify_risks_and_targets(input_str: str) -> str:
    loaded_df = get_loaded_df()
    if loaded_df is None:
        return "❌ Aucun dataset chargé. Veuillez d'abord utiliser `load_dataset()`."

    prompt_text = load_prompt("risk_identification_prompt.txt")
    
    # Préparer les variables attendues par le prompt
    colonnes = list(loaded_df.columns)
    types_colonnes = {col: str(dtype) for col, dtype in loaded_df.dtypes.items()}

    prompt = PromptTemplate(
        input_variables=["colonnes", "types_colonnes"],
        template=prompt_text
    )
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(
        colonnes=colonnes,
        types_colonnes=types_colonnes
    )


def test_loaded_df():
    loaded_df = get_loaded_df()
    print("[DEBUG data_explorer] loaded_df is:", loaded_df)