# config/llm_config.py

from langchain_community.llms import Ollama

def get_llm(model_name="llama2", temperature=0.2):
    """
    Initialise le modèle LLM via Ollama.

    Args:
        model_name (str): Nom du modèle Ollama (ex: "llama2", "mistral", "phi3", etc.)
        temperature (float): Niveau de créativité du LLM

    Returns:
        Ollama: instance du wrapper LangChain pour Ollama
    """
    return Ollama(model=model_name, temperature=temperature)
