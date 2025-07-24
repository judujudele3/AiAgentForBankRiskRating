# config/llm_config.py
from langchain_ollama import OllamaLLM  # Nouveau wrapper officiel


def get_llm(model_name="mistral", temperature=0.0):
    return OllamaLLM(model=model_name, temperature=temperature)
