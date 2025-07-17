# config/llm_config.py
from langchain_ollama import OllamaLLM  # Nouveau wrapper officiel

def get_llm(model_name="deepseek-r1:1.5b-qwen-distill-q4_K_M", temperature=0.0):
    return OllamaLLM(model=model_name, temperature=temperature)
