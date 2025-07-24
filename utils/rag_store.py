# utils/rag_store.py
import os
from langchain_community.vectorstores import FAISS
from config.embedding_config import get_embeddings

# Chemin où sera stocké l'index FAISS localement
FAISS_INDEX_PATH = "./data/faiss_index"

def get_faiss_index_path() -> str:
    """
    Retourne le chemin absolu du dossier d'index FAISS.
    """
    return os.path.abspath(FAISS_INDEX_PATH)

def load_faiss_or_none() -> FAISS | None:
    """
    Charge la base FAISS locale si elle existe, sinon retourne None.
    """
    path = get_faiss_index_path()
    if not os.path.exists(path):
        print("Index FAISS local non trouvé.")
        return None
    embeddings = get_embeddings()
    try:
        faiss_index = FAISS.load_local(path, embeddings)
        print(f"Index FAISS chargé depuis : {path}")
        return faiss_index
    except Exception as e:
        print(f"Erreur lors du chargement de l'index FAISS : {e}")
        return None

def save_faiss(db: FAISS) -> None:
    """
    Sauvegarde la base FAISS localement.
    """
    path = get_faiss_index_path()
    try:
        db.save_local(path)
        print(f"Index FAISS sauvegardé dans : {path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'index FAISS : {e}")
