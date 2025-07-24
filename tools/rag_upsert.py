# tools/rag_upsert_tool.py
import os
import glob
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
from utils.rag_store import get_embeddings, load_faiss_or_none, save_faiss

# Dossiers autorisés pour ingestion (sécurité)
ALLOWED_DIRS = ["./docs", "./data"]

def _is_path_allowed(path: str) -> bool:
    abs_path = os.path.abspath(path)
    return any(os.path.abspath(d) in abs_path for d in ALLOWED_DIRS)

def _expand_input_to_paths(input_str: str) -> List[str]:
    """
    Transforme une chaîne de chemins en liste.
    Supporte plusieurs chemins séparés par virgule ou saut de ligne.
    Supporte aussi un glob (ex: glob:docs/*.pdf).
    """
    input_str = input_str.strip()
    if input_str.startswith("glob:"):
        pattern = input_str[len("glob:"):].strip()
        return glob.glob(pattern)
    # split par virgule et newline
    parts = [p.strip() for p in input_str.replace("\n", ",").split(",")]
    return [p for p in parts if p]

def _load_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    elif ext in {".txt", ".md"}:
        return TextLoader(path, encoding="utf-8").load()
    else:
        raise ValueError(f"Extension non supportée: {ext}")

def rag_upsert_tool(paths_str: str) -> str:
    """
    Ingestion / mise à jour de la base FAISS avec les fichiers fournis.
    Args:
        paths_str (str): chemins des fichiers à ingérer (csv, glob, multiples chemins séparés).
    Returns:
        str: message de statut.
    """
    paths = _expand_input_to_paths(paths_str)
    if not paths:
        return "Aucun fichier valide détecté."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    embeddings = get_embeddings()
    db = load_faiss_or_none()

    all_chunks = []
    skipped = []
    for path in paths:
        if not os.path.exists(path):
            skipped.append((path, "introuvable"))
            continue
        if not _is_path_allowed(path):
            skipped.append((path, "chemin non autorisé"))
            continue
        try:
            docs = _load_file(path)
            all_chunks.extend(splitter.split_documents(docs))
        except Exception as e:
            skipped.append((path, f"erreur: {e}"))

    if not all_chunks and not db:
        return "Aucun chunk à indexer et la base FAISS n'existe pas encore."

    if db is None:
        db = FAISS.from_documents(all_chunks, embeddings)
    else:
        db.add_documents(all_chunks)

    save_faiss(db)
    return f"Ingestion terminée. {len(all_chunks)} chunks ajoutés. Fichiers ignorés: {skipped if skipped else 'aucun'}."

