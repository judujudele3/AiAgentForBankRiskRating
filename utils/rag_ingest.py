# utils/rag_ingest.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from config.embedding_config import get_embeddings
from utils.rag_store import load_faiss_or_none, save_faiss

DOCS_PATH = "./docs"

def load_pdfs(docs_path=DOCS_PATH):
    documents = []
    for filename in os.listdir(docs_path):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(docs_path, filename)
            print(f"Chargement du document : {filepath}")
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"Découpage en {len(chunks)} chunks.")
    return chunks

def ingest_docs_to_faiss():
    embeddings = get_embeddings()
    faiss_index = load_faiss_or_none()

    documents = load_pdfs()
    if not documents:
        print("Aucun document trouvé dans ./docs.")
        return

    chunks = chunk_documents(documents)

    if faiss_index is None:
        print("Création d'un nouvel index FAISS...")
        faiss_index = FAISS.from_documents(chunks, embeddings)
    else:
        print("Ajout de nouveaux documents à l'index existant...")
        faiss_index.add_documents(chunks)

    save_faiss(faiss_index)
    print("Ingestion terminée.")

if __name__ == "__main__":
    ingest_docs_to_faiss()
