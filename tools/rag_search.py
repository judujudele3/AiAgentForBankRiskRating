# tools/rag_search_tool.py
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
from utils.rag_store import load_faiss_or_none

def rag_search_tool(query: str) -> str:
    """
    Recherche dans la base FAISS.
    Args:
        query (str): La requête de recherche.
    Returns:
        str: Résumé des documents les plus pertinents.
    """
    db = load_faiss_or_none()
    if db is None:
        return "La base FAISS est vide ou introuvable. Veuillez d'abord ingérer des documents avec le tool RAG Upsert."
    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "Aucun résultat trouvé."
    return "\n\n---\n\n".join(d.page_content for d in docs)


