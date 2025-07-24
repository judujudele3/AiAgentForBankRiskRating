# agent/tool_registry.py

from langchain.agents import Tool
from tools.data_loader import load_dataset
from tools.data_explorer import detect_issues_and_suggest_actions, identify_risks_and_targets
from tools.model_trainer import train_model_tool, evaluate_model_tool
from tools.model_suggester import suggest_models_tool
from tools.rag_search import rag_search_tool
from tools.rag_upsert import rag_upsert_tool


rag_upsert = Tool(
    name="RAG Upsert",
    func=rag_upsert_tool,
    description="Ajoute ou met à jour des documents dans la base FAISS à partir des fichiers fournis (PDF, TXT, ...).",
)

rag_search = Tool(
    name="RAG Search",
    func=rag_search_tool,
    description="Permet de rechercher dans la base FAISS des documents pertinents en fonction de la question donnée.",
)

suggest_models = Tool(
    name="Suggest Suitable Models for Training",
    func=suggest_models_tool,
    description="Propose une liste de modèles de machine learning adaptés à la colonne cible fournie. "
                "Prend en entrée le nom ou l'identifiant de la colonne cible (ex: risque_credit)."
                )

train_model_tool_obj = Tool(
    name="Train Model",
    func=train_model_tool,
    description="Entraîne un modèle ML sur des données stockées dans le registre. Input dict : X_train_id, y_train_id, model_type, params."
)

evaluate_model_tool_obj = Tool(
    name="Evaluate Model",
    func=evaluate_model_tool,
    description="Évalue un modèle ML stocké dans le registre. Input dict : model_id, X_test_id, y_test_id."
)

load_dataset_tool = Tool(
    name="Load Dataset",
    func=load_dataset,
    description="Charge un fichier CSV et retourne un DataFrame pandas. " 
                "Prend en entrée le chemin du fichier CSV."
)

detect_issues_tool = Tool(
    name="Detect Data Issues and Suggest Actions",
    func=detect_issues_and_suggest_actions,
    description="Analyse le dernier DataFrame charger par Load Dataset pour détecter les problèmes comme valeurs manquantes, incohérences, colonnes redondantes, "
                "et propose des actions à effectuer pour préparer les données."
)

identify_risks_tool = Tool(
    name="Identify Risks and Target Columns",
    func=identify_risks_and_targets,
    description="Analyse le dernier DataFrame charger par Load Dataset pour identifier les types de risques possibles, les colonnes associées à ces risques, "
                "et suggère quelles colonnes pourraient servir pour entraîner un modèle de prédiction."
)


def get_tools():
    return [
        load_dataset_tool,
        identify_risks_tool,
        train_model_tool_obj,
        evaluate_model_tool_obj,
        suggest_models,
        rag_search,
        rag_upsert,
        # ajouter d'autres tools ici
    ]


