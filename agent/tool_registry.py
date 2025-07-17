# agent/tool_registry.py

from langchain.agents import Tool
from tools.data_loader import load_dataset
from tools.data_explorer import detect_issues_and_suggest_actions, identify_risks_and_targets


load_dataset_tool = Tool(
    name="Load Dataset",
    func=load_dataset,
    description="Charge un fichier CSV et retourne un DataFrame pandas. " 
                "Prend en entrée le chemin du fichier CSV."
)

detect_issues_tool = Tool(
    name="Detect Data Issues and Suggest Actions",
    func=detect_issues_and_suggest_actions,
    description="Analyse un DataFrame pour détecter les problèmes comme valeurs manquantes, incohérences, colonnes redondantes, "
                "et propose des actions à effectuer pour préparer les données."
)

identify_risks_tool = Tool(
    name="Identify Risks and Target Columns",
    func=identify_risks_and_targets,
    description="Analyse un DataFrame pour identifier les types de risques possibles, les colonnes associées à ces risques, "
                "et suggère quelles colonnes pourraient servir pour entraîner un modèle de prédiction."
)


def get_tools():
    return [
        load_dataset_tool,
        detect_issues_tool,
        identify_risks_tool,
        # ajouter d'autres tools ici
    ]
