# main.py

from agent.mcp_agent import create_mcp_agent
from langchain.agents import AgentExecutor

def main():
    agent: AgentExecutor = create_mcp_agent()

    print("\n================ Lancement de l'agent MCP ================\n")

    # Étape 1 : charger le dataset
    dataset_path = "C:\\Users\\amine\\OneDrive\\Documents\\Projects\\AIAgentForBankRiskRating\\AiAgentForBankRiskRating\\data\\DatasetEntrainement.txt"
    print(f"[INFO] Dataset cible : {dataset_path}")

    query = (
        f"Voila le chemin du fichier : {dataset_path}. Voici un fichier CSV avec des données de crédit. Charge-le, RAG, analyse les données qualitativement, RAG, identifie une colonne cible, RAG, puis suggère des modèles de machine learning adaptés, RAG et entraîne un modèle sur ces données. Donc après chaque étape tu révise ta réponse avec le RAG"
    )

    # Appel à l’agent
    result = agent.invoke({"input": query})

    print("\n================== Résultat de l'agent ==================\n")
    print(result)

if __name__ == "__main__":
    main()


