# main.py

from agent.mcp_agent import create_mcp_agent
from langchain.agents import AgentExecutor

def main():
    agent: AgentExecutor = create_mcp_agent()

    print("\n================ Lancement de l'agent MCP ================\n")

    # Étape 1 : charger le dataset
    dataset_path = "data/DatasetExampleCrédit.txt"
    print(f"[INFO] Dataset cible : {dataset_path}")

    query = (
        f"Charge le fichier CSV situé à ce chemin : {dataset_path}. "
        f"Ensuite, analyse les données pour détecter des problèmes éventuels, "
        f"et propose des étapes de nettoyage ou préparation."
        f"Puis identifie les risques présents dans les données, les colonnes associées, "
        f"et celles qui peuvent servir comme cible pour un modèle de scoring de crédit."
    )

    # Appel à l’agent
    result = agent.run(query)

    print("\n================== Résultat de l'agent ==================\n")
    print(result)

if __name__ == "__main__":
    main()
