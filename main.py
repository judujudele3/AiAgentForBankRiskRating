# main.py

from agent.mcp_agent import create_mcp_agent
from langchain.agents import AgentExecutor

def main():
    agent: AgentExecutor = create_mcp_agent()

    print("\n================ Lancement de l'agent MCP ================\n")

    # Étape 1 : charger le dataset
    dataset_path = "C:\\Users\\amine\\OneDrive\\Documents\\Projects\\AIAgentForBankRiskRating\\AiAgentForBankRiskRating\\data\\DatasetExampleCredit.txt"
    print(f"[INFO] Dataset cible : {dataset_path}")

    query = (
        f"Voila le chemin du fichier : {dataset_path}. "
    )

    # Appel à l’agent
    result = agent.invoke({"input": query})

    print("\n================== Résultat de l'agent ==================\n")
    print(result)

if __name__ == "__main__":
    main()


