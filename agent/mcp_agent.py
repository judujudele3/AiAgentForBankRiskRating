from langchain.agents import initialize_agent, AgentType
from config.llm_config import get_llm
from agent.tool_registry import get_tools
from langchain.prompts import PromptTemplate

def create_mcp_agent():
    llm = get_llm()
    tools = get_tools()

    prompt = PromptTemplate.from_template("""
Tu es un agent intelligent pour l'analyse de données bancaires. Tu as accès à des outils pour charger, explorer et comprendre des données structurées.
Tu peux aussi être amener à répondre à de simples questions.
Dans tous les cas n'hésite pas à recourir aux outils RAG pour améliorer tes réponses.
Utilise le format suivant :

Question: la question à résoudre
Thought: ta réflexion sur ce que tu dois faire
Action: l'outil à utiliser, parmi [{tool_names}]
Action Input: l'entrée à passer à l'outil
Observation: le résultat de l'outil
... (répète Thought/Action/Action Input/Observation autant que nécessaire)
Thought: J’ai maintenant assez d’informations
Final Answer: ta réponse finale à la question
                                          
Voici un exemple pour t'inspirer :
Question: Analyse ce fichier CSV pour détecter des problèmes de qualité des données.

Thought: Pour analyser ce fichier, je dois d'abord le charger.
Action: Load Dataset
Action Input: data/DatasetExampleCrédit.txt
Observation: Le dataset a été chargé avec 250 lignes et 18 colonnes.

Thought: Maintenant je vais analyser les colonnes pour détecter les valeurs manquantes.
Action: Detect Data Issues and Suggest Actions
Action Input: df
Observation: La colonne "revenue" a 15% de valeurs manquantes, la colonne "sector" est catégorielle mais contient des incohérences.

Thought: J’ai maintenant assez d’informations
Final Answer: Le fichier contient des données incohérentes et incomplètes. Il faut traiter les valeurs manquantes dans "revenue" et normaliser la colonne "sector".

Commence ici.

Question: {input}
{agent_scratchpad}
""")

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prompt=prompt,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent
