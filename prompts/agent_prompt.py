from langchain.prompts import PromptTemplate

FORMAT_INSTRUCTIONS = """Utilise le format suivant :

Question: la question à résoudre
Thought: ta réflexion sur ce que tu dois faire
Action: l'outil à utiliser, parmi [{tool_names}]
Action Input: l'entrée à passer à l'outil
Observation: le résultat de l'outil
... (répète Thought/Action/Action Input/Observation autant que nécessaire)
Thought: J’ai maintenant assez d’informations
Final Answer: ta réponse finale à la question"""

EXAMPLE = """Question: Analyse ce fichier CSV pour détecter des problèmes de qualité des données.

Thought: Pour analyser ce fichier, je dois d'abord le charger.
Action: Load Dataset
Action Input: data/DatasetExampleCrédit.txt
Observation: Le dataset a été chargé avec 250 lignes et 18 colonnes.

Thought: Maintenant je vais analyser les colonnes pour détecter les valeurs manquantes.
Action: Detect Data Issues and Suggest Actions
Action Input: df
Observation: La colonne "revenue" a 15% de valeurs manquantes, la colonne "sector" est catégorielle mais contient des incohérences.

Thought: J’ai maintenant assez d’informations
Final Answer: Le fichier contient des données incohérentes et incomplètes. Il faut traiter les valeurs manquantes dans "revenue" et normaliser la colonne "sector"."""

def get_agent_prompt():
    PREFIX = """Tu es un agent intelligent pour l'analyse de données bancaires. 
Tu as accès à des outils qui te permettent de charger, explorer et comprendre des données structurées pour identifier les risques 
et préparer les données pour l'entraînement de modèles ML."""

    SUFFIX = """Commence ici. N'oublie pas de réfléchir à chaque étape.

{format_instructions}

Voici un exemple pour t'inspirer :

{example}

Question: {input}
{agent_scratchpad}"""

    full_template = "\n\n".join([PREFIX, SUFFIX])

    return PromptTemplate.from_template(
        template=full_template,
        input_variables=["input", "agent_scratchpad", "format_instructions", "example", "tool_names"]
    )
