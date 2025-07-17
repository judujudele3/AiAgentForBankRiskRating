from agent.tool_registry import get_tools
from langchain.agents import initialize_agent, AgentType
from config.llm_config import get_llm

def create_mcp_agent():
    # 1. Initialiser le LLM local via Ollama
    llm = get_llm()

    # 2. Charger les tools (on en ajoutera plus tard)
    tools = get_tools()

    # 3. Initialiser l'agent LangChain
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent

