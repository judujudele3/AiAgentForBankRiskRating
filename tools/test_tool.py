# tools/test_tool.py

from langchain.tools import tool

@tool
def hello_world(name: str) -> str:
    """Dit bonjour à la personne."""
    return f"Bonjour {name}, je suis un agent intelligent."
