# tests/test_rag_with_agent.py
import unittest
from agent.mcp_agent import create_mcp_agent

class TestRAGWithAgent(unittest.TestCase):

    def setUp(self):
        self.agent = create_mcp_agent()

    def test_rag_search_via_agent(self):
        query = "D'après tes connaissances que peux-tu dire sur les risques liés au crédit bancaire ?"
        response = self.agent.invoke({"input": query})
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0, "L'agent doit retourner une réponse non vide.")
        print("\n=== Réponse de l'agent ===\n")
        print(response)

if __name__ == "__main__":
    unittest.main()
