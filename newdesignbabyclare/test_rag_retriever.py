# test_rag_retriever.py

import unittest
from rag_retriever import retrieve_interactions, generate_contextual_response
from openai import OpenAI

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

class TestRAGRetrieverFunctions(unittest.TestCase):

    def test_retrieve_interactions(self):
        interactions = retrieve_interactions(1)
        self.assertIsInstance(interactions, list)

    def test_generate_contextual_response(self):
        interactions = retrieve_interactions(1)
        response = generate_contextual_response("Hello", "Happy", interactions, client)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

if __name__ == '__main__':
    unittest.main()
