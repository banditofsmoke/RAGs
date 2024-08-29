import unittest
from unittest.mock import Mock
from src.rag import RAG

class TestRAG(unittest.TestCase):
    def setUp(self):
        self.mock_retriever = Mock()
        self.mock_llm = Mock()
        self.rag = RAG(self.mock_retriever, self.mock_llm, max_tokens=100)

    def test_query(self):
        self.mock_retriever.retrieve.return_value = [('Context 1', 0.9), ('Context 2', 0.8)]
        self.mock_llm.generate.return_value = "Generated response"

        result = self.rag.query("Test query")

        self.mock_retriever.retrieve.assert_called_once_with("Test query")
        self.mock_llm.generate.assert_called_once()
        self.assertEqual(result, "Generated response")

if __name__ == '__main__':
    unittest.main()