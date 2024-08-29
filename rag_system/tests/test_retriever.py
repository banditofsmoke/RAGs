import unittest
from unittest.mock import Mock
from src.retriever import Retriever

class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.mock_db = Mock()
        self.mock_embedding_model = Mock()
        self.retriever = Retriever(self.mock_db, self.mock_embedding_model, top_k=5)

    def test_retrieve(self):
        self.mock_embedding_model.embed.return_value = [0.1] * 384
        self.mock_db.similarity_search.return_value = [('content', 0.9)]

        result = self.retriever.retrieve("test query")
        
        self.mock_embedding_model.embed.assert_called_once_with("test query")
        self.mock_db.similarity_search.assert_called_once()
        self.assertEqual(result, [('content', 0.9)])

if __name__ == '__main__':
    unittest.main()