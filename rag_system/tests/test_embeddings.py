import unittest
import numpy as np
from src.embeddings import EmbeddingModel

class TestEmbeddingModel(unittest.TestCase):
    def setUp(self):
        self.model = EmbeddingModel('sentence-transformers/all-MiniLM-L6-v2')

    def test_embed(self):
        text = "This is a test sentence."
        embedding = self.model.embed(text)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (384,))  # Check the embedding dimension

if __name__ == '__main__':
    unittest.main()