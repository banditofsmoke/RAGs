import unittest
from src.database import Database

class TestDatabase(unittest.TestCase):
    def setUp(self):
        # Use a test database configuration
        self.db = Database({
            'host': 'localhost',
            'port': 5432,
            'user': 'test_user',
            'password': 'test_password',
            'dbname': 'test_rag_db'
        })

    def test_add_document(self):
        self.db.add_document('test_id', 'Test content', {'source': 'test.txt'})
        # You'd need to query the database to verify the insertion

    def test_add_chunks(self):
        chunks = [('chunk_id', 'doc_id', 'Chunk content', [0.1] * 384)]
        self.db.add_chunks(chunks)
        # You'd need to query the database to verify the insertion

    def test_similarity_search(self):
        result = self.db.similarity_search([0.1] * 384, top_k=1)
        self.assertIsInstance(result, list)

    def tearDown(self):
        self.db.close()

if __name__ == '__main__':
    unittest.main()