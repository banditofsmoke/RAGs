import unittest
from src.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

    def test_process_pdf(self):
        # You'll need a sample PDF file for this test
        result = self.processor._process_pdf('path/to/sample.pdf')
        self.assertIsInstance(result, list)
        self.assertTrue(all(hasattr(doc, 'page_content') for doc in result))

    def test_process_markdown(self):
        # You'll need a sample Markdown file for this test
        result = self.processor._process_markdown('path/to/sample.md')
        self.assertIsInstance(result, list)
        self.assertTrue(all(hasattr(doc, 'page_content') for doc in result))

if __name__ == '__main__':
    unittest.main()