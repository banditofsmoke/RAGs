import unittest
from unittest.mock import patch
import os
from src.llm import LLM

class TestLLM(unittest.TestCase):
    def setUp(self):
        self.llm = LLM()

    @patch.dict(os.environ, {'LLM_MODEL': 'phi3.5'})
    @patch('ollama.generate')
    def test_generate(self, mock_generate):
        mock_generate.return_value = {'response': 'Test response'}
        
        result = self.llm.generate("Test prompt", max_tokens=100)
        
        mock_generate.assert_called_once_with(model='phi3.5', prompt="Test prompt", max_tokens=100)
        self.assertEqual(result, 'Test response')

if __name__ == '__main__':
    unittest.main()