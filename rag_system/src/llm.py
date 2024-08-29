import os
import ollama

class LLM:
    def __init__(self):
        self.model_name = os.getenv('LLM_MODEL', 'phi3.5')

    def generate(self, prompt: str, max_tokens: int):
        response = ollama.generate(model=self.model_name, prompt=prompt, max_tokens=max_tokens)
        return response['response']