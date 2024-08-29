import logging
import os
import ollama

logger = logging.getLogger(__name__)

class LLM:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv('LLM_MODEL', 'Phi3.5')
        logger.info(f"LLM initialized with model: {self.model_name}")

    def generate(self, prompt: str, max_tokens: int):
        logger.info(f"Generating response for prompt of length {len(prompt)}")
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,
                }
            )
            generated_text = response['response']
            logger.info(f"Generated response of length {len(generated_text)}")
            return generated_text
        except Exception as e:
            logger.error(f"Error in LLM generation: {str(e)}", exc_info=True)
            raise