# context_compression.py

class ContextCompressor:
    def __init__(self):
        self.summarization_model = None
    
    def load_summarization_model(self, model):
        self.summarization_model = model
    
    def compress_document(self, document, query):
        # Extract relevant parts based on the query
        relevant_parts = self.extract_relevant_parts(document, query)
        # Summarize the relevant parts
        if self.summarization_model:
            return self.summarization_model.summarize(relevant_parts)
        return relevant_parts  # Fallback if no model is available
    
    def extract_relevant_parts(self, document, query):
        # Implement logic to extract parts of the document relevant to the query
        # This is a simplified version
        sentences = document.split('.')
        return '. '.join(sent for sent in sentences if any(word in sent for word in query.split()))