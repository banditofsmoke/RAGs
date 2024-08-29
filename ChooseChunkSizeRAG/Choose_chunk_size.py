# choose_chunk_size.py

class ChunkSizeSelector:
    def __init__(self):
        self.documents = {}
    
    def add_document(self, doc_id, content):
        self.documents[doc_id] = content
    
    def estimate_optimal_chunk_size(self, document):
        # Implement logic to determine optimal chunk size
        # This could be based on document length, complexity, etc.
        pass
    
    def chunk_document(self, doc_id, chunk_size):
        content = self.documents[doc_id]
        return [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    
    def process_document(self, doc_id):
        optimal_size = self.estimate_optimal_chunk_size(self.documents[doc_id])
        return self.chunk_document(doc_id, optimal_size)