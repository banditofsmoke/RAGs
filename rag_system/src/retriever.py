from src.database import Database
from src.embeddings import EmbeddingModel

class Retriever:
    def __init__(self, database: Database, embedding_model: EmbeddingModel, top_k: int):
        self.database = database
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(self, query: str):
        query_embedding = self.embedding_model.embed(query)
        return self.database.similarity_search(query_embedding, self.top_k)