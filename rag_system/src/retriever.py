from src.database import Database
from src.embeddings import EmbeddingModel
import logging

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, database, embedding_model, top_k):
        self.database = database
        self.embedding_model = embedding_model
        self.top_k = top_k
        logger.info(f"Retriever initialized with top_k={top_k}")

    def retrieve(self, query):
        try:
            logger.info(f"Retrieving context for query: {query}")
            query_embedding = self.embedding_model.embed(query)
            results = self.database.similarity_search(query_embedding, self.top_k)
            logger.info(f"Retrieved {len(results)} results for the query")
            return [{"content": r[0], "similarity": r[1]} for r in results]
        except Exception as e:
            logger.error(f"Error in retrieve method: {str(e)}")
            raise