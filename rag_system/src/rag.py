import logging
from src.retriever import Retriever
from src.llm import LLM

logger = logging.getLogger(__name__)

class RAG:
    def __init__(self, retriever: Retriever, llm: LLM, max_tokens: int):
        self.retriever = retriever
        self.llm = llm
        self.max_tokens = max_tokens
        logger.info(f"RAG initialized with max_tokens={max_tokens}")

    def query(self, user_query: str):
        logger.info(f"Processing RAG query: {user_query}")
        try:
            context = self.retriever.retrieve(user_query)
            prompt = self._construct_prompt(user_query, context)
            logger.debug(f"Generated prompt of length {len(prompt)}")
            response = self.llm.generate(prompt, self.max_tokens)
            logger.info("RAG query processed successfully")
            return response
        except Exception as e:
            logger.error(f"Error processing RAG query: {str(e)}", exc_info=True)
            raise

    def _construct_prompt(self, query: str, context: list):
        if not context:
            logger.warning("No relevant context found for the query")
            return f"Query: {query}\nAnswer: I don't have enough information to answer this query accurately."
        
        context_str = "\n".join([f"Content: {c['content']}\nRelevance: {c['similarity']}" for c in context])
        prompt = f"""Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer:"""
        return prompt