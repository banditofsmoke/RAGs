from src.retriever import Retriever
from src.llm import LLM

class RAG:
    def __init__(self, retriever: Retriever, llm: LLM, max_tokens: int):
        self.retriever = retriever
        self.llm = llm
        self.max_tokens = max_tokens

    def query(self, user_query: str):
        context = self.retriever.retrieve(user_query)
        prompt = self._construct_prompt(user_query, context)
        return self.llm.generate(prompt, self.max_tokens)

    def _construct_prompt(self, query: str, context: list):
        context_str = "\n".join([f"Content: {c[0]}\nRelevance: {c[1]}" for c in context])
        return f"""Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer:"""