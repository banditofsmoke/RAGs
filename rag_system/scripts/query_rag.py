import yaml
from src.database import Database
from src.embeddings import EmbeddingModel
from src.retriever import Retriever
from src.llm import LLM
from src.rag import RAG

def query_rag():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    db = Database(config['database'])
    embedding_model = EmbeddingModel(config['embeddings']['model'])
    retriever = Retriever(db, embedding_model, config['retriever']['top_k'])
    llm = LLM(config['llm']['model'])
    rag = RAG(retriever, llm, config['rag']['max_tokens'])

    while True:
        query = input("Enter your query (or 'q' to quit): ")
        if query.lower() == 'q':
            break
        response = rag.query(query)
        print(f"Response: {response}\n")

    db.close()

if __name__ == "__main__":
    query_rag()