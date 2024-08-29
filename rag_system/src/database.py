import psycopg2
from psycopg2.extras import execute_values
import json
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
import re
import logging

logger = logging.getLogger(__name__)

def addapt_numpy_array(numpy_array):
    return AsIs(tuple(numpy_array))

register_adapter(np.ndarray, addapt_numpy_array)

class Database:
    def __init__(self, config: dict):
        self.conn = psycopg2.connect(**config)
        self.create_tables()
        logger.info("Database connection established and tables created")

    def create_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT REFERENCES documents(id),
                    content TEXT NOT NULL,
                    embedding VECTOR(384)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT UNIQUE NOT NULL,
                    document_id TEXT REFERENCES documents(id),
                    occurrences INT NOT NULL
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops)")
        self.conn.commit()
        logger.info("Database tables created successfully")

    def add_document(self, doc_id: str, content: str, metadata: dict):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (id, content, metadata) VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (doc_id, content, json.dumps(metadata))
            )
            self.add_symbols(doc_id, content, cur)
        self.conn.commit()
        logger.info(f"Document added: {doc_id}")

    def add_symbols(self, doc_id: str, content: str, cur):
        symbols = re.findall(r'\b[A-Z][A-Z0-9]{2,}\b', content)
        symbol_counts = {symbol: symbols.count(symbol) for symbol in set(symbols)}
        for symbol, count in symbol_counts.items():
            cur.execute("""
                INSERT INTO symbols (symbol, document_id, occurrences)
                VALUES (%s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE
                SET occurrences = symbols.occurrences + EXCLUDED.occurrences
            """, (symbol, doc_id, count))
        logger.info(f"Symbols added for document: {doc_id}")

    def add_chunks(self, chunks: list):
        with self.conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO chunks (id, document_id, content, embedding)
                VALUES %s
                ON CONFLICT (id) DO NOTHING
            """, chunks)
        self.conn.commit()
        logger.info(f"Added {len(chunks)} chunks to database")

    def similarity_search(self, query_embedding, top_k: int = 5):
        query_embedding = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT content, 1 - (embedding <=> %s::vector) AS similarity
                    FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, top_k))
                results = cur.fetchall()
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise

    def close(self):
        self.conn.close()
        logger.info("Database connection closed")