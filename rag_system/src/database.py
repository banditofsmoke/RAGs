import psycopg2
from psycopg2.extras import execute_values
import json

class Database:
    def __init__(self, config: dict):
        self.conn = psycopg2.connect(**config)
        self.create_tables()

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
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops)")
        self.conn.commit()

    def add_document(self, doc_id: str, content: str, metadata: dict):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (id, content, metadata) VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (doc_id, content, json.dumps(metadata))
            )
        self.conn.commit()

    def add_chunks(self, chunks: list):
        with self.conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO chunks (id, document_id, content, embedding)
                VALUES %s
                ON CONFLICT (id) DO NOTHING
            """, chunks)
        self.conn.commit()

    def similarity_search(self, query_embedding, top_k: int = 5):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT content, 1 - (embedding <=> %s) AS similarity
                FROM chunks
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_embedding, top_k))
            return cur.fetchall()

    def close(self):
        self.conn.close()