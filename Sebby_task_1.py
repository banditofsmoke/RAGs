import warnings
warnings.filterwarnings("ignore")
import os
import glob
import hashlib
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch
from typing import List, Tuple, Union, Dict
import nltk
from nltk.tokenize import sent_tokenize
import logging
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import concurrent.futures
import json
from functools import lru_cache

nltk.download('punkt', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Load configuration from a JSON file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

EMBEDDING_MODEL = config['EMBEDDING_MODEL']
RAG_BATCH_SIZE = config['RAG_BATCH_SIZE']
CHUNK_SIZE = config['CHUNK_SIZE']
CHUNK_OVERLAP = config['CHUNK_OVERLAP']

class DocumentProcessor:
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.conn = self.get_db_connection()
        self.create_tables()
        self.hf = self.get_huggingface_model(EMBEDDING_MODEL)

    def get_db_connection(self):
        return psycopg2.connect(**self.db_config)

    def create_tables(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        source TEXT NOT NULL,
                        metadata JSONB
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id TEXT PRIMARY KEY,
                        document_id TEXT REFERENCES documents(id),
                        content TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        embedding VECTOR(384)
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops)
                """)
            self.conn.commit()
            logging.info("Database tables created successfully")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error creating tables: {str(e)}")
            raise

    @staticmethod
    def generate_hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def chunk_document(self, document: str, metadata: Dict) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.create_documents([document], [metadata])
        return chunks

    def add_document_and_chunks(self, document: str, source: str, metadata: Dict):
        try:
            document_id = self.generate_hash(document)
            chunks = self.chunk_document(document, metadata)

            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (id, content, source, metadata) VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    (document_id, document, source, json.dumps(metadata))
                )
                chunk_data = [
                    (self.generate_hash(chunk.page_content), document_id, chunk.page_content, i, self.hf.encode(chunk.page_content).tolist())
                    for i, chunk in enumerate(chunks)
                ]
                execute_batch(cur, """
                    INSERT INTO chunks (id, document_id, content, chunk_index, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, chunk_data)
            self.conn.commit()
            logging.info(f"Document {document_id} and its chunks added successfully")
            return document_id
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error adding document and chunks: {str(e)}")
            raise

    def get_chunk(self, id: str, is_document_id: bool = False) -> Union[Tuple[str, str, int], List[Tuple[str, str, int]]]:
        try:
            with self.conn.cursor() as cur:
                if is_document_id:
                    cur.execute("SELECT id, content, chunk_index FROM chunks WHERE document_id = %s ORDER BY chunk_index", (id,))
                    return cur.fetchall()
                else:
                    cur.execute("SELECT document_id, content, chunk_index FROM chunks WHERE id = %s", (id,))
                    return cur.fetchone()
        except Exception as e:
            logging.error(f"Error retrieving chunk: {str(e)}")
            raise

    @lru_cache(maxsize=1)
    def get_huggingface_model(self, model_name):
        return SentenceTransformer(model_name)

    @staticmethod
    def get_records_manager(namespace):
        return SQLRecordManager(namespace=namespace, db_url="sqlite:///.cache/sql_record_manager.db")

    @staticmethod
    def extract_text_from_pdf(pdf_path):
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def process_file(self, file_path: str) -> List[Document]:
        if file_path.endswith('.pdf'):
            text = self.extract_text_from_pdf(file_path)
        elif file_path.endswith('.md'):
            with open(file_path, 'r') as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        return self.chunk_document(text, {"source": file_path})

    def build_rag(self, company_name: str):
        if not os.path.exists("./.cache"):
            os.makedirs("./.cache")

        namespace = f"rags-docs-{company_name}".replace(" ","").replace("-","")
        rag_folder = f"./.cache/{company_name}/chroma/docs/"
        if not os.path.exists(rag_folder):
            os.makedirs(rag_folder)

        files = glob.glob(f"./data/{company_name}/**/*.pdf", recursive=True)
        files += glob.glob(f"./.cache/{company_name}/**/*.md", recursive=True)
        log.info(f"Found {len(files)} files for {company_name}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_file = {executor.submit(self.process_file, file): file for file in files}
            all_docs = []
            for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(files), desc=f"Processing files of {company_name}"):
                file = future_to_file[future]
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                    log.info(f"Processed {len(docs)} chunks from {file}")
                except Exception as exc:
                    log.error(f'{file} generated an exception: {exc}')

        record_manager = self.get_records_manager(namespace)
        chroma = Chroma(namespace, embedding_function=self.hf, persist_directory=rag_folder)
        log.info(f"Indexing {len(all_docs)} chunks from {len(files)} files about {company_name}")

        for i in tqdm(range(0, len(all_docs), RAG_BATCH_SIZE), desc=f"Indexing {company_name} documents", unit="Batch"):
            docs = all_docs[i:i+RAG_BATCH_SIZE]
            indexing = index(
                docs,
                record_manager,
                chroma,
                cleanup='incremental',
                source_id_key="source",
            )
            log.info(f"Indexed {indexing} documents for {company_name}")

            for doc in docs:
                self.add_document_and_chunks(doc.page_content, doc.metadata['source'], doc.metadata)

    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_embedding = self.hf.encode(query).tolist()
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT content, 1 - (embedding <=> %s) AS similarity
                FROM chunks
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_embedding, top_k))
            results = cur.fetchall()
        return results

    def close(self):
        self.conn.close()
        logging.info("Database connection closed")

# Usage example:
if __name__ == "__main__":
    processor = DocumentProcessor(config['DB_CONFIG'])

    try:
        # Build RAG for a company
        processor.build_rag("ExampleCompany")

        # Example of semantic search
        query = "What is the company's main product?"
        results = processor.semantic_search(query)
        print(f"Top results for query '{query}':")
        for content, similarity in results:
            print(f"Similarity: {similarity:.4f}")
            print(f"Content: {content[:100]}...")
            print()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        processor.close()