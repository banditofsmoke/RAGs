import os
import yaml
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingModel
from src.database import Database

def ingest_documents():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    doc_processor = DocumentProcessor(
        chunk_size=config['document_processing']['chunk_size'],
        chunk_overlap=config['document_processing']['chunk_overlap']
    )
    embedding_model = EmbeddingModel(config['embeddings']['model'])
    db = Database(config['database'])

    data_dir = 'data/raw'
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                documents = doc_processor.process_file(file_path)
                for doc in documents:
                    db.add_document(doc.metadata['source'], doc.page_content, doc.metadata)
                    embedding = embedding_model.embed(doc.page_content)
                    db.add_chunks([(doc.metadata['source'], doc.metadata['source'], doc.page_content, embedding)])
                print(f"Processed and ingested: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

    db.close()

if __name__ == "__main__":
    ingest_documents()