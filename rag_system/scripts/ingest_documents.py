import os
import yaml
import logging
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingModel
from src.database import Database

logger = logging.getLogger(__name__)

def ingest_documents():
    logger.info("Starting document ingestion process")

    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return

    doc_processor = DocumentProcessor(
        chunk_size=config['document_processing']['chunk_size'],
        chunk_overlap=config['document_processing']['chunk_overlap']
    )
    embedding_model = EmbeddingModel(config['embeddings']['model'])
    db = Database(config['database'])

    logger.info("Initialized DocumentProcessor, EmbeddingModel, and Database")

    # Update the path to the correct location
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'data', 'raw')
    logger.info(f"Scanning directory for documents: {data_dir}")

    if not os.path.exists(data_dir):
        logger.error(f"Directory does not exist: {data_dir}")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    logger.info(f"Found {len(files)} PDF files in the directory")

    if not files:
        logger.warning("No PDF files found in the directory")
        return

    for file in files:
        file_path = os.path.join(data_dir, file)
        logger.info(f"Processing file: {file_path}")
        try:
            documents = doc_processor.process_file(file_path)
            logger.info(f"File processed, generated {len(documents)} document chunks")

            for i, doc in enumerate(documents):
                logger.debug(f"Processing chunk {i+1}/{len(documents)} for {file_path}")
                db.add_document(doc.metadata['source'], doc.page_content, doc.metadata)
                embedding = embedding_model.embed(doc.page_content)
                db.add_chunks([(doc.metadata['source'], doc.metadata['source'], doc.page_content, embedding.tolist())])

            logger.info(f"Successfully processed and ingested: {file_path}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)

    db.close()
    logger.info("Document ingestion process completed")

if __name__ == "__main__":
    ingest_documents()