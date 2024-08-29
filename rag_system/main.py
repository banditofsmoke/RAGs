import argparse
import logging
from scripts.ingest_documents import ingest_documents
from scripts.query_rag import query_rag

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="RAG System CLI")
    parser.add_argument('action', choices=['ingest', 'query'], help='Action to perform')
    args = parser.parse_args()

    logger.info(f"Starting RAG system with action: {args.action}")

    if args.action == 'ingest':
        logger.info("Initiating document ingestion process")
        ingest_documents()
    elif args.action == 'query':
        logger.info("Initiating query process")
        query_rag()

    logger.info("RAG system operation completed")

if __name__ == "__main__":
    main()