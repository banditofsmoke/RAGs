import argparse
from scripts.ingest_documents import ingest_documents
from scripts.query_rag import query_rag

def main():
    parser = argparse.ArgumentParser(description="RAG System")
    parser.add_argument('action', choices=['ingest', 'query'], help="Action to perform")
    args = parser.parse_args()

    if args.action == 'ingest':
        ingest_documents()
    elif args.action == 'query':
        query_rag()

if __name__ == "__main__":
    main()