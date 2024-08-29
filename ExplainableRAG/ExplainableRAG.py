# explainable_retrieval.py

class ExplainableRetrieval:
    def __init__(self):
        self.documents = {}
        self.index = {}
    
    def add_document(self, doc_id, content):
        self.documents[doc_id] = content
        self.index_document(doc_id, content)
    
    def index_document(self, doc_id, content):
        words = content.lower().split()
        for word in words:
            if word not in self.index:
                self.index[word] = set()
            self.index[word].add(doc_id)
    
    def retrieve(self, query):
        query_words = query.lower().split()
        relevant_docs = set.intersection(*[self.index.get(word, set()) for word in query_words])
        return [(doc_id, self.explain_relevance(doc_id, query)) for doc_id in relevant_docs]
    
    def explain_relevance(self, doc_id, query):
        explanation = f"Document {doc_id} is relevant because:\n"
        for word in query.lower().split():
            if word in self.index and doc_id in self.index[word]:
                explanation += f"- It contains the query term '{word}'\n"
        return explanation