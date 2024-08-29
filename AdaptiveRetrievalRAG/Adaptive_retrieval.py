# adaptive_retrieval.py

from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class QueryType(Enum):
    FACTUAL = 1
    ANALYTICAL = 2
    CONTEXTUAL = 3

class AdaptiveRetrieval:
    def __init__(self):
        self.documents = {}
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None
        self.query_classifier = None

    def add_documents(self, documents):
        self.documents = documents
        self.document_vectors = self.vectorizer.fit_transform(documents.values())

    def load_query_classifier(self, classifier):
        self.query_classifier = classifier

    def classify_query(self, query):
        if self.query_classifier:
            return self.query_classifier.predict([query])[0]
        # Fallback classification based on query characteristics
        if any(word in query.lower() for word in ['what', 'who', 'when', 'where']):
            return QueryType.FACTUAL
        elif any(word in query.lower() for word in ['why', 'how', 'analyze']):
            return QueryType.ANALYTICAL
        else:
            return QueryType.CONTEXTUAL

    def retrieve(self, query):
        query_type = self.classify_query(query)
        query_vector = self.vectorizer.transform([query])
        
        if query_type == QueryType.FACTUAL:
            return self.factual_retrieval(query_vector)
        elif query_type == QueryType.ANALYTICAL:
            return self.analytical_retrieval(query_vector)
        else:
            return self.contextual_retrieval(query_vector)

    def factual_retrieval(self, query_vector):
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = similarities.argsort()[-5:][::-1]
        return [list(self.documents.keys())[i] for i in top_indices]

    def analytical_retrieval(self, query_vector):
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        weighted_similarities = similarities * np.array([len(doc.split()) for doc in self.documents.values()])
        top_indices = weighted_similarities.argsort()[-5:][::-1]
        return [list(self.documents.keys())[i] for i in top_indices]

    def contextual_retrieval(self, query_vector):
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        diverse_indices = []
        for _ in range(5):
            if len(diverse_indices) == 0:
                diverse_indices.append(similarities.argmax())
            else:
                remaining_indices = set(range(len(similarities))) - set(diverse_indices)
                next_index = max(remaining_indices, key=lambda i: similarities[i] * min(1 - cosine_similarity(
                    self.document_vectors[i], self.document_vectors[diverse_indices]).flatten()))
                diverse_indices.append(next_index)
        return [list(self.documents.keys())[i] for i in diverse_indices]