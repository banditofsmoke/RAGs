# ensemble_retrieval.py

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

class EnsembleRetrieval:
    def __init__(self):
        self.documents = {}
        self.tfidf_vectorizer = TfidfVectorizer()
        self.count_vectorizer = CountVectorizer()
        self.sentence_transformer = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.tfidf_vectors = None
        self.count_vectors = None
        self.semantic_vectors = None

    def add_documents(self, documents):
        self.documents = documents
        texts = list(documents.values())
        self.tfidf_vectors = self.tfidf_vectorizer.fit_transform(texts)
        self.count_vectors = self.count_vectorizer.fit_transform(texts)
        self.semantic_vectors = self.sentence_transformer.encode(texts)

    def retrieve(self, query):
        tfidf_results = self.tfidf_retrieval(query)
        count_results = self.count_retrieval(query)
        semantic_results = self.semantic_retrieval(query)
        
        ensemble_results = self.combine_results([tfidf_results, count_results, semantic_results])
        return ensemble_results[:10]  # Return top 10 results

    def tfidf_retrieval(self, query):
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_vectors).flatten()
        return self.get_top_results(similarities)

    def count_retrieval(self, query):
        query_vector = self.count_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.count_vectors).flatten()
        return self.get_top_results(similarities)

    def semantic_retrieval(self, query):
        query_vector = self.sentence_transformer.encode([query])
        similarities = cosine_similarity(query_vector, self.semantic_vectors).flatten()
        return self.get_top_results(similarities)

    def get_top_results(self, similarities):
        top_indices = similarities.argsort()[-10:][::-1]
        return [(list(self.documents.keys())[i], similarities[i]) for i in top_indices]

    def combine_results(self, result_lists):
        combined_scores = {}
        for results in result_lists:
            for doc_id, score in results:
                if doc_id not in combined_scores:
                    combined_scores[doc_id] = []
                combined_scores[doc_id].append(score)
        
        # Use reciprocal rank fusion to combine scores
        for doc_id in combined_scores:
            combined_scores[doc_id] = sum(1 / (r + 1) for r in range(len(combined_scores[doc_id])))
        
        return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)