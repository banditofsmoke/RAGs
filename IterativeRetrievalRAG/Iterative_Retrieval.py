# iterative_retrieval.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class IterativeRetrieval:
    def __init__(self):
        self.documents = {}
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None
        self.max_iterations = 3

    def add_documents(self, documents):
        self.documents = documents
        self.document_vectors = self.vectorizer.fit_transform(documents.values())

    def retrieve(self, initial_query):
        query = initial_query
        all_results = set()
        for _ in range(self.max_iterations):
            query_vector = self.vectorizer.transform([query])
            iteration_results = self.single_iteration_retrieval(query_vector)
            all_results.update(iteration_results)
            if len(all_results) >= 10:  # Arbitrary threshold
                break
            query = self.generate_follow_up_query(query, iteration_results)
        return list(all_results)[:10]  # Return top 10 unique results

    def single_iteration_retrieval(self, query_vector):
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = similarities.argsort()[-5:][::-1]
        return [list(self.documents.keys())[i] for i in top_indices]

    def generate_follow_up_query(self, original_query, results):
        # In a real system, this could use a language model to generate a follow-up query
        # Here, we'll use a simple keyword extraction approach
        result_texts = [self.documents[doc_id] for doc_id in results]
        combined_text = ' '.join(result_texts)
        word_freq = {}
        for word in combined_text.lower().split():
            if word not in original_query.lower():
                word_freq[word] = word_freq.get(word, 0) + 1
        top_keywords = sorted(word_freq, key=word_freq.get, reverse=True)[:3]
        return f"{original_query} {' '.join(top_keywords)}"