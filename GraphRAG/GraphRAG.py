# graph_rag.py

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class GraphRAG:
    def __init__(self):
        self.graph = nx.Graph()
        self.documents = {}
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None

    def add_documents(self, documents):
        self.documents = documents
        self.document_vectors = self.vectorizer.fit_transform(documents.values())
        self._build_graph()

    def _build_graph(self):
        for i, (doc_id, content) in enumerate(self.documents.items()):
            self.graph.add_node(doc_id, content=content)
            # Connect similar documents
            similarities = cosine_similarity(self.document_vectors[i], self.document_vectors).flatten()
            top_similar = similarities.argsort()[-6:][::-1][1:]  # Top 5 similar docs (excluding self)
            for j in top_similar:
                other_doc_id = list(self.documents.keys())[j]
                self.graph.add_edge(doc_id, other_doc_id, weight=similarities[j])

    def retrieve(self, query):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        initial_doc_id = list(self.documents.keys())[similarities.argmax()]
        
        results = []
        visited = set()
        to_visit = [(initial_doc_id, similarities[similarities.argmax()])]
        
        while to_visit and len(results) < 10:
            current_doc_id, score = to_visit.pop(0)
            if current_doc_id not in visited:
                visited.add(current_doc_id)
                results.append((current_doc_id, score))
                
                for neighbor in self.graph.neighbors(current_doc_id):
                    if neighbor not in visited:
                        neighbor_score = score * self.graph[current_doc_id][neighbor]['weight']
                        to_visit.append((neighbor, neighbor_score))
                
                to_visit.sort(key=lambda x: x[1], reverse=True)
        
        return results

    def get_context(self, doc_id):
        context = [self.documents[doc_id]]
        for neighbor in self.graph.neighbors(doc_id):
            context.append(self.documents[neighbor])
        return context