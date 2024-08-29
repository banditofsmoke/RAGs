# raptor.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RAPTOR:
    def __init__(self):
        self.documents = {}
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None
        self.summarizer = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    def add_documents(self, documents):
        self.documents = documents
        self.document_vectors = self.vectorizer.fit_transform(documents.values())

    def retrieve(self, query):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = similarities.argsort()[-10:][::-1]
        
        results = []
        for i in top_indices:
            doc_id = list(self.documents.keys())[i]
            summary = self.summarize(self.documents[doc_id])
            results.append((doc_id, summary, similarities[i]))
        
        return self.recursive_summarization(results, query)

    def summarize(self, text):
        inputs = self.tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = self.summarizer.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def recursive_summarization(self, results, query, depth=0, max_depth=3):
        if depth == max_depth or len(results) == 1:
            return results

        combined_summary = " ".join([summary for _, summary, _ in results])
        new_summary = self.summarize(combined_summary)
        
        new_results = []
        for doc_id, _, score in results:
            similarity = cosine_similarity(
                self.vectorizer.transform([new_summary]),
                self.vectorizer.transform([self.documents[doc_id]])
            )[0][0]
            new_results.append((doc_id, new_summary, score * similarity))
        
        new_results.sort(key=lambda x: x[2], reverse=True)
        return self.recursive_summarization(new_results[:len(new_results)//2], query, depth+1, max_depth)