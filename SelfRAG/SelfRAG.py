# self_rag.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SelfRAG:
    def __init__(self):
        self.documents = {}
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None
        self.generator = AutoModelForCausalLM.from_pretrained("gpt2")
        self.generator_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.classifier = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.classifier_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    def add_documents(self, documents):
        self.documents = documents
        self.document_vectors = self.vectorizer.fit_transform(documents.values())

    def retrieve(self, query):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = similarities.argsort()[-5:][::-1]
        retrieved_docs = [list(self.documents.keys())[i] for i in top_indices]
        
        return self.generate_and_evaluate(query, retrieved_docs)

    def generate_and_evaluate(self, query, retrieved_docs):
        context = " ".join([self.documents[doc_id] for doc_id in retrieved_docs])
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        
        generated_responses = []
        for _ in range(3):  # Generate 3 candidate responses
            input_ids = self.generator_tokenizer.encode(prompt, return_tensors="pt")
            output = self.generator.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
            response = self.generator_tokenizer.decode(output[0], skip_special_tokens=True)
            generated_responses.append(response)
        
        scored_responses = self.evaluate_responses(query, generated_responses)
        best_response = max(scored_responses, key=lambda x: x[1])
        
        return best_response[0]

    def evaluate_responses(self, query, responses):
        scored_responses = []
        for response in responses:
            # Evaluate relevance
            relevance_score = cosine_similarity(
                self.vectorizer.transform([query]),
                self.vectorizer.transform([response])
            )[0][0]
            
            # Evaluate quality
            inputs = self.classifier_tokenizer(response, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.classifier(**inputs)
            quality_score = torch.softmax(outputs.logits, dim=1)[0][1].item()  # Assuming positive sentiment = higher quality
            
            combined_score = (relevance_score + quality_score) / 2
            scored_responses.append((response, combined_score))
        
        return scored_responses