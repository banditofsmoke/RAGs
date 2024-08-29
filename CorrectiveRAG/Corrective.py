# corrective_rag.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

class CorrectiveRAG:
    def __init__(self):
        self.documents = {}
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None
        self.generator = AutoModelForCausalLM.from_pretrained("gpt2-medium")
        self.generator_tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
        self.qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        self.nlp = spacy.load("en_core_web_sm")

    def add_documents(self, documents):
        self.documents = documents
        self.document_vectors = self.vectorizer.fit_transform(documents.values())

    def retrieve(self, query):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = similarities.argsort()[-5:][::-1]
        retrieved_docs = [list(self.documents.keys())[i] for i in top_indices]
        
        return self.generate_and_correct(query, retrieved_docs)

    def generate_and_correct(self, query, retrieved_docs):
        context = " ".join([self.documents[doc_id] for doc_id in retrieved_docs])
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        
        # Generate initial response
        input_ids = self.generator_tokenizer.encode(prompt, return_tensors="pt")
        output = self.generator.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
        initial_response = self.generator_tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract facts from the initial response
        facts = self.extract_facts(initial_response)
        
        # Verify and correct facts
        corrected_facts = self.verify_and_correct_facts(query, context, facts)
        
        # Regenerate response with corrected facts
        corrected_prompt = f"Context: {context}\nQuestion: {query}\nCorrected facts: {' '.join(corrected_facts)}\nImproved Answer:"
        input_ids = self.generator_tokenizer.encode(corrected_prompt, return_tensors="pt")
        output = self.generator.generate(input_ids, max_length=250, num_return_sequences=1, no_repeat_ngram_size=2)
        corrected_response = self.generator_tokenizer.decode(output[0], skip_special_tokens=True)
        
        return corrected_response

    def extract_facts(self, text):
        doc = self.nlp(text)
        facts = []
        for sent in doc.sents:
            if any(token.dep_ in ["nsubj", "attr"] for token in sent):
                facts.append(sent.text)
        return facts

    def verify_and_correct_facts(self, query, context, facts):
        corrected_facts = []
        for fact in facts:
            # Use QA model to verify the fact
            inputs = self.qa_tokenizer(question=fact, context=context, return_tensors="pt")
            with torch.no_grad():
                outputs = self.qa_model(**inputs)
            
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer = self.qa_tokenizer.convert_tokens_to_string(self.qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
            
            if answer.lower() in fact.lower():
                corrected_facts.append(fact)
            else:
                # If the fact is not verified, try to correct it
                corrected_fact = self.correct_fact(fact, answer, context)
                corrected_facts.append(corrected_fact)
        
        return corrected_facts

    def correct_fact(self, original_fact, verified_answer, context):
        # Simple correction: replace the incorrect part with the verified answer
        doc = self.nlp(original_fact)
        for ent in doc.ents:
            if ent.text.lower() not in verified_answer.lower():
                original_fact = original_fact.replace(ent.text, verified_answer)
                break
        
        return original_fact

    def evaluate_correction(self, original_response, corrected_response, query, context):
        # Evaluate the quality of correction
        original_relevance = cosine_similarity(
            self.vectorizer.transform([original_response]),
            self.vectorizer.transform([context])
        )[0][0]
        
        corrected_relevance = cosine_similarity(
            self.vectorizer.transform([corrected_response]),
            self.vectorizer.transform([context])
        )[0][0]
        
        improvement = corrected_relevance - original_relevance
        
        return {
            "original_relevance": original_relevance,
            "corrected_relevance": corrected_relevance,
            "improvement": improvement
        }