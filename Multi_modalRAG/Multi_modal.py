# multi_modal.py

import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np

class MultiModalRetrieval:
    def __init__(self):
        self.documents = {}
        self.images = {}
        self.text_model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.image_model = resnet50(pretrained=True)
        self.image_model.fc = torch.nn.Identity()  # Remove the final fully connected layer
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def add_document(self, doc_id, text, image_path):
        self.documents[doc_id] = text
        self.images[doc_id] = image_path

    def encode_text(self, text):
        inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def encode_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self.image_model(image_tensor)
        return features.squeeze().numpy()

    def retrieve(self, query_text, query_image_path=None):
        query_text_vector = self.encode_text(query_text)
        
        results = []
        for doc_id in self.documents:
            doc_text_vector = self.encode_text(self.documents[doc_id])
            doc_image_vector = self.encode_image(self.images[doc_id])
            
            text_similarity = np.dot(query_text_vector, doc_text_vector) / (np.linalg.norm(query_text_vector) * np.linalg.norm(doc_text_vector))
            
            if query_image_path:
                query_image_vector = self.encode_image(query_image_path)
                image_similarity = np.dot(query_image_vector, doc_image_vector) / (np.linalg.norm(query_image_vector) * np.linalg.norm(doc_image_vector))
                combined_similarity = (text_similarity + image_similarity) / 2
            else:
                combined_similarity = text_similarity
            
            results.append((doc_id, combined_similarity))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:10]