import torch
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class TextProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, model, tokenizer, lemmatizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.lemmatizer = lemmatizer if lemmatizer else None
        self.label_ids = model.config.id2label

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        predictions = []
        text, image_path = X
        if isinstance(text, str): 
            if self.lemmatizer:
                text = " ".join([self.lemmatizer.lemmatize(word) for word in text.split()])
        elif isinstance(text, list): 
            text = " ".join(text)

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_labels = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
        predicted_animals = [self.label_ids[label] for label in predicted_labels if self.label_ids[label] != "O"]
        predictions.append(predicted_animals)

        return [predictions[0], image_path]

    def __sklearn_clone__(self):
       return TextProcessor(self.model, self.tokenizer, self.lemmatizer, self.label_ids)