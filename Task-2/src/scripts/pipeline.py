import os
import torch
import tensorflow as tf
from transformers import AutoModelForTokenClassification, AutoTokenizer
from nltk.stem import WordNetLemmatizer
from omegaconf import DictConfig
import hydra
from sklearn.pipeline import Pipeline
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

class ImageProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, model, image_size, class_names):
        self.model = model
        self.image_size = image_size
        self.class_names = class_names

    def fit(self, X, y=None):
        return self

    def transform(self, X): 
        pred, image_path = X
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(self.image_size, self.image_size))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0) / 255.0 
        
        predictions = self.model.predict(image_array)
        predicted_label = np.argmax(predictions, axis=-1)[0]
        
        predicted_animal = self.class_names[predicted_label]
        print([pred, predicted_animal] )
        return [pred, predicted_animal] 

class FinalPredictor(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        text_predictions, image_predictions = X

        return [image_predictions in text_predictions] 

@hydra.main(config_path="../../config", config_name="ner-config.yaml")
def main(cfg: DictConfig):
    config_dir = hydra.utils.get_original_cwd()
    global_cfg = hydra.compose(config_name="config.yaml")
    
    # Define paths
    root_path = os.path.abspath(os.path.join(config_dir, '../..'))
    path_to_model = os.path.join(root_path, "models", cfg.model.ner_model_name)
    data_path = os.path.join(root_path, "data", "raw-test")
    models_path = os.path.join(root_path, "models")
    
    # Load NER model and tokenizer
    model_ner = AutoModelForTokenClassification.from_pretrained(path_to_model)
    tokenizer_ner = AutoTokenizer.from_pretrained(path_to_model)
    lemmatizer = WordNetLemmatizer()

    # Load Image classification model
    image_model_name = f"model_{global_cfg.image_size}.h5"
    image_model_path = os.path.join(models_path, image_model_name)
    model_image = tf.keras.models.load_model(image_model_path)

    # Text input 
    text_input = "Is the horse on the picture" 
    image_input = os.path.join(data_path, "horse.jpg") 

    # Create the pipeline
    pipeline = Pipeline([
        ('text_processor', TextProcessor(model_ner, tokenizer_ner, lemmatizer)),
        ('image_processor', ImageProcessor(model_image, global_cfg.image_size, list(global_cfg.LABELS))),
        ('final_predictor', FinalPredictor())
    ])

    # Apply the pipeline to the input
    result = pipeline.transform([text_input, image_input])

    print(f"Do the text and image match? {result[0]}") 

if __name__ == "__main__":
    main()
