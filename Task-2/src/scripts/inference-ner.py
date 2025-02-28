import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer
import os
from nltk.stem import WordNetLemmatizer
import hydra
import torch
from omegaconf import DictConfig

def predict(text, model, tokenizer, lemmatizer=None):
    # If lemmatizer is provided, lemmatize the text
    if lemmatizer:
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Disable gradient computation for inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted labels by selecting the class with the highest probability
    predictions = torch.argmax(outputs.logits, dim=-1)

    # Convert tensor to NumPy array for easier handling
    predicted_labels = predictions[0].cpu().numpy()

    # Map the predicted labels to their corresponding class names
    label_ids = model.config.id2label

    # Extract predicted animals (excluding 'O' which is for non-entities)
    predicted_animals = [label_ids[label] for label in predicted_labels if label_ids[label] != "O"]

    return predicted_animals


@hydra.main(config_path="../../config", config_name="ner-config.yaml")
def main(cfg: DictConfig):
    config_dir = hydra.utils.get_original_cwd()
    global_cfg = hydra.compose(config_name="config.yaml")
    
    # Define paths
    root_path = os.path.abspath(os.path.join(config_dir, '../..')) 
    path_to_model = os.path.join(root_path, "modele", cfg.model.ner_model_name)

    model = AutoModelForTokenClassification.from_pretrained(path_to_model)
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)

    lemmatizer = WordNetLemmatizer()

    text = ["The puppy are running on the picture, and horses are lying on the beg", "Threre is a cat", "The some spiders under the coach"]
    
    for sentence in text:
        predictions = predict(sentence, model, tokenizer, lemmatizer)
        print(predictions)

if __name__ == "__main__":
    main()
