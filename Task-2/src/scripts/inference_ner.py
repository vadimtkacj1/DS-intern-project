from transformers import AutoModelForTokenClassification, AutoTokenizer
import os
from nltk.stem import WordNetLemmatizer
import hydra
from omegaconf import DictConfig
from predict_ner import predict


@hydra.main(config_path="../../config", config_name="ner-config.yaml")
def main(cfg: DictConfig):
    config_dir = hydra.utils.get_original_cwd()

    # Define paths
    root_path = os.path.abspath(os.path.join(config_dir, '../..')) 
    path_to_model = os.path.join(root_path, "models", cfg.model.ner_model_name)

    model = AutoModelForTokenClassification.from_pretrained(path_to_model)
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)

    lemmatizer = WordNetLemmatizer()

    text = ["The puppy are running on the picture, and horses are lying on the beg", "Threre is a cat", "The some spiders under the coach"]
    
    for sentence in text:
        predictions = predict(sentence, model, tokenizer, lemmatizer)
        print(sentence, predictions)

if __name__ == "__main__":
    main()
