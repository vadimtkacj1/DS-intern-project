import os
import tensorflow as tf
from transformers import AutoModelForTokenClassification, AutoTokenizer
from nltk.stem import WordNetLemmatizer
from omegaconf import DictConfig
import hydra
from sklearn.pipeline import Pipeline
from ImageProcessor import ImageProcessor
from TextProcessor import TextProcessor
from UnifyingProcessor import UnifyingProcessor

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

    # form model name, all models start with name model underscope and size of trained model
    image_model_name = f"model_{global_cfg.image_size}.h5"

    # Load Image classification model
    image_model_path = os.path.join(models_path, image_model_name)
    model_image = tf.keras.models.load_model(image_model_path)


    # Create the pipeline
    pipeline = Pipeline([
        ('text_processor', TextProcessor(model_ner, tokenizer_ner, lemmatizer)),
        ('image_processor', ImageProcessor(model_image, global_cfg.image_size, list(global_cfg.LABELS))),
        ('unifying_processor', UnifyingProcessor())
    ])

    # Text input 
    text_input = "Is the horse on the picture" 
    image_input = os.path.join(data_path, "horse.jpg") 
    
    # Apply the pipeline to the input
    result = pipeline.transform([text_input, image_input])

    print(f"Do the text and image match? {result[0]}") 

if __name__ == "__main__":
    main()
