import pandas as pd
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments, AutoTokenizer, DataCollatorForTokenClassification
import nltk
import os
from nltk.stem import WordNetLemmatizer
import hydra
from omegaconf import DictConfig
from metrics import compute_metrics
from labels_manipulation import tokenize_and_align_labels
from preparation_ner_dataset import prepare_dataset


@hydra.main(config_path="../../config", config_name="ner-config.yaml")
def main(cfg: DictConfig):
    config_dir = hydra.utils.get_original_cwd()
    global_cfg = hydra.compose(config_name="config.yaml")
    
    # Define paths
    root_path = os.path.abspath(os.path.join(config_dir, '../..')) 
    data_path = os.path.join(root_path, "data", "processed")
    path_to_model = os.path.join(root_path, "modele", cfg.model.ner_model_name)
    training_output_dir = os.path.join(root_path, cfg.paths.training_output_dir)
    logging_dir = os.path.join(root_path, cfg.paths.logging_dir_name)

    structured_text_df = pd.read_csv(os.path.join(data_path, cfg.paths.data_file))

    # Create label mappings
    label2id = {label: id for id, label in enumerate(cfg.labels.label_ids)}
    id2label = {id: label for label, id in label2id.items()}

    # Download necessary NLTK data
    nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True)

    raw_data = prepare_dataset(structured_text_df, lemmatizer, seed=global_cfg.SEED)
    tokenized_datasets = raw_data.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_data["train"].column_names,
        fn_kwargs={"tokenizer": tokenizer}
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    label_ids_count = len(cfg.labels.label_ids)
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model.name, num_labels=label_ids_count, id2label=id2label, label2id=label2id
    )

    training_args = training_args = TrainingArguments(
        output_dir=training_output_dir,
        eval_strategy=cfg.training.evaluation_strategy,
        save_strategy=cfg.training.save_strategy,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.epochs,
        weight_decay=cfg.training.weight_decay,
        logging_dir=logging_dir,
        logging_steps=cfg.training.logging_steps,
        report_to=cfg.training.report_to
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer
    )

    trainer.train()

    trainer.save_model(path_to_model)
    print(f"Model saved as {cfg.model.ner_model_name}")


if __name__ == "__main__":
    main()
