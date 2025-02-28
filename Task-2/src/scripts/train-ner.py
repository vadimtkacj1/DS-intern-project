import pandas as pd
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments, AutoTokenizer, DataCollatorForTokenClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset, DatasetDict
import nltk
import os
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
import hydra
from omegaconf import DictConfig

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None

    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id

            if word_id is None:
                label = -100
            else:
                label = labels[word_id]
            new_labels.append(label)

        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]

            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["words"], truncation=True, padding=True, is_split_into_words=True
    )

    all_labels = examples["ner_tags"]
    new_labels = []
    
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        labels_with_tokens = align_labels_with_tokens(labels, word_ids)
        new_labels.append(labels_with_tokens)

    tokenized_inputs["labels"] = new_labels

    return tokenized_inputs

def prepare_dataset(df,lemmatizer, split_ratio=0.2, seed=42):
    raw_data_dict = {}

    for idx in set(df.Sentence_ID.values):
        sentence = df[df.Sentence_ID == idx]
        words = list(sentence.Words.values)

        raw_data_dict[idx] = {
            'words': [lemmatizer.lemmatize(word) for word in words],
            'original_labels': list(sentence.Labels.values),
            'ner_tags': list(sentence.ner_tags.values)
        }

    data_list = [
        {
            'id': idx,
            'words': data['words'],
            'ner_tags': data['ner_tags']
        }
        for idx, data in raw_data_dict.items()
    ]

    data_list = shuffle(data_list, random_state=seed)

    train_dataset = Dataset.from_dict({k: [d[k] for d in data_list] for k in data_list[0]})

    train_valid_split = train_dataset.train_test_split(test_size=split_ratio, seed=seed)

    return DatasetDict({
        "train": train_valid_split["train"],
        "valid": train_valid_split["test"]
    })

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)

    # Only consider non-masked labels
    valid_indices = labels != -100
    predictions = predictions[valid_indices]
    labels = labels[valid_indices]

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


@hydra.main(config_path="../../config", config_name="ner-config.yaml")
def main(cfg: DictConfig):
    config_dir = hydra.utils.get_original_cwd()
    global_cfg = hydra.compose(config_name="config.yaml")
    
    # Define paths
    root_path = os.path.abspath(os.path.join(config_dir, '../..')) 
    data_path = os.path.join(root_path, "data", "processed")
    path_to_model = os.path.join(root_path, "modele", cfg.model.ner_model_name)
    training_output_dir = os.path.join(root_path, cfg.paths.training_output_dir_NAME)
    logging_dir = os.path.join(root_path, cfg.paths.logging_dir_NAME)

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
        evaluation_strategy=cfg.training.evaluation_strategy,
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
        tokenizer=tokenizer
    )

    trainer.train()

    trainer.save_model(path_to_model)
    print(f"Model saved as {cfg.model.ner_model_name}")


if __name__ == "__main__":
    main()
