from datasets import Dataset, DatasetDict
from sklearn.utils import shuffle

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