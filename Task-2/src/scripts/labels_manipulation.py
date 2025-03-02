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
