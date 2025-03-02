from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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