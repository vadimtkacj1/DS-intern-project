import torch

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