import numpy as np
from sklearn.utils import class_weight

def calculate_class_weights(train_data):
    # Get class labels for class weight calculation
    train_labels = np.concatenate([y for _, y in train_data], axis=0)
    train_labels_indices = np.argmax(train_labels, axis=1)

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(train_labels_indices), 
        y=train_labels_indices
    )

    # Map class weights to a dictionary
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    return class_weight_dict