# MNIST Image Classification

## Overview
This project implements an image classification solution using the MNIST dataset. Three different models are used to classify handwritten digits:

1. **Random Forest (RF)**
2. **Feed-Forward Neural Network (NN)**
3. **Convolutional Neural Network (CNN)**

Each model is implemented as a separate class that follows the `MnistClassifierInterface`, ensuring a consistent API for training and prediction. The `MnistClassifier` acts as a wrapper that allows users to select a specific model.


## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repo_url>
cd mnist_classifier
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### **1. Training a Model**
To train a model, run the following command:
```bash
python train.py --algorithm <algorithm_name>
```
Replace `<algorithm_name>` with one of:
- `rf` (Random Forest)
- `nn` (Feed-Forward Neural Network)
- `cnn` (Convolutional Neural Network)

Example:
```bash
python train.py --algorithm cnn
```

### **2. Making Predictions**
To make predictions on test data:
```bash
python predict.py --algorithm <algorithm_name>
```
Example:
```bash
python predict.py --algorithm nn
```

---

## MnistClassifierInterface
The `MnistClassifierInterface` ensures all classifiers implement a common API:
```python
from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass
```

Each classifier implements this interface and follows the same structure.

---

## Demonstration (Jupyter Notebook)
A `demo_notebook.ipynb` is provided with a full example, including:
- Loading the MNIST dataset
- Training each model
- Making predictions
- Evaluating performance

To run the notebook:
```bash
jupyter notebook demo_notebook.ipynb
```

---

## Edge Cases Considered
- Handling of missing or corrupted data
- Performance evaluation using accuracy, confusion matrix, and classification report
- Consistency in input/output format across different models

---

## Dependencies

Install the required packages using:
```bash
pip install -r requirements.txt
```

### **Main Libraries Used:**
- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`
- `torch`
- `matplotlib`
- `seaborn`

---

## Contribution
Feel free to contribute by submitting a pull request or reporting issues.

---

## License
This project is licensed under the MIT License.

