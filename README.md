# Project Overview
This project consists of two main tasks:

## Task 1: Image Classification with OOP
This task involves building a machine learning pipeline for the MNIST dataset using three different classification models:
1. **Random Forest (RF)**
2. **Feed-Forward Neural Network (NN)**
3. **Convolutional Neural Network (CNN)**

Each model is implemented as a separate class that follows the `MnistClassifierInterface`, which contains two abstract methods:
- `train()`: Trains the model.
- `predict()`: Makes predictions using the trained model.

Additionally, all three models are encapsulated within a higher-level `MnistClassifier` class, which takes an algorithm name as an input parameter (`cnn`, `rf`, or `nn`) and provides a unified prediction interface.


## Task 2: Named Entity Recognition + Image Classification Pipeline
This task involves building a machine learning pipeline that integrates:
1. **Named Entity Recognition (NER)**: Extracts animal names from text using a transformer-based model (excluding large language models - LLMs).
2. **Image Classification**: Classifies animals in images based on a custom dataset.

The pipeline takes a user-provided text (e.g., *"There is a cow in the picture."*) and an image of an animal, then determines whether the statement is accurate.

### Dataset Requirements
- The dataset must contain at least **10 animal classes**.
- The NER model must be trained specifically for extracting animal names from text.

### Pipeline Flow
1. The user inputs a **text description** and an **image**.
2. The **NER model** extracts the animal name(s) from the text.
3. The **Image Classification model** predicts the animal in the image.
4. The system **compares** the NER output with the classification result and returns `True` (if they match) or `False` (if they don't).


## Requirements
- **Python**:
  - **3.6 - 3.9** (if using GPU)
  - **3.6 or higher** (if using CPU)
- **Libraries**:
  - `tensorflow`, `torch`, `transformers`, `scikit-learn`, `numpy`, `matplotlib`, `pandas`, `opencv-python`
- **Additional dependencies** are listed in the `requirements.txt` file.


This README provides a high-level overview of the project structure, tasks, and execution instructions. For further details, refer to the respective scripts in `task1/` and `task2/` directories.

