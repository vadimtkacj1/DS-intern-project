# README: Named Entity Recognition + Image Classification Pipeline

## Task Overview
This project focuses on building a machine learning pipeline that combines two different tasks:
1. **Named Entity Recognition (NER)**: Extracts animal names from text.
2. **Image Classification**: Classifies animals in images.

The main goal of this pipeline is to take a user-provided text and an image, and determine if the text description of the image (e.g., "There is a cow in the picture") is accurate.

## Requirements

### Python version:
- **For GPU Support (Image Classification Model)**: Python <= 3.9 (TensorFlow GPU support is available only for Python versions 3.9 or lower).
- **For CPU-Only Setup**: Python >= 3.6 (TensorFlow CPU support is available for newer Python versions).

### TensorFlow version:
- **For GPU Support**: TensorFlow <= 2.10 (For TensorFlow GPU support).
- **For CPU-Only Setup**: TensorFlow >= 2.16 (For TensorFlow CPU support with newer Python versions).

### CUDA & cuDNN:
- CUDA >= 11.2
- cuDNN >= 8.1

## Installation Instructions

### 1. Setup the Environment
Before running the code, ensure that you have Python installed on your machine. We recommend using **Python 3.6 - 3.9** for compatibility with TensorFlow and CUDA for GPU support.

#### GPU Setup for Image Classification:
If you're training the **Image Classification model** on a GPU, follow these steps to set up your environment:

1. Install **CUDA 11.2** and **cuDNN 8.1**.
2. Ensure that your machine has a compatible GPU with the proper drivers installed.
3. Install the correct version of TensorFlow that supports GPU:

```bash
pip install tensorflow-gpu==2.10
```

## 2. Dataset Information

### Image Classification Dataset:
For the Image Classification task, we used the **Animals-10** dataset from Kaggle:  
[Kaggle Dataset - Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

This dataset contains images of 10 different animal classes: dog, cat, horse, spyder, butterfly, chicken, sheep, cow, squirrel, elephant; which are used for training and evaluating the image classification model. 

### Named Entity Recognition (NER) Dataset:
For the NER task, the dataset was manually created using English texts. The dataset contains various sentences with animal names labeled as named entities. This dataset is used to train and fine-tune a Named Entity Recognition model to extract animal names from text accurately.

## Usage

### 1. Run Dataset Laoding dataset:
- You need to go to notebooks and run loading_animal_dataset.ipynb to get dataset

### 2. Train the Models:
All files are in src/script, to set params go to config dir

#### Image Classification Dataset:
Need to start train_image_classification.py

#### Named Entity Recognition (NER) Dataset:
Need to start train_ner.py

### 3. Run Inference:
All files are in src/script, to set params go to config dir

#### Image Classification Dataset:
Need to start inference_image_classification.py

#### Named Entity Recognition (NER) Dataset:
Need to start inference_ner.py