import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
from models.cnn import CNNModel
import hydra
from omegaconf import DictConfig

def load_dataset(image_size, data_path, batch_size, validation_split=0.2, seed=123):
    label_mode = "categorical"

    train_data = tf.keras.utils.image_dataset_from_directory(
        data_path,
        image_size=(image_size, image_size),
        label_mode=label_mode,
        batch_size=batch_size,
        seed=seed,
        validation_split=validation_split,
        subset="training",
        interpolation='bilinear'
    )

    validation_data = tf.keras.utils.image_dataset_from_directory(
        data_path,
        image_size=(image_size, image_size),
        label_mode=label_mode,
        batch_size=batch_size,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
        interpolation='bilinear'
    )

    return train_data, validation_data

def prepare_and_normalize_data(train_data, validation_data):
    # Normalize the images
    train_data = train_data.map(lambda x, y: (x / 255.0, y))
    validation_data = validation_data.map(lambda x, y: (x / 255.0, y))

    return train_data, validation_data

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

def set_device(gpu_enabled):
    # Check if GPU is available and if the config requests GPU usage
    if gpu_enabled and tf.config.list_physical_devices('GPU'):
        print("Using GPU for training.")

        # Set memory growth on GPU (optional)
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("Using CPU for training.")

@hydra.main(config_path="../../config", config_name="image-classification-config.yaml")
def main(cfg: DictConfig):
    config_dir = hydra.utils.get_original_cwd()
    global_cfg = hydra.compose(config_name="config.yaml")
    
    # Define paths
    root_path = os.path.abspath(os.path.join(config_dir, '../..')) 
    models_path = os.path.join(root_path, "models")
    data_path = os.path.join(root_path, "data", "raw")

    set_device(cfg.training.gpu_enabled)

    train_data, validation_data = load_dataset(global_cfg.image_size, data_path, cfg.training.batch_size, seed=global_cfg.SEED)
    class_names = train_data.class_names
    train_data, validation_data = prepare_and_normalize_data(train_data, validation_data)

    class_weight_dict = calculate_class_weights(train_data)

    input_shape = (global_cfg.image_size, global_cfg.image_size, 3)
    num_classes = len(class_names)
    metric = ["accuracy"]

    model = CNNModel(input_shape=input_shape, num_classes=num_classes, metric=metric)

    model.train(
        train_data, 
        epochs=cfg.training.epochs, 
        validation_data=validation_data, 
        class_weight=class_weight_dict,
        batch_size=cfg.training.batch_size
    )
    
    model_name = f"model_{global_cfg.image_size}.h5"
    model_file_path = os.path.join(models_path, model_name)
    model.save(model_file_path)
    print(f"Model saved as {model_name}")

if __name__ == "__main__":
    main()
