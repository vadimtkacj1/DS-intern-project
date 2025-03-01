import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import hydra
from omegaconf import DictConfig

def get_test_dataset(image_size, data_path, batch_size, seed=123, interpolation='bilinear', label_mode="categorical"):
    test_data = tf.keras.utils.image_dataset_from_directory(
        data_path,
        image_size=(image_size, image_size),
        label_mode=label_mode,
        batch_size=batch_size,
        seed=seed,
        interpolation=interpolation
    )
    
    # Normalize the test data
    test_data = test_data.map(lambda x, y: (x / 255.0, y))
    
    return test_data

@hydra.main(config_path="../../config", config_name="ner-config.yaml")
def main(cfg: DictConfig):
    config_dir = hydra.utils.get_original_cwd()
    global_cfg = hydra.compose(config_name="config.yaml")
    
    # Define paths
    root_path = os.path.abspath(os.path.join(config_dir, '../..')) 
    models_path = os.path.join(root_path, "models")
    data_path = os.path.join(root_path, "data", "raw")

    model_name = f"model_{global_cfg.image_size}.h5"
    model_file_path = os.path.join(models_path, model_name)
    model = tf.keras.models.load_model(model_file_path)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"Model {model_name} loaded.")

    test_data = get_test_dataset(global_cfg.image_size, data_path, cfg.training.batch_size, seed=global_cfg.SEED)

    loss, accuracy = model.evaluate(test_data)
    print(loss, accuracy)

    print(f"Accuracy for image size {global_cfg.image_size}: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
