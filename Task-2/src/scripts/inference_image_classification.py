import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import hydra
from omegaconf import DictConfig 
from dataset_loading import load_test_dataset

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

    test_data = load_test_dataset(global_cfg.image_size, data_path, cfg.training.batch_size, seed=global_cfg.SEED)

    loss, accuracy = model.evaluate(test_data)
    print(loss, accuracy)

    print(f"Accuracy for image size {global_cfg.image_size}: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
