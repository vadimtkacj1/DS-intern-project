import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
sys.path.append("..")
from models.CNNModel import CNNModel
import hydra
from omegaconf import DictConfig
from weights_manipulation import calculate_class_weights
from dataset_loading import load_dataset
from preparation_image_dataset import normalize_data
from gpu_manipulating import set_device

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
    train_data, validation_data = normalize_data(train_data, validation_data)

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
