import tensorflow as tf

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