import tensorflow as tf

def load_test_dataset(image_size, data_path, batch_size, seed=123, interpolation='bilinear', label_mode="categorical"):
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