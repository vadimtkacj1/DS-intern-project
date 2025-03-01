import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ImageProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, model, image_size, class_names):
        self.model = model
        self.image_size = image_size
        self.class_names = class_names

    def fit(self, X, y=None):
        return self

    def transform(self, X): 
        pred, image_path = X
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(self.image_size, self.image_size))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0) / 255.0 
        
        predictions = self.model.predict(image_array)
        predicted_label = np.argmax(predictions, axis=-1)[0]
        
        predicted_animal = self.class_names[predicted_label]
        return [pred, predicted_animal]

    def __sklearn_clone__(self):
       return ImageProcessor(self.model, self.image_size, self.class_names)