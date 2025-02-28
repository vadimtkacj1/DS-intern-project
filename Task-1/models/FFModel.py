from models.MnistClassifierInterface import MnistClassifierInterface
import tensorflow as tf

class FFModel(MnistClassifierInterface):
    def __init__(self):
        self.__model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.__model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train):
        self.__model.fit(x_train, y_train, epochs=15, validation_split=0.2, batch_size=120)

    def predict(self, x_pred):
        return self.__model.predict(x_pred)