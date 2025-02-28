from tensorflow.keras import layers, models 
from models.MnistClassifierInterface import MnistClassifierInterface

class CNNModel(MnistClassifierInterface):    
    def __init__(self):
        self.__model = self.__create_model()
        self.__model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def __create_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        return model
    
    def train(self, x_train, y_train):
        self.__model.fit(x_train, y_train, epochs=15, validation_split=0.2, batch_size=120) 

    def predict(self, x_pred):
        return self.__model.predict(x_pred)

