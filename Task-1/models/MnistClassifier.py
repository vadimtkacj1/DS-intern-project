from models.CNNModel import CNNModel
from models.RFModel import RFModel
from models.FFModel import FFModel

class MnistClassifier:
    def __init__(self, algorithm_name):
        if algorithm_name == 'cnn':
            self.__model = CNNModel()
        elif algorithm_name == 'rf':
            self.__model = RFModel()
        elif algorithm_name == 'nn':
            self.__model = FFModel()

    #TODO: add params
    def train(self, x_train, y_train):
        return self.__model.train(x_train, y_train)

    def predict(self, y_pred):
        return self.__model.predict(y_pred)
