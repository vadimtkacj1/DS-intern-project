from models.MnistClassifierInterface import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier

class RFModel(MnistClassifierInterface):
    def __init__(self):
        self.__model = RandomForestClassifier(n_estimators=100)
        
    def train(self, x_train, y_train):
        x_train = x_train.reshape(-1, 784)
        self.__model.fit(x_train, y_train)

    def predict(self, x_pred):
        return self.__model.predict(x_pred)