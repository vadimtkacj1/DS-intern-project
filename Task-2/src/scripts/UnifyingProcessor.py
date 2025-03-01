from sklearn.base import BaseEstimator

class UnifyingProcessor(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        text_predictions, image_predictions = X

        return [image_predictions in text_predictions] 
    
    def __sklearn_clone__(self):
       return UnifyingProcessor()