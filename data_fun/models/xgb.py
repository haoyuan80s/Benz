import xgboost as xgb
import numpy as np

class XGB(object):
    def __init__(self):
        self.model = None

    def train(self, X, y, params, num_boost_round = 10):
        """
        c.f. http://xgboost.readthedocs.io/en/latest/python/python_api.html
        """
        print("training XGB model......")
        dtrain = xgb.DMatrix(X, y)
        self.model = xgb.train(params, dtrain, num_boost_round = num_boost_round)
        
    
    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X))
