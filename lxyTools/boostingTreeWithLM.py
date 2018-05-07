import numpy as np
import gc

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


class BoosterLmClassifier:
    def __init__(self, nm, lm=LogisticRegression()):
        self.nonlinearModel = nm
        self.linearModel = lm
        self.onehot = OneHotEncoder(dtype=np.uint8)

    def fit(self, X, y):
        self.nonlinearModel.fit(X, y)
        leafX = self.nonlinearModel.booster_.predict(X, pred_leaf=True)

        self.onehot.fit(leafX)
        onhotX = self.onehot.transform(leafX)

        self.linearModel.fit(onhotX, y)

        del onhotX
        del leafX
        gc.collect()

    def predict_proba(self, X):
        leafX = self.nonlinearModel.booster_.predict(X, pred_leaf=True)
        onhotX = self.onehot.transform(leafX)
        out = self.linearModel.predict_proba(onhotX)

        del leafX
        del onhotX
        gc.collect()

        return out
