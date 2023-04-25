from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles, make_blobs
import sklearn
import numpy as np


class PredictProba:
    def __init__(self, noise, factor, random_state, csv_name=None):
        self.noise = noise
        self.factor = factor
        self.random_state = random_state
        self.csv_name = csv_name

    def calculate(self):
        X, y = make_circles(noise=self.noise, factor=self.factor, random_state=self.random_state)
        y_named = np.array(["blue", "red"])[y]
        X_train, X_test, y_train_named, y_test_named, y_train, y_test = \
            train_test_split(X, y_named, y, random_state=0)
        gbrt = GradientBoostingClassifier(random_state=0)
        gbrt.fit(X_train, y_train_named)

        return {
            "array_form": list(X_test.shape),
            "decision_func_form": list(gbrt.decision_function(X_test).shape),
            "decision_func": list(gbrt.decision_function(X_test))
        }



