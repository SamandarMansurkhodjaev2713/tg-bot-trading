import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

class FinWorldAnalyst:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def save(self, path):
        dump(self.model, path)
