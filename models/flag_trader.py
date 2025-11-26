from sklearn.ensemble import RandomForestClassifier
from joblib import dump

class FlagTraderAnalyst:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def save(self, path):
        dump(self.model, path)
