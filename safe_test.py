
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_model():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

if __name__ == "__main__":
    model = train_model()
    print("Model training completed safely")