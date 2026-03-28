import joblib
import numpy as np

def test_model():
    model = joblib.load("model.pkl")

    sample = np.array([[3, 1, 22, 1, 0, 7.25, 2]])

    prediction = model.predict(sample)

    assert prediction[0] in [0, 1]