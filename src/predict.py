import joblib
import numpy as np

# Load model & encoders
model = joblib.load("model.pkl")
le_sex = joblib.load("le_sex.pkl")
le_embarked = joblib.load("le_embarked.pkl")

# Example input
sample = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}

# Convert categorical
sample["Sex"] = le_sex.transform([sample["Sex"]])[0]
sample["Embarked"] = le_embarked.transform([sample["Embarked"]])[0]

# Convert to array
features = np.array([[
    sample["Pclass"],
    sample["Sex"],
    sample["Age"],
    sample["SibSp"],
    sample["Parch"],
    sample["Fare"],
    sample["Embarked"]
]])

prediction = model.predict(features)

print("Survival Prediction:", prediction[0])