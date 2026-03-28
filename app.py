from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
le_sex = joblib.load("le_sex.pkl")
le_embarked = joblib.load("le_embarked.pkl")

@app.route("/")
def home():
    return "Titanic Model API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Encode
    sex = le_sex.transform([data["Sex"]])[0]
    embarked = le_embarked.transform([data["Embarked"]])[0]

    features = np.array([[
        data["Pclass"],
        sex,
        data["Age"],
        data["SibSp"],
        data["Parch"],
        data["Fare"],
        embarked
    ]])

    prediction = model.predict(features)

    return jsonify({
        "Survived": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# whatever you changed in app.py then run this command :
# >>docker run -p 5001:5000 -v ${PWD}:/app mlops-project    