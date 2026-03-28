# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("data/train.csv")

# Drop unnecessary columns
data = data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Handle missing values
data["Age"].fillna(data["Age"].median(), inplace=True)
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# Encode categorical columns
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

data["Sex"] = le_sex.fit_transform(data["Sex"])
data["Embarked"] = le_embarked.fit_transform(data["Embarked"])

# Features & Target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save everything
joblib.dump(model, "model.pkl")
joblib.dump(le_sex, "le_sex.pkl")
joblib.dump(le_embarked, "le_embarked.pkl")

print("Model trained successfully!")