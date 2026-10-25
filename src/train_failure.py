import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("data/ai4i2020.csv")

df.columns = (
    df.columns
    .str.replace('[', '', regex=False)
    .str.replace(']', '', regex=False)
)

features = [
    "Air temperature K",
    "Process temperature K",
    "Rotational speed rpm",
    "Torque Nm",
    "Tool wear min"
]

X = df[features]
y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

joblib.dump(model, "models/failure_model.pkl")

print("Model saved successfully!")