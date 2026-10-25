import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

df = pd.read_csv("data/ai4i2020.csv")

df.columns = (
    df.columns
    .str.replace('[', '', regex=False)
    .str.replace(']', '', regex=False)
)

# Approximate Remaining Useful Life
df["RUL"] = 300 - df["Tool wear min"]

features = [
    "Air temperature K",
    "Process temperature K",
    "Rotational speed rpm",
    "Torque Nm",
    "Tool wear min"
]

X = df[features]
y = df["RUL"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))

joblib.dump(model, "models/rul_model.pkl")

print("RUL model saved successfully!")