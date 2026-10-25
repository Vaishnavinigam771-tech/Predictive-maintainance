import joblib
import numpy as np

failure_model = joblib.load("models/failure_model.pkl")
rul_model = joblib.load("models/rul_model.pkl")

def predict_machine(
    air_temp,
    process_temp,
    rpm,
    torque,
    wear
):
    sample = np.array([[
        air_temp,
        process_temp,
        rpm,
        torque,
        wear
    ]])

    failure_prob = failure_model.predict_proba(sample)[0][1]

    rul = rul_model.predict(sample)[0]

    return failure_prob, rul