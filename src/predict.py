# src/predict.py

import joblib
import pandas as pd
from pathlib import Path


MODEL_PATH = Path("reports/churn_model.pkl")


def load_model():
    return joblib.load(MODEL_PATH)


def predict(input_data: pd.DataFrame):
    model = load_model()
    return model.predict(input_data), model.predict_proba(input_data)


if __name__ == "__main__":
    print("Prediction module ready.")