# src/preprocess.py

import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")
PROCESSED_DATA_PATH = Path("data/processed/cleaned_telco_customer_churn.csv")


def load_data():
    return pd.read_csv(RAW_DATA_PATH)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Feature Engineering
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
    )

    df["high_value_customer"] = (df["MonthlyCharges"] > 70).astype(int)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def save_processed_data(df: pd.DataFrame):
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)


def main():
    df = load_data()
    df = preprocess_data(df)
    save_processed_data(df)
    print("✅ Data preprocessing completed.")


if __name__ == "__main__":
    main()