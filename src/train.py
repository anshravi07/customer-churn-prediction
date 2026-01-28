# src/train.py

import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_telco_customer_churn.csv"
MODEL_PATH = BASE_DIR / "reports" / "churn_model.pkl"

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def build_pipeline(X):
    num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_features = X.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    return model

def main():
    print("📥 Loading data...")
    df = load_data()

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]

    print("🔀 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("⚙️ Building model pipeline...")
    model = build_pipeline(X)

    print("🚀 Training model...")
    model.fit(X_train, y_train)

    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {roc_auc:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\n✅ Model training completed & saved at: {MODEL_PATH}")

if __name__ == "__main__":
    main()