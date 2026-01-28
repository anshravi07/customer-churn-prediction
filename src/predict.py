import pandas as pd
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_telco_customer_churn.csv"
MODEL_PATH = BASE_DIR / "reports" / "churn_model.pkl"
REPORT_PATH = BASE_DIR / "reports" / "churn_summary.md"

# Load artifacts
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run train.py first.")
    return joblib.load(MODEL_PATH)

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Processed data not found. Run preprocess.py first.")
    return pd.read_csv(DATA_PATH)

# Prediction
def predict_churn(model, df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=["customerID", "Churn"])

    churn_prob = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["churn_probability"] = churn_prob
    df["high_risk"] = (df["churn_probability"] >= 0.7).astype(int)

    return df

# Insights generation
def generate_insights(df: pd.DataFrame) -> dict:
    insights = {}

    insights["total_customers"] = len(df)
    insights["high_risk_pct"] = round(df["high_risk"].mean() * 100, 2)

    # High-risk segment analysis
    contract_risk = (
        df.groupby("Contract")["high_risk"]
        .mean()
        .sort_values(ascending=False)
    )

    insights["highest_risk_contract"] = contract_risk.idxmax()

    insights["top_drivers"] = [
        "MonthlyCharges",
        "tenure",
        "Contract"
    ]

    return insights

# Report generation
def save_report(insights: dict):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Customer Churn Automated Report\n\n")

        f.write(f"**Total Customers Analyzed:** {insights['total_customers']}\n\n")
        f.write(
            f"**High-Risk Customers:** {insights['high_risk_pct']}% predicted to churn\n\n"
        )
        f.write(
            f"**Highest Risk Segment:** {insights['highest_risk_contract']}\n\n"
        )

        f.write("## Key Churn Drivers\n")
        for driver in insights["top_drivers"]:
            f.write(f"- {driver}\n")

        f.write(
            "\n## Recommended Actions\n"
            "- Encourage customers to shift to long-term contracts\n"
            "- Offer loyalty benefits to high monthly charge customers\n"
            "- Proactively engage customers with low tenure\n"
        )
def save_predictions(df: pd.DataFrame):
    output_path = BASE_DIR / "reports" / "high_risk_customers.csv"

    df_to_save = df[
        ["customerID", "churn_probability", "high_risk"]
    ].sort_values("churn_probability", ascending=False)

    df_to_save.to_csv(output_path, index=False)

    print(f"High-risk customer list saved at: {output_path}")

# Main pipeline
def main():
    print("Loading model and data...")
    model = load_model()
    df = load_data()

    print("Predicting churn risk...")
    df = predict_churn(model, df)

    print("Generating insights...")
    insights = generate_insights(df)

    save_report(insights)
    save_predictions(df)


    print("\nAutomated churn insights generated successfully.")
    print(f"Report saved at: {REPORT_PATH}")


if __name__ == "__main__":
    main()