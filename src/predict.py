import pandas as pd
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_telco_customer_churn.csv"
MODEL_PATH = BASE_DIR / "reports" / "churn_model.pkl"
REPORT_PATH = BASE_DIR / "reports" / "churn_summary.md"

# Load artifacts
def load_model():
    return joblib.load(MODEL_PATH)


def load_data():
    return pd.read_csv(DATA_PATH)

def predict_churn(model, df):
    X = df.drop(columns=["customerID", "Churn"])
    churn_prob = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["churn_probability"] = churn_prob
    df["high_risk"] = (df["churn_probability"] >= 0.7).astype(int)

    return df

# Generate insights
def generate_insights(df):
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

    # Top churn drivers (from domain + model behavior)
    insights["top_drivers"] = [
        "MonthlyCharges",
        "tenure",
        "Contract",
    ]

    return insights

# Save report
def save_report(insights):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# 📊 Customer Churn Automated Report\n\n")
        f.write(f"**Total Customers Analyzed:** {insights['total_customers']}\n\n")
        f.write(
            f"**High-Risk Customers:** {insights['high_risk_pct']}% predicted to churn\n\n"
        )
        f.write(
            f"**Highest Risk Segment:** {insights['highest_risk_contract']}\n\n"
        )

        f.write("## 🔑 Key Churn Drivers\n")
        for driver in insights["top_drivers"]:
            f.write(f"- {driver}\n")

        f.write(
            "\n## 💡 Recommended Actions\n"
            "- Offer incentives to move customers to long-term contracts\n"
            "- Target high monthly charge customers with loyalty benefits\n"
            "- Proactively engage customers with low tenure\n"
        )

def main():
    print("📥 Loading model and data...")
    model = load_model()
    df = load_data()

    print("🔮 Predicting churn risk...")
    df = predict_churn(model, df)

    print("📊 Generating insights...")
    insights = generate_insights(df)

    save_report(insights)

    print("\n✅ Automated churn insights generated successfully.")
    print(f"📄 Report saved at: {REPORT_PATH}")


if __name__ == "__main__":
    main()