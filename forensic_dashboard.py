import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("FORENSIC INVESTIGATION VISUALIZATION DASHBOARD")
print("=" * 60)


# ==================================================
# LOAD RESULTS
# ==================================================

print("\nLoading forensic investigation results...")

df = pd.read_csv(
    "forensic_investigation_results.csv"
)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ==================================================
# DEFINE PRIORITY ORDER
# ==================================================

priority_order = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL"
]


# ==================================================
# PRIORITY DISTRIBUTION
# ==================================================

print("\nGenerating priority distribution chart...")

priority_counts = (
    df["forensic_priority_level"]
    .value_counts()
    .reindex(priority_order, fill_value=0)
)

plt.figure(figsize=(9, 6))

plt.bar(
    priority_counts.index,
    priority_counts.values
)

plt.title("Forensic Investigation Priority Distribution")
plt.xlabel("Investigation Priority")
plt.ylabel("Number of Transactions")

for i, value in enumerate(priority_counts.values):
    plt.text(
        i,
        value,
        f"{value:,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    "forensic_priority_distribution.png",
    dpi=300
)

plt.close()

print(
    "Saved: forensic_priority_distribution.png"
)


# ==================================================
# FRAUD RATE BY PRIORITY LEVEL
# ==================================================

print("\nGenerating fraud rate chart...")

priority_summary = (
    df.groupby(
        "forensic_priority_level"
    )["actual_fraud"]
    .agg(
        total_cases="count",
        fraud_cases="sum",
        fraud_rate="mean"
    )
)

priority_summary["fraud_rate"] = (
    priority_summary["fraud_rate"] * 100
)

priority_summary = (
    priority_summary
    .reindex(priority_order)
    .fillna(0)
)

plt.figure(figsize=(9, 6))

plt.bar(
    priority_summary.index,
    priority_summary["fraud_rate"]
)

plt.title("Fraud Rate by Forensic Investigation Priority")
plt.xlabel("Investigation Priority")
plt.ylabel("Fraud Rate (%)")

for i, value in enumerate(
    priority_summary["fraud_rate"]
):
    plt.text(
        i,
        value,
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.ylim(
    0,
    max(priority_summary["fraud_rate"]) * 1.15
)

plt.tight_layout()

plt.savefig(
    "fraud_rate_by_priority.png",
    dpi=300
)

plt.close()

print(
    "Saved: fraud_rate_by_priority.png"
)


# ==================================================
# FRAUD PROBABILITY DISTRIBUTION
# ==================================================

print("\nGenerating fraud probability distribution...")

plt.figure(figsize=(10, 6))

fraud_cases = df[
    df["actual_fraud"] == 1
]["fraud_probability"]

genuine_cases = df[
    df["actual_fraud"] == 0
]["fraud_probability"]

plt.hist(
    genuine_cases,
    bins=50,
    alpha=0.7,
    label="Genuine Transactions"
)

plt.hist(
    fraud_cases,
    bins=50,
    alpha=0.7,
    label="Fraud Transactions"
)

plt.title("Fraud Likelihood Score Distribution")
plt.xlabel("Fraud Probability")
plt.ylabel("Number of Transactions")

plt.legend()

plt.tight_layout()

plt.savefig(
    "fraud_probability_distribution.png",
    dpi=300
)

plt.close()

print(
    "Saved: fraud_probability_distribution.png"
)


# ==================================================
# FRAUD PROBABILITY VS BEHAVIORAL RISK
# ==================================================

print(
    "\nGenerating ML Probability vs Behavioral Risk chart..."
)

# Use sampling because dataset is very large
sample_size = min(100000, len(df))

sample_df = df.sample(
    n=sample_size,
    random_state=42
)

plt.figure(figsize=(10, 7))

plt.scatter(
    sample_df[
        sample_df["actual_fraud"] == 0
    ]["fraud_probability"],
    
    sample_df[
        sample_df["actual_fraud"] == 0
    ]["behavioral_risk_score"],
    
    alpha=0.3,
    s=10,
    label="Genuine"
)

plt.scatter(
    sample_df[
        sample_df["actual_fraud"] == 1
    ]["fraud_probability"],
    
    sample_df[
        sample_df["actual_fraud"] == 1
    ]["behavioral_risk_score"],
    
    alpha=0.7,
    s=20,
    label="Fraud"
)

plt.title(
    "Machine Learning Fraud Probability vs Behavioral Risk"
)

plt.xlabel(
    "Random Forest Fraud Probability"
)

plt.ylabel(
    "Behavioral Risk Score"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "ml_vs_behavioral_risk.png",
    dpi=300
)

plt.close()

print(
    "Saved: ml_vs_behavioral_risk.png"
)


# ==================================================
# FINAL PRIORITY SCORE DISTRIBUTION
# ==================================================

print(
    "\nGenerating forensic priority score distribution..."
)

plt.figure(figsize=(10, 6))

plt.hist(
    df["forensic_priority_score"],
    bins=50
)

plt.title(
    "Final Forensic Priority Score Distribution"
)

plt.xlabel(
    "Forensic Priority Score"
)

plt.ylabel(
    "Number of Transactions"
)

plt.tight_layout()

plt.savefig(
    "forensic_priority_score_distribution.png",
    dpi=300
)

plt.close()

print(
    "Saved: forensic_priority_score_distribution.png"
)


# ==================================================
# SUMMARY TABLE
# ==================================================

print("\n" + "=" * 60)
print("FORENSIC DASHBOARD SUMMARY")
print("=" * 60)

summary = pd.DataFrame({
    "Priority_Level": priority_order,
    "Total_Cases": [
        priority_counts[level]
        for level in priority_order
    ],
    "Fraud_Cases": [
        priority_summary.loc[
            level,
            "fraud_cases"
        ]
        for level in priority_order
    ],
    "Fraud_Rate_Percentage": [
        priority_summary.loc[
            level,
            "fraud_rate"
        ]
        for level in priority_order
    ]
})

print(
    summary
    .round(4)
    .to_string(index=False)
)


# ==================================================
# SAVE SUMMARY
# ==================================================

summary.to_csv(
    "forensic_dashboard_summary.csv",
    index=False
)

print(
    "\nSaved: forensic_dashboard_summary.csv"
)


# ==================================================
# COMPLETION
# ==================================================

print("\n" + "=" * 60)
print("FORENSIC DASHBOARD COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated visualization files:")

print("1. forensic_priority_distribution.png")
print("2. fraud_rate_by_priority.png")
print("3. fraud_probability_distribution.png")
print("4. ml_vs_behavioral_risk.png")
print("5. forensic_priority_score_distribution.png")

print("\nForensic Interpretation:")

print(
    "The visualization dashboard summarizes how transactions are "
    "distributed across investigation priority levels and compares "
    "machine learning fraud likelihood with behavioral risk indicators. "
    "These visualizations support forensic investigators by showing "
    "which transactions should receive immediate attention."
)