import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 75)
print("TEMPORAL FRAUD BEHAVIOR AND CONCEPT DRIFT ANALYSIS")
print("=" * 75)


# ============================================================
# 1. LOAD PAYSim DATA
# ============================================================

print("\nLoading PaySim dataset...")

df = pd.read_csv("paysim.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 2. BASIC VALIDATION
# ============================================================

required_columns = [
    "step",
    "amount",
    "type",
    "isFraud"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    print("\nERROR: Missing columns:")
    print(missing_columns)

    raise SystemExit


# ============================================================
# 3. CREATE TIME PERIODS
# ============================================================

print("\nCreating chronological time periods...")

minimum_step = df["step"].min()
maximum_step = df["step"].max()

print("Minimum step:", minimum_step)
print("Maximum step:", maximum_step)


# Divide the transaction history into 10 chronological periods.

df["time_period"] = pd.qcut(
    df["step"],
    q=10,
    duplicates="drop"
)


# ============================================================
# 4. TEMPORAL FRAUD STATISTICS
# ============================================================

print("\nCalculating fraud statistics by time period...")


temporal_summary = (
    df.groupby(
        "time_period",
        observed=True
    )
    .agg(
        total_transactions=("isFraud", "count"),
        fraud_transactions=("isFraud", "sum"),
        total_amount=("amount", "sum"),
        average_amount=("amount", "mean")
    )
)


temporal_summary["fraud_rate"] = (
    temporal_summary["fraud_transactions"]
    /
    temporal_summary["total_transactions"]
) * 100


temporal_summary["fraud_amount_ratio"] = (
    temporal_summary["total_amount"]
)


print("\n" + "=" * 75)
print("TEMPORAL FRAUD DISTRIBUTION")
print("=" * 75)


print(
    temporal_summary[
        [
            "total_transactions",
            "fraud_transactions",
            "fraud_rate",
            "average_amount"
        ]
    ]
    .round(4)
    .to_string()
)


# ============================================================
# 5. FRAUD RATE CHANGE
# ============================================================

first_fraud_rate = (
    temporal_summary["fraud_rate"].iloc[0]
)

last_fraud_rate = (
    temporal_summary["fraud_rate"].iloc[-1]
)


fraud_rate_change = (
    last_fraud_rate
    -
    first_fraud_rate
)


relative_change = (
    fraud_rate_change
    /
    first_fraud_rate
) * 100 if first_fraud_rate != 0 else 0


print("\n" + "=" * 75)
print("TEMPORAL FRAUD RATE CHANGE")
print("=" * 75)


print(
    f"\nFirst period fraud rate: "
    f"{first_fraud_rate:.6f}%"
)

print(
    f"Last period fraud rate: "
    f"{last_fraud_rate:.6f}%"
)

print(
    f"Absolute change: "
    f"{fraud_rate_change:+.6f}%"
)

print(
    f"Relative change: "
    f"{relative_change:+.2f}%"
)


# ============================================================
# 6. FIND HIGHEST-RISK PERIOD
# ============================================================

highest_risk_period = (
    temporal_summary[
        "fraud_rate"
    ].idxmax()
)


highest_risk_rate = (
    temporal_summary.loc[
        highest_risk_period,
        "fraud_rate"
    ]
)


print("\n" + "=" * 75)
print("HIGHEST FRAUD-RATE PERIOD")
print("=" * 75)


print(
    "\nHighest fraud-rate period:"
)

print(
    highest_risk_period
)

print(
    f"Fraud rate: "
    f"{highest_risk_rate:.6f}%"
)


# ============================================================
# 7. TRANSACTION TYPE TEMPORAL ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("FRAUD BEHAVIOR BY TRANSACTION TYPE")
print("=" * 75)


type_summary = (
    df.groupby("type")
    .agg(
        total_transactions=("isFraud", "count"),
        fraud_transactions=("isFraud", "sum"),
        average_amount=("amount", "mean")
    )
)


type_summary["fraud_rate"] = (
    type_summary["fraud_transactions"]
    /
    type_summary["total_transactions"]
) * 100


print(
    type_summary
    .sort_values(
        "fraud_rate",
        ascending=False
    )
    .round(6)
    .to_string()
)


# ============================================================
# 8. FRAUD AMOUNT ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("FRAUD TRANSACTION AMOUNT ANALYSIS")
print("=" * 75)


fraud_amount = df.loc[
    df["isFraud"] == 1,
    "amount"
]


genuine_amount = df.loc[
    df["isFraud"] == 0,
    "amount"
]


print(
    f"\nAverage fraud transaction amount: "
    f"{fraud_amount.mean():,.2f}"
)

print(
    f"Median fraud transaction amount: "
    f"{fraud_amount.median():,.2f}"
)

print(
    f"Maximum fraud transaction amount: "
    f"{fraud_amount.max():,.2f}"
)


print(
    f"\nAverage genuine transaction amount: "
    f"{genuine_amount.mean():,.2f}"
)

print(
    f"Median genuine transaction amount: "
    f"{genuine_amount.median():,.2f}"
)


# ============================================================
# 9. TEMPORAL FRAUD AMOUNT
# ============================================================

temporal_fraud_amount = (
    df.groupby(
        "time_period",
        observed=True
    )
    .apply(
        lambda x: pd.Series({

            "fraud_amount":
                x.loc[
                    x["isFraud"] == 1,
                    "amount"
                ].sum(),

            "average_fraud_amount":
                x.loc[
                    x["isFraud"] == 1,
                    "amount"
                ].mean()

        }),
        include_groups=False
    )
)


print("\n" + "=" * 75)
print("TEMPORAL FRAUD AMOUNT ANALYSIS")
print("=" * 75)


print(
    temporal_fraud_amount
    .round(2)
    .to_string()
)


# ============================================================
# 10. SAVE TEMPORAL RESULTS
# ============================================================

temporal_output = temporal_summary.reset_index()

temporal_output.to_csv(
    "temporal_fraud_analysis.csv",
    index=False
)


type_summary.reset_index().to_csv(
    "temporal_transaction_type_analysis.csv",
    index=False
)


print("\nSaved:")
print("temporal_fraud_analysis.csv")
print("temporal_transaction_type_analysis.csv")


# ============================================================
# 11. GRAPH — FRAUD RATE OVER TIME
# ============================================================

print("\nGenerating temporal fraud-rate graph...")


plt.figure(
    figsize=(12, 6)
)


plt.plot(
    range(
        1,
        len(temporal_summary) + 1
    ),
    temporal_summary["fraud_rate"],
    marker="o"
)


plt.xlabel(
    "Chronological Time Period"
)

plt.ylabel(
    "Fraud Rate (%)"
)

plt.title(
    "Temporal Variation in Fraud Rate"
)


plt.xticks(
    range(
        1,
        len(temporal_summary) + 1
    )
)


plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    "temporal_fraud_rate.png",
    dpi=300
)

plt.close()


print(
    "Saved: temporal_fraud_rate.png"
)


# ============================================================
# 12. GRAPH — FRAUD TRANSACTION COUNT
# ============================================================

print(
    "\nGenerating fraud transaction count graph..."
)


plt.figure(
    figsize=(12, 6)
)


plt.bar(
    range(
        1,
        len(temporal_summary) + 1
    ),
    temporal_summary["fraud_transactions"]
)


plt.xlabel(
    "Chronological Time Period"
)

plt.ylabel(
    "Number of Fraud Transactions"
)

plt.title(
    "Fraud Transaction Count Across Time"
)


plt.xticks(
    range(
        1,
        len(temporal_summary) + 1
    )
)


plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    "temporal_fraud_count.png",
    dpi=300
)

plt.close()


print(
    "Saved: temporal_fraud_count.png"
)


# ============================================================
# 13. GRAPH — AVERAGE TRANSACTION AMOUNT
# ============================================================

print(
    "\nGenerating transaction amount graph..."
)


plt.figure(
    figsize=(12, 6)
)


plt.plot(
    range(
        1,
        len(temporal_summary) + 1
    ),
    temporal_summary["average_amount"],
    marker="o"
)


plt.xlabel(
    "Chronological Time Period"
)

plt.ylabel(
    "Average Transaction Amount"
)

plt.title(
    "Average Transaction Amount Across Time"
)


plt.xticks(
    range(
        1,
        len(temporal_summary) + 1
    )
)


plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    "temporal_transaction_amount.png",
    dpi=300
)

plt.close()


print(
    "Saved: temporal_transaction_amount.png"
)


# ============================================================
# 14. CONCEPT DRIFT INTERPRETATION
# ============================================================

print("\n" + "=" * 75)
print("CONCEPT DRIFT INTERPRETATION")
print("=" * 75)


if abs(relative_change) < 10:

    drift_interpretation = (
        "The overall fraud rate remains relatively stable "
        "between the earliest and latest chronological periods."
    )

elif relative_change > 0:

    drift_interpretation = (
        "The fraud rate increases toward the later "
        "chronological period, indicating a possible "
        "change in fraud prevalence or transaction behavior."
    )

else:

    drift_interpretation = (
        "The fraud rate decreases toward the later "
        "chronological period, indicating a possible "
        "change in fraud prevalence or transaction behavior."
    )


print(
    "\n" + drift_interpretation
)


print(
    """
Temporal analysis is important in forensic fraud detection
because fraudulent behavior may not remain constant over time.

The transaction history was divided into chronological
periods and fraud prevalence, transaction volume and
transaction amounts were evaluated independently for each
period.

This analysis helps determine whether the fraud environment
changes between earlier and later transaction periods.

The results should be interpreted together with the
chronological train, validation and test design.

A change in fraud prevalence does not automatically prove
model concept drift. It provides evidence that the
transaction environment may be changing and that continued
model monitoring may be necessary.

Therefore, temporal analysis provides an additional
forensic monitoring layer beyond conventional classification
metrics.
"""
)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("TEMPORAL DRIFT ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 75)


print("\nGenerated files:")

print("1. temporal_fraud_analysis.csv")
print("2. temporal_transaction_type_analysis.csv")
print("3. temporal_fraud_rate.png")
print("4. temporal_fraud_count.png")
print("5. temporal_transaction_amount.png")