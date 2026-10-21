import pandas as pd
import numpy as np
import joblib

print("=" * 60)
print("FORENSIC RISK PRIORITIZATION SYSTEM")
print("=" * 60)


# ==================================================
# LOAD MODEL
# ==================================================

print("\nLoading trained Random Forest model...")

rf_model = joblib.load("fraud_rf_model.pkl")

print("Model loaded successfully.")


# ==================================================
# LOAD BEHAVIORAL PROFILES
# ==================================================

print("\nLoading behavioral profiles...")

behavioral_df = pd.read_csv(
    "behavioral_profiles.csv"
)

print("Behavioral profiles loaded successfully.")
print("Behavioral dataset shape:", behavioral_df.shape)


# ==================================================
# LOAD TEST DATA
# ==================================================

print("\nLoading test data...")

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze("columns")

print("Test data loaded successfully.")
print("Test dataset shape:", X_test.shape)


# ==================================================
# GENERATE FRAUD PROBABILITIES
# ==================================================

print("\nGenerating fraud likelihood scores...")

fraud_probabilities = rf_model.predict_proba(
    X_test
)[:, 1]

print("Fraud likelihood scores generated successfully.")


# ==================================================
# CREATE TEST CASE DATAFRAME
# ==================================================

print("\nCreating forensic case dataset...")

test_cases = X_test.copy()

test_cases["actual_fraud"] = y_test.to_numpy()

test_cases["fraud_probability"] = (
    fraud_probabilities
)

print("Forensic case dataset created.")


# ==================================================
# EXTRACT TEST PERIOD
# ==================================================

print("\nMatching behavioral profiles with test period...")

total_rows = len(behavioral_df)

test_start = int(total_rows * 0.85)

behavioral_test = (
    behavioral_df
    .iloc[test_start:]
    .reset_index(drop=True)
)

print("Behavioral test shape:", behavioral_test.shape)


# ==================================================
# COMBINE MODEL + BEHAVIORAL DATA
# ==================================================

print("\nCombining fraud likelihood and behavioral risk...")

forensic_cases = test_cases.copy()

forensic_cases[
    "behavioral_risk_score"
] = behavioral_test[
    "behavioral_risk_score"
].to_numpy()

forensic_cases[
    "behavioral_risk_level"
] = behavioral_test[
    "behavioral_risk_level"
].to_numpy()

forensic_cases[
    "orig_amount_deviation_ratio"
] = behavioral_test[
    "orig_amount_deviation_ratio"
].to_numpy()

forensic_cases[
    "orig_previous_transaction_count"
] = behavioral_test[
    "orig_previous_transaction_count"
].to_numpy()

forensic_cases[
    "orig_step_gap"
] = behavioral_test[
    "orig_step_gap"
].to_numpy()

forensic_cases[
    "orig_unique_destinations"
] = behavioral_test[
    "orig_unique_destinations"
].to_numpy()

print("Data combined successfully.")


# ==================================================
# NORMALIZE BEHAVIORAL RISK SCORE
# ==================================================

print("\nNormalizing behavioral risk score...")

forensic_cases[
    "behavioral_risk_normalized"
] = (
    forensic_cases["behavioral_risk_score"]
    / 85
)

forensic_cases[
    "behavioral_risk_normalized"
] = (
    forensic_cases[
        "behavioral_risk_normalized"
    ]
    .clip(0, 1)
)


# ==================================================
# FINAL FORENSIC PRIORITY SCORE
# ==================================================

print("\nCalculating final forensic priority score...")

# ML fraud probability receives higher importance
# Behavioral profiling provides supporting context

forensic_cases[
    "forensic_priority_score"
] = (
    0.80
    * forensic_cases["fraud_probability"]
    +
    0.20
    * forensic_cases["behavioral_risk_normalized"]
)


# ==================================================
# FINAL FORENSIC PRIORITY LEVEL
# ==================================================

print("\nAssigning investigation priority levels...")


def priority_level(score):

    if score >= 0.80:
        return "CRITICAL"

    elif score >= 0.55:
        return "HIGH"

    elif score >= 0.30:
        return "MEDIUM"

    else:
        return "LOW"


forensic_cases[
    "forensic_priority_level"
] = (
    forensic_cases[
        "forensic_priority_score"
    ].apply(priority_level)
)


# ==================================================
# PRIORITY DISTRIBUTION
# ==================================================

print("\n" + "=" * 60)
print("FORENSIC PRIORITY DISTRIBUTION")
print("=" * 60)

print(
    forensic_cases[
        "forensic_priority_level"
    ].value_counts()
)


# ==================================================
# FRAUD RATE BY PRIORITY LEVEL
# ==================================================

print("\nFraud Rate by Investigation Priority:")

priority_summary = (
    forensic_cases
    .groupby(
        "forensic_priority_level"
    )["actual_fraud"]
    .agg(
        total_cases="count",
        fraud_cases="sum",
        fraud_rate="mean"
    )
)

priority_summary[
    "fraud_rate"
] = (
    priority_summary[
        "fraud_rate"
    ] * 100
)

priority_summary = (
    priority_summary
    .sort_values(
        "fraud_rate",
        ascending=False
    )
)

print(
    priority_summary.round(4)
)


# ==================================================
# TOP INVESTIGATION CASES
# ==================================================

print("\n" + "=" * 60)
print("TOP FORENSIC INVESTIGATION CASES")
print("=" * 60)


top_cases = (
    forensic_cases
    .sort_values(
        "forensic_priority_score",
        ascending=False
    )
    .head(20)
)


display_columns = [
    "actual_fraud",
    "fraud_probability",
    "behavioral_risk_score",
    "behavioral_risk_level",
    "forensic_priority_score",
    "forensic_priority_level",
    "amount",
    "orig_amount_deviation_ratio",
    "orig_previous_transaction_count",
    "orig_step_gap",
    "orig_unique_destinations"
]


print(
    top_cases[
        display_columns
    ]
    .to_string(index=False)
)


# ==================================================
# CRITICAL CASE ANALYSIS
# ==================================================

print("\n" + "=" * 60)
print("CRITICAL CASE ANALYSIS")
print("=" * 60)


critical_cases = forensic_cases[
    forensic_cases[
        "forensic_priority_level"
    ] == "CRITICAL"
]


critical_fraud_count = (
    critical_cases[
        "actual_fraud"
    ].sum()
)


print("\nTotal Critical Cases:")
print(len(critical_cases))

print("\nConfirmed Fraud Cases in Critical Group:")
print(critical_fraud_count)

print("\nFraud Probability in Critical Group:")

print(
    critical_cases[
        "fraud_probability"
    ].describe()
)


# ==================================================
# SAVE RESULTS
# ==================================================

print("\nSaving forensic investigation results...")


forensic_cases.to_csv(
    "forensic_investigation_results.csv",
    index=False
)


print(
    "Saved successfully:"
)

print(
    "forensic_investigation_results.csv"
)


# ==================================================
# FINAL INTERPRETATION
# ==================================================

print("\n" + "=" * 60)
print("FORENSIC RISK PRIORITIZATION COMPLETED")
print("=" * 60)


print("\nForensic Interpretation:")

print(
    "The final forensic prioritization system combines machine learning "
    "fraud likelihood with behavioral transaction context. The Random "
    "Forest fraud probability receives the largest weight because it was "
    "validated directly against known fraud labels, while behavioral "
    "indicators provide supporting evidence based on account history and "
    "transaction patterns. The resulting priority score can help "
    "investigators allocate attention to transactions requiring the most "
    "urgent review."
)