import pandas as pd
import numpy as np

print("=" * 60)
print("ACCOUNT BEHAVIORAL PROFILING")
print("=" * 60)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("\nLoading PaySim dataset...")

df = pd.read_csv(
    "paysim.csv",
    usecols=[
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "isFraud"
    ]
)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# SORT CHRONOLOGICALLY
# --------------------------------------------------

print("\nSorting transactions chronologically...")

df = df.sort_values("step").reset_index(drop=True)

print("Sorting completed.")


# ==================================================
# SENDER ACCOUNT HISTORY
# ==================================================

print("\nCalculating sender transaction history...")

# Number of previous transactions by sender
df["orig_previous_transaction_count"] = (
    df.groupby("nameOrig", sort=False).cumcount()
)


# --------------------------------------------------
# HISTORICAL AVERAGE AMOUNT
# --------------------------------------------------

print("Calculating sender historical average amount...")

orig_cumulative_amount = (
    df.groupby("nameOrig", sort=False)["amount"]
    .cumsum()
)

orig_transaction_number = (
    df["orig_previous_transaction_count"]
)

df["orig_historical_avg_amount"] = np.where(
    orig_transaction_number > 0,
    (orig_cumulative_amount - df["amount"])
    / orig_transaction_number,
    df["amount"]
)


# ==================================================
# TRANSACTION AMOUNT DEVIATION
# ==================================================

print("Calculating transaction amount deviation...")

df["orig_amount_deviation_ratio"] = (
    df["amount"]
    / df["orig_historical_avg_amount"]
)

df["orig_amount_deviation_ratio"] = (
    df["orig_amount_deviation_ratio"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(1)
)


# ==================================================
# TRANSACTION FREQUENCY
# ==================================================

print("Calculating sender transaction frequency...")

df["orig_previous_step"] = (
    df.groupby("nameOrig", sort=False)["step"]
    .shift(1)
)

df["orig_step_gap"] = (
    df["step"]
    - df["orig_previous_step"]
)

df["orig_step_gap"] = (
    df["orig_step_gap"]
    .fillna(-1)
)


# ==================================================
# UNIQUE DESTINATION ACCOUNTS
# ==================================================

print("Calculating destination diversity...")

# Create sender-destination pair
pair_first_occurrence = ~df.duplicated(
    subset=["nameOrig", "nameDest"]
)

# Mark first occurrence of every destination for each sender
new_destination_flag = pair_first_occurrence.astype(np.int32)

# Cumulative unique destinations per sender
cumulative_unique_destinations = (
    new_destination_flag
    .groupby(df["nameOrig"], sort=False)
    .cumsum()
)

# Exclude current transaction from history
df["orig_unique_destinations"] = (
    cumulative_unique_destinations
    - new_destination_flag
)

df["orig_unique_destinations"] = (
    df["orig_unique_destinations"]
    .fillna(0)
)


# ==================================================
# DESTINATION ACCOUNT HISTORY
# ==================================================

print("Calculating destination account history...")

df["dest_previous_transaction_count"] = (
    df.groupby("nameDest", sort=False).cumcount()
)


# --------------------------------------------------
# DESTINATION HISTORICAL AVERAGE AMOUNT
# --------------------------------------------------

print("Calculating destination historical average amount...")

dest_cumulative_amount = (
    df.groupby("nameDest", sort=False)["amount"]
    .cumsum()
)

dest_transaction_number = (
    df["dest_previous_transaction_count"]
)

df["dest_historical_avg_amount"] = np.where(
    dest_transaction_number > 0,
    (dest_cumulative_amount - df["amount"])
    / dest_transaction_number,
    df["amount"]
)


# ==================================================
# BEHAVIORAL RISK INDICATORS
# ==================================================

print("Creating behavioral risk indicators...")

# Transaction amount is 5 times or more than
# the sender's historical average

df["unusual_amount_flag"] = (
    df["orig_amount_deviation_ratio"] >= 5
).astype(np.int8)


# New or rarely used sender account

df["new_sender_flag"] = (
    df["orig_previous_transaction_count"] < 2
).astype(np.int8)


# Rapid activity:
# previous transaction occurred within 1 step

df["rapid_transaction_flag"] = (
    (df["orig_step_gap"] >= 0)
    &
    (df["orig_step_gap"] <= 1)
).astype(np.int8)


# ==================================================
# BEHAVIORAL RISK SCORE
# ==================================================

print("Calculating behavioral risk score...")

df["behavioral_risk_score"] = (
    df["unusual_amount_flag"] * 40
    +
    df["new_sender_flag"] * 25
    +
    df["rapid_transaction_flag"] * 20
).astype(np.int16)


# ==================================================
# BEHAVIORAL RISK LEVEL
# ==================================================

print("Assigning behavioral risk levels...")

df["behavioral_risk_level"] = np.select(
    [
        df["behavioral_risk_score"] >= 60,
        df["behavioral_risk_score"] >= 30
    ],
    [
        "HIGH",
        "MEDIUM"
    ],
    default="LOW"
)


# ==================================================
# RESULTS
# ==================================================

print("\n" + "=" * 60)
print("BEHAVIORAL PROFILE SUMMARY")
print("=" * 60)

behavior_summary = (
    df.groupby("isFraud")[
        [
            "orig_previous_transaction_count",
            "orig_amount_deviation_ratio",
            "orig_step_gap",
            "orig_unique_destinations",
            "dest_previous_transaction_count",
            "behavioral_risk_score"
        ]
    ]
    .mean()
)

print("\nAverage Behavioral Features:")
print(behavior_summary.round(3))


# --------------------------------------------------
# RISK DISTRIBUTION
# --------------------------------------------------

print("\nBehavioral Risk Distribution:")

print(
    df["behavioral_risk_level"]
    .value_counts()
)


# --------------------------------------------------
# FRAUD RATE BY RISK LEVEL
# --------------------------------------------------

print("\nFraud Rate by Behavioral Risk Level:")

risk_fraud_rate = (
    df.groupby("behavioral_risk_level")["isFraud"]
    .agg(
        total_transactions="count",
        fraud_transactions="sum",
        fraud_rate="mean"
    )
)

risk_fraud_rate["fraud_rate"] = (
    risk_fraud_rate["fraud_rate"] * 100
)

print(
    risk_fraud_rate
    .sort_values(
        "fraud_rate",
        ascending=False
    )
    .round(4)
)


# ==================================================
# SAMPLE HIGH-RISK TRANSACTIONS
# ==================================================

print("\nSample High Behavioral Risk Transactions:")

high_risk_cases = df[
    df["behavioral_risk_level"] == "HIGH"
]

sample_columns = [
    "step",
    "nameOrig",
    "nameDest",
    "amount",
    "orig_previous_transaction_count",
    "orig_amount_deviation_ratio",
    "orig_step_gap",
    "orig_unique_destinations",
    "behavioral_risk_score",
    "behavioral_risk_level",
    "isFraud"
]

print(
    high_risk_cases[
        sample_columns
    ]
    .head(10)
    .to_string(index=False)
)


# ==================================================
# SAVE BEHAVIORAL DATA
# ==================================================

print("\nSaving behavioral profiles...")

behavioral_columns = [
    "step",
    "nameOrig",
    "nameDest",
    "amount",
    "isFraud",
    "orig_previous_transaction_count",
    "orig_historical_avg_amount",
    "orig_amount_deviation_ratio",
    "orig_step_gap",
    "orig_unique_destinations",
    "dest_previous_transaction_count",
    "dest_historical_avg_amount",
    "unusual_amount_flag",
    "new_sender_flag",
    "rapid_transaction_flag",
    "behavioral_risk_score",
    "behavioral_risk_level"
]

df[
    behavioral_columns
].to_csv(
    "behavioral_profiles.csv",
    index=False
)


print("\nBehavioral profiles saved successfully:")
print("behavioral_profiles.csv")


# ==================================================
# COMPLETION
# ==================================================

print("\n" + "=" * 60)
print("BEHAVIORAL PROFILING COMPLETED SUCCESSFULLY")
print("=" * 60)


print("\nForensic Interpretation:")

print(
    "The behavioral profiling component analyzes historical transaction "
    "patterns rather than relying only on the characteristics of a single "
    "transaction. It evaluates unusual transaction amounts, sender account "
    "history, transaction frequency and destination diversity. These "
    "behavioral indicators provide additional forensic context alongside "
    "the machine learning fraud likelihood score."
)