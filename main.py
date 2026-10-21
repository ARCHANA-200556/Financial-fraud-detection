import pandas as pd

df = pd.read_csv("paysim.csv")

print("=" * 60)
print("PAYSIM DATASET OVERVIEW")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\n" + "=" * 60)
print("DATA QUALITY CHECK")
print("=" * 60)

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())


print("\n" + "=" * 60)
print("FRAUD CLASS DISTRIBUTION")
print("=" * 60)

fraud_counts = df["isFraud"].value_counts()

print("\nTransaction Counts:")
print(fraud_counts)

print("\nFraud Percentage:")
fraud_percentage = (
    df["isFraud"]
    .value_counts(normalize=True)
    .sort_index() * 100
)

print(fraud_percentage.round(4))

print("\nInterpretation:")

if fraud_percentage.get(1, 0) < 5:
    print(
        "Fraudulent transactions represent a very small proportion "
        "of the dataset. Therefore, the dataset is highly imbalanced."
    )
    print(
        "Accuracy alone will not be sufficient for evaluating fraud "
        "detection models. Precision, recall, F1-score, ROC-AUC and "
        "PR-AUC will be considered."
    )
    
    print("\n" + "=" * 60)
print("FORENSIC ANALYSIS BY TRANSACTION TYPE")
print("=" * 60)

# Total transactions by type
transaction_type_summary = (
    df.groupby("type")["isFraud"]
    .agg(
        total_transactions="count",
        fraud_transactions="sum",
        fraud_rate="mean"
    )
    .sort_values("fraud_rate", ascending=False)
)

# Convert fraud rate to percentage
transaction_type_summary["fraud_rate"] = (
    transaction_type_summary["fraud_rate"] * 100
)

print("\nTransaction Type Summary:")
print(transaction_type_summary.round(4))

# Transaction types containing fraud
fraud_types = transaction_type_summary[
    transaction_type_summary["fraud_transactions"] > 0
]

print("\nTransaction Types With Recorded Fraud:")
print(fraud_types.index.tolist())

print("\nForensic Interpretation:")
print(
    "The fraud distribution across transaction types provides "
    "transaction-level context for the fraud assessment. "
    "A transaction type associated with a higher observed fraud rate "
    "may contribute contextual evidence, but transaction type alone "
    "cannot establish that an individual transaction is fraudulent."
)

print("\n" + "=" * 60)
print("FORENSIC ANALYSIS: AMOUNT AND BALANCE PATTERNS")
print("=" * 60)

# Compare genuine and fraudulent transaction amounts
amount_summary = (
    df.groupby("isFraud")["amount"]
    .agg(
        transaction_count="count",
        mean_amount="mean",
        median_amount="median",
        min_amount="min",
        max_amount="max"
    )
)

print("\nAmount Summary:")
print(amount_summary.round(2))


# Sender balance change
df["orig_balance_change"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

# Destination balance change
df["dest_balance_change"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)

balance_summary = (
    df.groupby("isFraud")[
        [
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "orig_balance_change",
            "dest_balance_change"
        ]
    ]
    .mean()
)

print("\nAverage Balance Pattern Summary:")
print(balance_summary.round(2))

print("\nForensic Interpretation:")
print(
    "The amount and balance patterns of fraudulent and genuine "
    "transactions will be compared to identify characteristics that "
    "may provide evidence for fraud likelihood. These characteristics "
    "will later be used as model features and interpreted together "
    "with the model's fraud likelihood score."
)
print("\n" + "=" * 60)
print("FORENSIC ANALYSIS: BALANCE CONSISTENCY")
print("=" * 60)

# Expected sender balance after transaction
df["expected_newbalanceOrig"] = (
    df["oldbalanceOrg"] - df["amount"]
)

# Difference between expected and actual sender balance
df["orig_balance_error"] = (
    df["newbalanceOrig"] - df["expected_newbalanceOrig"]
)

# Expected destination balance after transaction
df["expected_newbalanceDest"] = (
    df["oldbalanceDest"] + df["amount"]
)

# Difference between expected and actual destination balance
df["dest_balance_error"] = (
    df["newbalanceDest"] - df["expected_newbalanceDest"]
)

# Absolute errors for easier analysis
df["abs_orig_balance_error"] = df["orig_balance_error"].abs()
df["abs_dest_balance_error"] = df["dest_balance_error"].abs()

# Compare fraud and genuine transactions
balance_error_summary = (
    df.groupby("isFraud")[
        ["abs_orig_balance_error", "abs_dest_balance_error"]
    ]
    .agg(["mean", "median", "max"])
)

print("\nBalance Consistency Error Summary:")
print(balance_error_summary.round(2))

print("\nForensic Interpretation:")
print(
    "Balance consistency errors measure the difference between the "
    "expected account balance after a transaction and the recorded "
    "balance in the dataset. Large errors may indicate unusual "
    "transaction or balance behavior and can be used as supporting "
    "forensic evidence rather than standalone proof of fraud."
)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING FOR FRAUD AND FORENSIC ANALYSIS")
print("=" * 60)

# 1. Amount relative to sender's available balance
df["amount_to_orig_balance_ratio"] = (
    df["amount"] / df["oldbalanceOrg"].replace(0, float("nan"))
)

# 2. Sender balance depletion ratio
df["orig_balance_depletion_ratio"] = (
    (df["oldbalanceOrg"] - df["newbalanceOrig"])
    / df["oldbalanceOrg"].replace(0, float("nan"))
)

# Replace undefined ratios caused by zero original balance
df["amount_to_orig_balance_ratio"] = (
    df["amount_to_orig_balance_ratio"]
    .replace([float("inf"), -float("inf")], float("nan"))
    .fillna(0)
)

df["orig_balance_depletion_ratio"] = (
    df["orig_balance_depletion_ratio"]
    .replace([float("inf"), -float("inf")], float("nan"))
    .fillna(0)
)

# Compare engineered features for genuine and fraudulent transactions
feature_summary = (
    df.groupby("isFraud")[
        [
            "amount_to_orig_balance_ratio",
            "orig_balance_depletion_ratio",
            "abs_orig_balance_error",
            "abs_dest_balance_error"
        ]
    ]
    .agg(["mean", "median"])
)

print("\nEngineered Feature Summary:")
print(feature_summary.round(4))

print("\nForensic Interpretation:")
print(
    "The engineered features describe how the transaction amount "
    "relates to the sender's available balance and how account balances "
    "change during the transaction. Their usefulness will not be assumed "
    "from summary statistics alone. The final importance of these features "
    "will be validated through model performance and SHAP explanations."
)

print("\n" + "=" * 60)
print("ML DATASET PREPARATION")
print("=" * 60)

# Check the existing rule-based flag
print("\nisFlaggedFraud Distribution:")
print(df["isFlaggedFraud"].value_counts())

# Features selected for initial ML modelling
feature_columns = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_change",
    "dest_balance_change",
    "abs_orig_balance_error",
    "abs_dest_balance_error",
    "amount_to_orig_balance_ratio",
    "orig_balance_depletion_ratio"
]

target_column = "isFraud"

X = df[feature_columns].copy()
y = df[target_column].copy()

# One-hot encode transaction type
X = pd.get_dummies(
    X,
    columns=["type"],
    drop_first=False,
    dtype=int
)

print("\nML Feature Shape:")
print(X.shape)

print("\nML Features:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts())

print("\nForensic Design Note:")
print(
    "Raw account identifiers are excluded from the initial detection model "
    "to reduce identifier memorization. Account history will be handled "
    "separately in the behavioral profiling component."
)

print("\n" + "=" * 60)
print("CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

# Sort transactions by chronological step
model_data = df[feature_columns + [target_column]].copy()
model_data = model_data.sort_values("step").reset_index(drop=True)

# Recreate ML features
X = model_data[feature_columns].copy()
y = model_data[target_column].copy()

# One-hot encode transaction type
X = pd.get_dummies(
    X,
    columns=["type"],
    drop_first=False,
    dtype=int
)

# 70% Train, 15% Validation, 15% Test
n = len(X)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end].copy()
y_train = y.iloc[:train_end].copy()

X_val = X.iloc[train_end:val_end].copy()
y_val = y.iloc[train_end:val_end].copy()

X_test = X.iloc[val_end:].copy()
y_test = y.iloc[val_end:].copy()

print("\nTraining Set:")
print(X_train.shape)
print(y_train.value_counts())

print("\nValidation Set:")
print(X_val.shape)
print(y_val.value_counts())

print("\nTest Set:")
print(X_test.shape)
print(y_test.value_counts())

print("\nSplit Interpretation:")
print(
    "The training data contains earlier transactions, while validation "
    "and test data contain later transactions. This design better reflects "
    "a fraud detection scenario where a model is evaluated on future "
    "transaction activity."
)

print("\n" + "=" * 60)
print("CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

# Use the already prepared ML feature matrix X
# Sort data by transaction step while keeping features and target aligned

sorted_indices = df.sort_values("step").index

X_sorted = X.loc[sorted_indices].reset_index(drop=True)
y_sorted = y.loc[sorted_indices].reset_index(drop=True)

# 70% Training, 15% Validation, 15% Testing
n = len(X_sorted)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X_sorted.iloc[:train_end].copy()
y_train = y_sorted.iloc[:train_end].copy()

X_val = X_sorted.iloc[train_end:val_end].copy()
y_val = y_sorted.iloc[train_end:val_end].copy()

X_test = X_sorted.iloc[val_end:].copy()
y_test = y_sorted.iloc[val_end:].copy()

print("\nTraining Set:")
print("Shape:", X_train.shape)
print(y_train.value_counts())

print("\nValidation Set:")
print("Shape:", X_val.shape)
print(y_val.value_counts())

print("\nTest Set:")
print("Shape:", X_test.shape)
print(y_test.value_counts())

print("\nSplit Interpretation:")
print(
    "The dataset is divided chronologically using the transaction step. "
    "Earlier transactions are used for training, while later transactions "
    "are reserved for validation and final testing."
)

print("\n" + "=" * 60)
print("FRAUD DISTRIBUTION ACROSS DATA SPLITS")
print("=" * 60)

split_summary = pd.DataFrame({
    "Dataset": ["Training", "Validation", "Test"],
    "Total_Transactions": [
        len(y_train),
        len(y_val),
        len(y_test)
    ],
    "Fraud_Transactions": [
        y_train.sum(),
        y_val.sum(),
        y_test.sum()
    ]
})

split_summary["Fraud_Rate_Percentage"] = (
    split_summary["Fraud_Transactions"]
    / split_summary["Total_Transactions"]
    * 100
)

print(split_summary.round(4).to_string(index=False))

print("\nForensic Interpretation:")

print(
    "The fraud rate differs across chronological data splits. "
    "Therefore, model performance should be evaluated on the later "
    "test period rather than relying only on performance within the "
    "training period. This helps examine whether the detection approach "
    "remains effective when transaction patterns change over time."
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

print("\n" + "=" * 60)
print("BASELINE RANDOM FOREST MODEL")
print("=" * 60)

# Random Forest baseline with class weighting
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

rf_model.fit(X_train, y_train)

print("Training completed.")

# Validation predictions
y_val_pred = rf_model.predict(X_val)
y_val_prob = rf_model.predict_proba(X_val)[:, 1]

# Evaluation metrics
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, zero_division=0)
recall = recall_score(y_val, y_val_pred, zero_division=0)
f1 = f1_score(y_val, y_val_pred, zero_division=0)
roc_auc = roc_auc_score(y_val, y_val_prob)
pr_auc = average_precision_score(y_val, y_val_prob)

print("\nVALIDATION RESULTS")
print("-" * 60)

print(f"Accuracy : {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall   : {recall:.6f}")
print(f"F1-Score : {f1:.6f}")
print(f"ROC-AUC  : {roc_auc:.6f}")
print(f"PR-AUC   : {pr_auc:.6f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

print("\nForensic Interpretation:")
print(
    "The baseline model produces a fraud likelihood score for each "
    "transaction. The current classification results use the default "
    "0.50 probability threshold. In the next step, different thresholds "
    "will be evaluated to determine an operational decision boundary "
    "based on the trade-off between missed fraud and false alerts."
)
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE AND LEAKAGE CHECK")
print("=" * 60)

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print(feature_importance.to_string(index=False))

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("FORENSIC FEATURE ABLATION TEST")
print("=" * 60)

# Remove the most dominant balance-derived feature
reduced_features = [
    feature for feature in X_train.columns
    if feature != "abs_orig_balance_error"
]

X_train_reduced = X_train[reduced_features]
X_val_reduced = X_val[reduced_features]

rf_reduced_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest without abs_orig_balance_error...")

rf_reduced_model.fit(X_train_reduced, y_train)

# Predictions
y_val_pred_reduced = rf_reduced_model.predict(X_val_reduced)
y_val_prob_reduced = rf_reduced_model.predict_proba(X_val_reduced)[:, 1]

# Metrics
accuracy_reduced = accuracy_score(y_val, y_val_pred_reduced)
precision_reduced = precision_score(
    y_val,
    y_val_pred_reduced,
    zero_division=0
)
recall_reduced = recall_score(
    y_val,
    y_val_pred_reduced,
    zero_division=0
)
f1_reduced = f1_score(
    y_val,
    y_val_pred_reduced,
    zero_division=0
)
roc_auc_reduced = roc_auc_score(
    y_val,
    y_val_prob_reduced
)
pr_auc_reduced = average_precision_score(
    y_val,
    y_val_prob_reduced
)

print("\nREDUCED FEATURE MODEL RESULTS")
print("-" * 60)

print(f"Accuracy : {accuracy_reduced:.6f}")
print(f"Precision: {precision_reduced:.6f}")
print(f"Recall   : {recall_reduced:.6f}")
print(f"F1-Score : {f1_reduced:.6f}")
print(f"ROC-AUC  : {roc_auc_reduced:.6f}")
print(f"PR-AUC   : {pr_auc_reduced:.6f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_reduced))

print("\n" + "=" * 60)
print("FORENSIC ROBUSTNESS TEST: ORIGINAL FEATURES")
print("=" * 60)

# Use a simpler feature set based primarily on original dataset columns
robust_features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]

X_train_robust = X_train[robust_features]
X_val_robust = X_val[robust_features]

rf_robust_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest using original transaction features...")

rf_robust_model.fit(X_train_robust, y_train)

# Predictions
y_val_pred_robust = rf_robust_model.predict(X_val_robust)
y_val_prob_robust = rf_robust_model.predict_proba(X_val_robust)[:, 1]

# Metrics
accuracy_robust = accuracy_score(y_val, y_val_pred_robust)
precision_robust = precision_score(
    y_val,
    y_val_pred_robust,
    zero_division=0
)
recall_robust = recall_score(
    y_val,
    y_val_pred_robust,
    zero_division=0
)
f1_robust = f1_score(
    y_val,
    y_val_pred_robust,
    zero_division=0
)
roc_auc_robust = roc_auc_score(
    y_val,
    y_val_prob_robust
)
pr_auc_robust = average_precision_score(
    y_val,
    y_val_prob_robust
)

print("\nORIGINAL FEATURE MODEL RESULTS")
print("-" * 60)

print(f"Accuracy : {accuracy_robust:.6f}")
print(f"Precision: {precision_robust:.6f}")
print(f"Recall   : {recall_robust:.6f}")
print(f"F1-Score : {f1_robust:.6f}")
print(f"ROC-AUC  : {roc_auc_robust:.6f}")
print(f"PR-AUC   : {pr_auc_robust:.6f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_robust))

print("\nForensic Interpretation:")
print(
    "This experiment removes the engineered balance-consistency and "
    "ratio features. Comparing its performance with the full model "
    "shows whether fraud detection depends primarily on engineered "
    "features or whether the original transaction information itself "
    "contains sufficient evidence."
)
import numpy as np

print("\n" + "=" * 60)
print("FRAUD LIKELIHOOD SCORE ANALYSIS")
print("=" * 60)

# Use probabilities from the full Random Forest model
# y_val_prob was already generated from rf_model

fraud_cases = y_val_prob[y_val.to_numpy() == 1]
genuine_cases = y_val_prob[y_val.to_numpy() == 0]

print("\nFraud Likelihood Score Summary")

print("\nActual Fraud Transactions:")
print(pd.Series(fraud_cases).describe())

print("\nActual Genuine Transactions:")
print(pd.Series(genuine_cases).describe())

# Define forensic risk levels
def risk_level(probability):
    if probability >= 0.90:
        return "CRITICAL"
    elif probability >= 0.70:
        return "HIGH"
    elif probability >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"

# Example transactions from validation data
validation_cases = X_val.copy()
validation_cases["actual_fraud"] = y_val.to_numpy()
validation_cases["fraud_probability"] = y_val_prob
validation_cases["risk_level"] = validation_cases[
    "fraud_probability"
].apply(risk_level)

print("\nRisk Level Distribution:")
print(validation_cases["risk_level"].value_counts())

print("\nSample High-Risk Transactions:")
print(
    validation_cases[
        validation_cases["risk_level"].isin(["HIGH", "CRITICAL"])
    ]
    .sort_values("fraud_probability", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\nForensic Interpretation:")
print(
    "Instead of treating the model output as only a binary decision, "
    "each transaction is assigned a fraud likelihood score and a risk "
    "level. HIGH and CRITICAL scores can be prioritized for forensic "
    "investigation, while MEDIUM-risk transactions may require "
    "additional evidence or monitoring."
)
print("\n" + "=" * 60)
print("FRAUD DECISION THRESHOLD ANALYSIS")
print("=" * 60)

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

threshold_results = []

for threshold in thresholds:

    threshold_predictions = (
        y_val_prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        threshold_predictions
    ).ravel()

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_val,
            threshold_predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_val,
            threshold_predictions,
            zero_division=0
        ),
        "F1_Score": f1_score(
            y_val,
            threshold_predictions,
            zero_division=0
        ),
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nThreshold Comparison:")
print(threshold_df.round(6).to_string(index=False))

# Find threshold with highest F1-score
best_threshold_row = threshold_df.loc[
    threshold_df["F1_Score"].idxmax()
]

best_threshold = best_threshold_row["Threshold"]

print("\nBest Threshold Based on Validation F1-Score:")
print(best_threshold)

print("\nBest Threshold Result:")
print(best_threshold_row)

print("\nForensic Interpretation:")
print(
    "The decision threshold is selected using validation data rather "
    "than assuming 0.50 is optimal. The threshold determines the "
    "trade-off between false alerts and missed fraud cases. In a "
    "forensic investigation setting, the preferred threshold can also "
    "be selected based on the relative cost of missing a fraudulent "
    "transaction versus investigating a genuine transaction."
)

print("\n" + "=" * 60)
print("FINAL TEST SET EVALUATION")
print("=" * 60)

# Use the final trained Random Forest model
# Generate fraud likelihood scores for the unseen test data
y_test_prob = rf_model.predict_proba(X_test)[:, 1]

# Apply the threshold selected from validation data
final_threshold = best_threshold

y_test_pred = (
    y_test_prob >= final_threshold
).astype(int)

# Calculate final test metrics
test_accuracy = accuracy_score(y_test, y_test_pred)

test_precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

test_roc_auc = roc_auc_score(
    y_test,
    y_test_prob
)

test_pr_auc = average_precision_score(
    y_test,
    y_test_prob
)

# Confusion matrix
test_cm = confusion_matrix(
    y_test,
    y_test_pred
)

tn, fp, fn, tp = test_cm.ravel()

print("\nFinal Threshold Used:")
print(final_threshold)

print("\nFINAL TEST RESULTS")
print("-" * 60)

print(f"Accuracy : {test_accuracy:.6f}")
print(f"Precision: {test_precision:.6f}")
print(f"Recall   : {test_recall:.6f}")
print(f"F1-Score : {test_f1:.6f}")
print(f"ROC-AUC  : {test_roc_auc:.6f}")
print(f"PR-AUC   : {test_pr_auc:.6f}")

print("\nFinal Confusion Matrix:")
print(test_cm)

print("\nCase Investigation Summary:")
print(f"True Negatives  : {tn}")
print(f"False Positives : {fp}")
print(f"False Negatives : {fn}")
print(f"True Positives  : {tp}")

print("\nForensic Interpretation:")
print(
    "The final evaluation uses the later, previously unseen test period. "
    "The threshold was selected on validation data and is not re-optimized "
    "using the test set. Therefore, the test results provide a more realistic "
    "assessment of how the fraud detection system may perform on future "
    "transaction activity."
)

print(
    f"\nAt the selected threshold of {final_threshold}, "
    f"the system correctly identified {tp} fraudulent transactions "
    f"and missed {fn} fraudulent transactions."
)

print(
    f"The model also generated {fp} false alerts, meaning genuine "
    f"transactions incorrectly classified as potentially fraudulent."
)

print(
    "These counts are important for forensic decision-making because "
    "false negatives represent potentially missed fraud cases, whereas "
    "false positives represent investigation workload."
)

print("\n" + "=" * 60)
print("FORENSIC INVESTIGATION OF MISSED FRAUD CASES")
print("=" * 60)

# Create a forensic investigation table for the test set
test_cases = X_test.copy()

test_cases["actual_fraud"] = y_test.to_numpy()
test_cases["fraud_probability"] = y_test_prob
test_cases["predicted_class"] = y_test_pred

# Identify fraudulent transactions missed by the model
missed_fraud = test_cases[
    (test_cases["actual_fraud"] == 1) &
    (test_cases["predicted_class"] == 0)
]

print("\nNumber of Missed Fraud Cases:")
print(len(missed_fraud))

print("\nMissed Fraud Transactions:")
print(
    missed_fraud.sort_values(
        "fraud_probability",
        ascending=False
    ).to_string(index=False)
)

print("\nFraud Likelihood Scores of Missed Cases:")
print(
    missed_fraud[
        [
            "fraud_probability",
            "actual_fraud",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "orig_balance_change",
            "abs_orig_balance_error",
            "amount_to_orig_balance_ratio",
            "orig_balance_depletion_ratio"
        ]
    ].to_string(index=False)
)

print("\nForensic Interpretation:")
print(
    "These transactions were confirmed as fraudulent in the dataset, "
    "but their predicted fraud likelihood remained below the selected "
    "decision threshold. Therefore, they represent false negatives."
)

print(
    "The forensic investigation should examine which transaction and "
    "balance characteristics caused these cases to receive lower fraud "
    "likelihood scores. This analysis can help identify limitations of "
    "the current detection model."
)
import joblib

joblib.dump(rf_model, "fraud_rf_model.pkl")

X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nModel and test data saved successfully.")