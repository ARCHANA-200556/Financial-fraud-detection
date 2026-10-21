import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

print("=" * 75)
print("FORENSIC THRESHOLD AND COST ANALYSIS")
print("=" * 75)


# ============================================================
# 1. LOAD RANDOM FOREST
# ============================================================

print("\nLoading trained Random Forest...")

rf_model = joblib.load(
    "random_forest_model.pkl"
)

print("Random Forest loaded successfully.")


# ============================================================
# 2. LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

X_val = pd.read_csv(
    "X_val.csv"
)

y_val = pd.read_csv(
    "y_val.csv"
).squeeze("columns")

print("Validation data loaded.")
print("Validation shape:", X_val.shape)
print("Validation fraud cases:", int(y_val.sum()))
print("Validation genuine cases:", int((y_val == 0).sum()))


# ============================================================
# 3. LOAD FINAL TEST DATA
# ============================================================

print("\nLoading unseen test data...")

X_test = pd.read_csv(
    "X_test.csv"
)

y_test = pd.read_csv(
    "y_test.csv"
).squeeze("columns")

print("Test data loaded.")
print("Test shape:", X_test.shape)
print("Test fraud cases:", int(y_test.sum()))
print("Test genuine cases:", int((y_test == 0).sum()))


# ============================================================
# 4. GENERATE VALIDATION SCORES
# ============================================================

print("\nGenerating validation fraud scores...")

val_probability = rf_model.predict_proba(
    X_val
)[:, 1]

print("Validation fraud scores generated.")


# ============================================================
# 5. FORENSIC COST SETTINGS
# ============================================================

# Missing a fraud case is considered much more serious
# than sending a genuine transaction for investigation.

FALSE_NEGATIVE_COST = 100
FALSE_POSITIVE_COST = 1

print("\n" + "=" * 75)
print("FORENSIC COST ASSUMPTIONS")
print("=" * 75)

print(
    f"\nFalse Negative Cost : {FALSE_NEGATIVE_COST}"
)

print(
    f"False Positive Cost : {FALSE_POSITIVE_COST}"
)

print(
    "\nInterpretation:"
)

print(
    "A missed fraudulent transaction is assigned a higher "
    "forensic cost because undetected fraud represents "
    "potential financial loss and investigative risk."
)

print(
    "A false positive is assigned a lower cost because it "
    "primarily represents additional investigation workload."
)


# ============================================================
# 6. THRESHOLD RANGE
# ============================================================

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)

threshold_results = []


# ============================================================
# 7. EVALUATE EVERY THRESHOLD ON VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("VALIDATION THRESHOLD ANALYSIS")
print("=" * 75)

for threshold in thresholds:

    predictions = (
        val_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions
    ).ravel()

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    forensic_cost = (
        fn * FALSE_NEGATIVE_COST
        +
        fp * FALSE_POSITIVE_COST
    )

    fraud_capture_rate = (
        tp / (tp + fn)
    ) * 100

    false_alert_rate = (
        fp / (fp + tn)
    ) * 100

    threshold_results.append({

        "Threshold": threshold,

        "True_Negatives": tn,

        "False_Positives": fp,

        "False_Negatives": fn,

        "True_Positives": tp,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1,

        "Fraud_Capture_Rate": fraud_capture_rate,

        "False_Alert_Rate": false_alert_rate,

        "Forensic_Cost": forensic_cost

    })


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# 8. DISPLAY VALIDATION RESULTS
# ============================================================

print("\nThreshold Performance:")

display_columns = [

    "Threshold",
    "False_Positives",
    "False_Negatives",
    "True_Positives",
    "Precision",
    "Recall",
    "F1_Score",
    "Fraud_Capture_Rate",
    "False_Alert_Rate",
    "Forensic_Cost"

]

print(
    threshold_df[
        display_columns
    ].round(6).to_string(
        index=False
    )
)


# ============================================================
# 9. SELECT THRESHOLD
# ============================================================

# Primary objective:
# minimize forensic cost.
#
# Secondary objective:
# maximize recall.
#
# Tertiary objective:
# maximize precision.

threshold_df = threshold_df.sort_values(

    by=[
        "Forensic_Cost",
        "False_Negatives",
        "False_Positives"
    ],

    ascending=[
        True,
        True,
        True
    ]

)


selected_threshold = float(
    threshold_df.iloc[0]["Threshold"]
)


# ============================================================
# 10. DISPLAY SELECTED THRESHOLD
# ============================================================

selected_row = threshold_df.iloc[0]


print("\n" + "=" * 75)
print("FORENSIC THRESHOLD SELECTION")
print("=" * 75)

print(
    f"\nSelected Threshold: "
    f"{selected_threshold:.2f}"
)

print(
    f"Validation False Positives: "
    f"{int(selected_row['False_Positives'])}"
)

print(
    f"Validation False Negatives: "
    f"{int(selected_row['False_Negatives'])}"
)

print(
    f"Validation Precision: "
    f"{selected_row['Precision']:.6f}"
)

print(
    f"Validation Recall: "
    f"{selected_row['Recall']:.6f}"
)

print(
    f"Validation F1-Score: "
    f"{selected_row['F1_Score']:.6f}"
)

print(
    f"Validation Fraud Capture Rate: "
    f"{selected_row['Fraud_Capture_Rate']:.4f}%"
)

print(
    f"Validation Forensic Cost: "
    f"{selected_row['Forensic_Cost']:.2f}"
)


# ============================================================
# 11. SAVE VALIDATION THRESHOLD RESULTS
# ============================================================

threshold_df.to_csv(
    "threshold_cost_analysis.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "threshold_cost_analysis.csv"
)


# ============================================================
# 12. LOAD CALIBRATED MODEL
# ============================================================

print("\n" + "=" * 75)
print("LOADING CALIBRATED RANDOM FOREST")
print("=" * 75)

calibrated_model = joblib.load(
    "calibrated_fraud_rf_model.pkl"
)

print(
    "Calibrated Random Forest loaded successfully."
)


# ============================================================
# 13. GENERATE CALIBRATED TEST PROBABILITIES
# ============================================================

print("\nGenerating calibrated test probabilities...")

test_probability = calibrated_model.predict_proba(
    X_test
)[:, 1]

print(
    "Calibrated test probabilities generated."
)


# ============================================================
# 14. APPLY VALIDATION-SELECTED THRESHOLD TO TEST
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)

test_predictions = (
    test_probability >= selected_threshold
).astype(int)


tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions
).ravel()


test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probability
)

test_pr_auc = average_precision_score(
    y_test,
    test_probability
)

test_forensic_cost = (
    fn * FALSE_NEGATIVE_COST
    +
    fp * FALSE_POSITIVE_COST
)

test_capture_rate = (
    tp / (tp + fn)
) * 100

test_false_alert_rate = (
    fp / (fp + tn)
) * 100


# ============================================================
# 15. PRINT FINAL TEST RESULTS
# ============================================================

print(
    f"\nFinal Threshold: "
    f"{selected_threshold:.2f}"
)

print(
    f"\nAccuracy: "
    f"{(tn + tp) / (tn + fp + fn + tp):.6f}"
)

print(
    f"Precision: "
    f"{test_precision:.6f}"
)

print(
    f"Recall: "
    f"{test_recall:.6f}"
)

print(
    f"F1-Score: "
    f"{test_f1:.6f}"
)

print(
    f"ROC-AUC: "
    f"{test_roc_auc:.6f}"
)

print(
    f"PR-AUC: "
    f"{test_pr_auc:.6f}"
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    np.array([
        [tn, fp],
        [fn, tp]
    ])
)


print("\nForensic Case Counts:")

print(
    f"True Negatives  : {tn}"
)

print(
    f"False Positives : {fp}"
)

print(
    f"False Negatives : {fn}"
)

print(
    f"True Positives  : {tp}"
)


print(
    f"\nFraud Capture Rate: "
    f"{test_capture_rate:.4f}%"
)

print(
    f"False Alert Rate: "
    f"{test_false_alert_rate:.6f}%"
)

print(
    f"Forensic Cost: "
    f"{test_forensic_cost:.2f}"
)


# ============================================================
# 17. TEST PROBABILITY SUMMARY
# ============================================================

fraud_probability = test_probability[
    y_test.to_numpy() == 1
]

genuine_probability = test_probability[
    y_test.to_numpy() == 0
]


print("\n" + "=" * 75)
print("TEST FRAUD LIKELIHOOD INTERPRETATION")
print("=" * 75)

print(
    f"\nAverage fraud probability: "
    f"{fraud_probability.mean():.6f}"
)

print(
    f"Average genuine probability: "
    f"{genuine_probability.mean():.6f}"
)

print(
    f"Minimum fraud probability: "
    f"{fraud_probability.min():.6f}"
)

print(
    f"Maximum genuine probability: "
    f"{genuine_probability.max():.6f}"
)


# ============================================================
# 18. SAVE FINAL TEST THRESHOLD RESULTS
# ============================================================

final_test_summary = pd.DataFrame({

    "Metric": [

        "Selected Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "PR-AUC",
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives",
        "Fraud Capture Rate",
        "False Alert Rate",
        "Forensic Cost",
        "Average Fraud Probability",
        "Average Genuine Probability"

    ],

    "Value": [

        selected_threshold,
        (tn + tp) / (tn + fp + fn + tp),
        test_precision,
        test_recall,
        test_f1,
        test_roc_auc,
        test_pr_auc,
        tn,
        fp,
        fn,
        tp,
        test_capture_rate,
        test_false_alert_rate,
        test_forensic_cost,
        fraud_probability.mean(),
        genuine_probability.mean()

    ]

})


final_test_summary.to_csv(
    "final_threshold_test_results.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "final_threshold_test_results.csv"
)


# ============================================================
# 19. GRAPH 1 — F1 / PRECISION / RECALL
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1_Score"],
    marker="o",
    label="F1-Score"
)

plt.axvline(
    selected_threshold,
    linestyle="--",
    label=f"Selected Threshold = {selected_threshold:.2f}"
)

plt.xlabel(
    "Fraud Decision Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Threshold vs Classification Performance"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "threshold_performance.png",
    dpi=300
)

plt.close()


print(
    "\nSaved:"
)

print(
    "threshold_performance.png"
)


# ============================================================
# 20. GRAPH 2 — FALSE POSITIVE / FALSE NEGATIVE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["False_Positives"],
    marker="o",
    label="False Positives"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["False_Negatives"],
    marker="o",
    label="False Negatives"
)

plt.axvline(
    selected_threshold,
    linestyle="--",
    label=f"Selected Threshold = {selected_threshold:.2f}"
)

plt.xlabel(
    "Fraud Decision Threshold"
)

plt.ylabel(
    "Number of Cases"
)

plt.title(
    "Threshold vs Forensic Classification Errors"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "threshold_error_analysis.png",
    dpi=300
)

plt.close()


print(
    "threshold_error_analysis.png"
)


# ============================================================
# 21. GRAPH 3 — FORENSIC COST
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Forensic_Cost"],
    marker="o"
)

plt.axvline(
    selected_threshold,
    linestyle="--",
    label=f"Selected Threshold = {selected_threshold:.2f}"
)

plt.xlabel(
    "Fraud Decision Threshold"
)

plt.ylabel(
    "Forensic Cost"
)

plt.title(
    "Threshold vs Forensic Investigation Cost"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "threshold_forensic_cost.png",
    dpi=300
)

plt.close()


print(
    "threshold_forensic_cost.png"
)


# ============================================================
# 22. FORENSIC INTERPRETATION
# ============================================================

print("\n" + "=" * 75)
print("FORENSIC INTERPRETATION")
print("=" * 75)

print(
    f"""
The fraud decision threshold was analyzed using the chronological
validation period rather than the final test period.

This prevents information from the previously unseen test period
from influencing the operational decision boundary.

A cost-sensitive framework was used because the consequences of
false negatives and false positives are different in a fraud
investigation.

A false negative represents a fraudulent transaction that remains
undetected and therefore receives a higher forensic cost.

A false positive represents a genuine transaction sent for
investigation and therefore contributes primarily to investigation
workload.

The selected threshold was {selected_threshold:.2f} based on the
validation-period forensic cost.

This threshold was then applied unchanged to the previously unseen
test period.

The final test results therefore provide an independent assessment
of the selected forensic decision rule.

The threshold analysis also demonstrates that fraud probability
should not be interpreted in isolation. The operational decision
depends on the selected threshold and the relative consequences of
missed fraud versus unnecessary investigation.

The calibrated fraud probability, behavioral evidence and SHAP
explanations will subsequently be combined to support the final
forensic risk-prioritization decision.
"""
)


print("\n" + "=" * 75)
print("THRESHOLD AND COST ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 75)