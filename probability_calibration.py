import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve
)

from sklearn.frozen import FrozenEstimator

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


print("=" * 70)
print("FRAUD PROBABILITY CALIBRATION")
print("=" * 70)


# ============================================================
# 1. LOAD TRAINED RANDOM FOREST
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


# ============================================================
# 3. LOAD UNSEEN TEST DATA
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


# ============================================================
# 4. ORIGINAL RANDOM FOREST PROBABILITIES
# ============================================================

print("\nGenerating original Random Forest probabilities...")

val_prob_original = (
    rf_model.predict_proba(X_val)[:, 1]
)

test_prob_original = (
    rf_model.predict_proba(X_test)[:, 1]
)

print("Original probabilities generated.")


# ============================================================
# 5. ORIGINAL BRIER SCORE
# ============================================================

print("\nCalculating original probability quality...")

original_val_brier = brier_score_loss(
    y_val,
    val_prob_original
)

original_test_brier = brier_score_loss(
    y_test,
    test_prob_original
)

print(
    f"Original validation Brier score: "
    f"{original_val_brier:.8f}"
)

print(
    f"Original test Brier score: "
    f"{original_test_brier:.8f}"
)


# ============================================================
# 6. CALIBRATE USING VALIDATION DATA
# ============================================================

print("\nCreating calibration model...")

calibrated_model = CalibratedClassifierCV(
    estimator=FrozenEstimator(rf_model),
    method="sigmoid"
)

print(
    "Calibration method: Sigmoid / Platt scaling"
)

print(
    "Fitting calibrator using validation data..."
)

calibrated_model.fit(
    X_val,
    y_val
)

print(
    "Probability calibration completed."
)


# ============================================================
# 7. CALIBRATED VALIDATION PROBABILITIES
# ============================================================

print("\nGenerating calibrated validation probabilities...")

val_prob_calibrated = (
    calibrated_model.predict_proba(X_val)[:, 1]
)

print(
    "Calibrated validation probabilities generated."
)


# ============================================================
# 8. CALIBRATED TEST PROBABILITIES
# ============================================================

print("\nGenerating calibrated test probabilities...")

test_prob_calibrated = (
    calibrated_model.predict_proba(X_test)[:, 1]
)

print(
    "Calibrated test probabilities generated."
)


# ============================================================
# 9. BRIER SCORE AFTER CALIBRATION
# ============================================================

calibrated_val_brier = brier_score_loss(
    y_val,
    val_prob_calibrated
)

calibrated_test_brier = brier_score_loss(
    y_test,
    test_prob_calibrated
)


print("\n" + "=" * 70)
print("PROBABILITY CALIBRATION RESULTS")
print("=" * 70)

print(
    f"\nValidation Brier Score - Before: "
    f"{original_val_brier:.8f}"
)

print(
    f"Validation Brier Score - After : "
    f"{calibrated_val_brier:.8f}"
)

print(
    f"\nTest Brier Score - Before: "
    f"{original_test_brier:.8f}"
)

print(
    f"Test Brier Score - After : "
    f"{calibrated_test_brier:.8f}"
)


# ============================================================
# 10. TEST CLASSIFICATION USING ORIGINAL SCORE
# ============================================================

threshold = 0.60

test_pred_original = (
    test_prob_original >= threshold
).astype(int)

test_pred_calibrated = (
    test_prob_calibrated >= threshold
).astype(int)


# ============================================================
# 11. ORIGINAL TEST METRICS
# ============================================================

original_precision = precision_score(
    y_test,
    test_pred_original,
    zero_division=0
)

original_recall = recall_score(
    y_test,
    test_pred_original,
    zero_division=0
)

original_f1 = f1_score(
    y_test,
    test_pred_original,
    zero_division=0
)

original_roc_auc = roc_auc_score(
    y_test,
    test_prob_original
)

original_pr_auc = average_precision_score(
    y_test,
    test_prob_original
)


# ============================================================
# 12. CALIBRATED TEST METRICS
# ============================================================

calibrated_precision = precision_score(
    y_test,
    test_pred_calibrated,
    zero_division=0
)

calibrated_recall = recall_score(
    y_test,
    test_pred_calibrated,
    zero_division=0
)

calibrated_f1 = f1_score(
    y_test,
    test_pred_calibrated,
    zero_division=0
)

calibrated_roc_auc = roc_auc_score(
    y_test,
    test_prob_calibrated
)

calibrated_pr_auc = average_precision_score(
    y_test,
    test_prob_calibrated
)


# ============================================================
# 13. CONFUSION MATRICES
# ============================================================

original_cm = confusion_matrix(
    y_test,
    test_pred_original
)

calibrated_cm = confusion_matrix(
    y_test,
    test_pred_calibrated
)


# ============================================================
# 14. PRINT COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Metric": [
        "Brier Score",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "PR-AUC"
    ],

    "Original_RF": [
        original_test_brier,
        original_precision,
        original_recall,
        original_f1,
        original_roc_auc,
        original_pr_auc
    ],

    "Calibrated_RF": [
        calibrated_test_brier,
        calibrated_precision,
        calibrated_recall,
        calibrated_f1,
        calibrated_roc_auc,
        calibrated_pr_auc
    ]

})


print("\n" + "=" * 70)
print("ORIGINAL vs CALIBRATED RANDOM FOREST")
print("=" * 70)

print(
    comparison.round(8).to_string(
        index=False
    )
)


# ============================================================
# 15. CONFUSION MATRIX COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL RANDOM FOREST CONFUSION MATRIX")
print("=" * 70)

print(original_cm)


print("\n" + "=" * 70)
print("CALIBRATED RANDOM FOREST CONFUSION MATRIX")
print("=" * 70)

print(calibrated_cm)


# ============================================================
# 16. FRAUD SCORE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("FRAUD LIKELIHOOD SCORE SUMMARY")
print("=" * 70)


fraud_original = test_prob_original[
    y_test.to_numpy() == 1
]

genuine_original = test_prob_original[
    y_test.to_numpy() == 0
]

fraud_calibrated = test_prob_calibrated[
    y_test.to_numpy() == 1
]

genuine_calibrated = test_prob_calibrated[
    y_test.to_numpy() == 0
]


print("\nOriginal Random Forest")

print(
    "Mean fraud score:",
    round(
        fraud_original.mean(),
        6
    )
)

print(
    "Mean genuine score:",
    round(
        genuine_original.mean(),
        6
    )
)


print("\nCalibrated Random Forest")

print(
    "Mean fraud probability:",
    round(
        fraud_calibrated.mean(),
        6
    )
)

print(
    "Mean genuine probability:",
    round(
        genuine_calibrated.mean(),
        6
    )
)


# ============================================================
# 17. CALIBRATION CURVE
# ============================================================

print("\nGenerating calibration curve...")


fraction_of_positives_original, mean_predicted_value_original = (
    calibration_curve(
        y_test,
        test_prob_original,
        n_bins=10,
        strategy="quantile"
    )
)


fraction_of_positives_calibrated, mean_predicted_value_calibrated = (
    calibration_curve(
        y_test,
        test_prob_calibrated,
        n_bins=10,
        strategy="quantile"
    )
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    mean_predicted_value_original,
    fraction_of_positives_original,
    marker="o",
    label="Original Random Forest"
)

plt.plot(
    mean_predicted_value_calibrated,
    fraction_of_positives_calibrated,
    marker="o",
    label="Calibrated Random Forest"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel(
    "Mean Predicted Probability"
)

plt.ylabel(
    "Observed Fraud Frequency"
)

plt.title(
    "Fraud Probability Calibration"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "probability_calibration_curve.png",
    dpi=300
)

plt.close()


print(
    "\nSaved:"
)

print(
    "probability_calibration_curve.png"
)


# ============================================================
# 18. SAVE CALIBRATED MODEL
# ============================================================

joblib.dump(
    calibrated_model,
    "calibrated_fraud_rf_model.pkl"
)

print(
    "\nCalibrated model saved:"
)

print(
    "calibrated_fraud_rf_model.pkl"
)


# ============================================================
# 19. SAVE TEST SCORES
# ============================================================

calibration_results = pd.DataFrame({

    "actual_fraud": y_test.to_numpy(),

    "original_fraud_score":
        test_prob_original,

    "calibrated_fraud_probability":
        test_prob_calibrated,

    "original_prediction":
        test_pred_original,

    "calibrated_prediction":
        test_pred_calibrated

})


calibration_results.to_csv(
    "calibrated_test_results.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "calibrated_test_results.csv"
)


# ============================================================
# 20. FORENSIC INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("FORENSIC INTERPRETATION")
print("=" * 70)

print(
    """
The Random Forest produces a fraud likelihood score for each
transaction. However, a model score should not automatically
be interpreted as a perfectly calibrated probability.

Therefore, a separate probability calibration stage was applied
using the validation period, which was kept independent from the
original model-training data.

The calibrated output provides a more defensible basis for
interpreting fraud likelihood.

The previously unseen test period remains reserved for final
evaluation.

The Brier score is used to assess the quality of the predicted
probabilities, while precision, recall, F1-score, ROC-AUC and PR-AUC
continue to evaluate fraud detection performance.

The calibration curve compares predicted fraud probabilities with
the observed fraud frequency. A model closer to the diagonal
represents better probability calibration.
"""
)


print("\n" + "=" * 70)
print("PROBABILITY CALIBRATION COMPLETED SUCCESSFULLY")
print("=" * 70)