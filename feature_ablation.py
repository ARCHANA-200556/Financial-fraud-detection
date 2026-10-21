import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

print("=" * 75)
print("FEATURE ABLATION STUDY")
print("EVALUATING THE CONTRIBUTION OF FORENSIC FEATURES")
print("=" * 75)


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

print("\nLoading training data...")

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze("columns")

print("Training data loaded.")
print("Training shape:", X_train.shape)
print("Training fraud cases:", int(y_train.sum()))


# ============================================================
# 2. LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

X_val = pd.read_csv("X_val.csv")
y_val = pd.read_csv("y_val.csv").squeeze("columns")

print("Validation data loaded.")
print("Validation shape:", X_val.shape)
print("Validation fraud cases:", int(y_val.sum()))


# ============================================================
# 3. CREATE CONTROLLED TRAINING SAMPLE
# ============================================================

print("\n" + "=" * 75)
print("CREATING CONTROLLED TRAINING SAMPLE")
print("=" * 75)

fraud_train = X_train[y_train == 1].copy()
fraud_y = y_train[y_train == 1].copy()

genuine_train = X_train[y_train == 0].copy()
genuine_y = y_train[y_train == 0].copy()

print("\nAvailable fraud cases:", len(fraud_train))
print("Available genuine cases:", len(genuine_train))


RANDOM_STATE = 42

GENUINE_SAMPLE_SIZE = 500000

print(
    "\nSelecting",
    GENUINE_SAMPLE_SIZE,
    "genuine transactions..."
)

genuine_sample = genuine_train.sample(
    n=GENUINE_SAMPLE_SIZE,
    random_state=RANDOM_STATE
)

genuine_sample_y = genuine_y.loc[
    genuine_sample.index
]


X_controlled = pd.concat(
    [
        genuine_sample,
        fraud_train
    ],
    axis=0
)

y_controlled = pd.concat(
    [
        genuine_sample_y,
        fraud_y
    ],
    axis=0
)


# Shuffle

rng = np.random.RandomState(
    RANDOM_STATE
)

shuffle_indices = rng.permutation(
    len(X_controlled)
)

X_controlled = X_controlled.iloc[
    shuffle_indices
].reset_index(drop=True)

y_controlled = y_controlled.iloc[
    shuffle_indices
].reset_index(drop=True)


print("\nControlled training sample:")
print("Shape:", X_controlled.shape)

print("\nClass distribution:")
print(y_controlled.value_counts())


# ============================================================
# 4. FEATURE GROUPS
# ============================================================

print("\n" + "=" * 75)
print("DEFINING FEATURE GROUPS")
print("=" * 75)


# Original transaction features

original_features = [
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


# Balance consistency features

balance_features = [
    "orig_balance_change",
    "dest_balance_change",
    "abs_orig_balance_error",
    "abs_dest_balance_error"
]


# Ratio features

ratio_features = [
    "amount_to_orig_balance_ratio",
    "orig_balance_depletion_ratio"
]


# Keep only columns that actually exist

original_features = [
    feature
    for feature in original_features
    if feature in X_train.columns
]

balance_features = [
    feature
    for feature in balance_features
    if feature in X_train.columns
]

ratio_features = [
    feature
    for feature in ratio_features
    if feature in X_train.columns
]


print(
    "\nOriginal features:",
    len(original_features)
)

print(
    "Balance features:",
    len(balance_features)
)

print(
    "Ratio features:",
    len(ratio_features)
)


# ============================================================
# 5. DEFINE EXPERIMENTS
# ============================================================

experiments = {

    "Original Features":
        original_features,

    "Original + Balance":
        original_features + balance_features,

    "Original + Balance + Ratios":
        original_features
        + balance_features
        + ratio_features,

    "Full Engineered Features":
        list(X_train.columns)

}


print("\n" + "=" * 75)
print("ABLATION EXPERIMENTS")
print("=" * 75)


for experiment_name, features in experiments.items():

    print(
        f"\n{experiment_name}: "
        f"{len(features)} features"
    )


# ============================================================
# 6. RUN EXPERIMENTS
# ============================================================

results = []


for experiment_name, features in experiments.items():

    print("\n")
    print("=" * 75)
    print(experiment_name.upper())
    print("=" * 75)


    # --------------------------------------------------------
    # Prepare experiment data
    # --------------------------------------------------------

    X_train_exp = X_controlled[
        features
    ]

    X_val_exp = X_val[
        features
    ]


    print(
        "\nTraining shape:",
        X_train_exp.shape
    )

    print(
        "Validation shape:",
        X_val_exp.shape
    )


    # --------------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------------

    print(
        "\nTraining Random Forest..."
    )


    model = RandomForestClassifier(

        n_estimators=100,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1

    )


    model.fit(
        X_train_exp,
        y_controlled
    )


    print(
        "Training completed."
    )


    # --------------------------------------------------------
    # Generate validation probabilities
    # --------------------------------------------------------

    print(
        "\nGenerating validation probabilities..."
    )


    probability = model.predict_proba(
        X_val_exp
    )[:, 1]


    # Use existing decision threshold for
    # controlled feature comparison.

    threshold = 0.60


    prediction = (
        probability >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        prediction
    )

    precision = precision_score(
        y_val,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        prediction,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_val,
        probability
    )

    pr_auc = average_precision_score(
        y_val,
        probability
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_val,
        prediction
    )


    tn, fp, fn, tp = cm.ravel()


    fraud_capture_rate = (
        tp / (tp + fn)
    ) * 100


    false_alert_rate = (
        fp / (fp + tn)
    ) * 100


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nVALIDATION RESULTS")
    print("-" * 75)

    print(
        f"Accuracy : {accuracy:.6f}"
    )

    print(
        f"Precision: {precision:.6f}"
    )

    print(
        f"Recall   : {recall:.6f}"
    )

    print(
        f"F1-Score : {f1:.6f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.6f}"
    )

    print(
        f"PR-AUC   : {pr_auc:.6f}"
    )


    print("\nConfusion Matrix:")
    print(cm)


    print("\nForensic Case Counts:")

    print(
        "True Negatives :",
        tn
    )

    print(
        "False Positives:",
        fp
    )

    print(
        "False Negatives:",
        fn
    )

    print(
        "True Positives :",
        tp
    )


    print(
        f"\nFraud Capture Rate: "
        f"{fraud_capture_rate:.4f}%"
    )

    print(
        f"False Alert Rate: "
        f"{false_alert_rate:.6f}%"
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({

        "Experiment": experiment_name,

        "Feature_Count": len(features),

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1,

        "ROC_AUC": roc_auc,

        "PR_AUC": pr_auc,

        "False_Positives": int(fp),

        "False_Negatives": int(fn),

        "True_Positives": int(tp),

        "True_Negatives": int(tn),

        "Fraud_Capture_Rate": fraud_capture_rate,

        "False_Alert_Rate": false_alert_rate

    })


    # Free model before next experiment

    del model

    del X_train_exp

    del X_val_exp

    del probability

    del prediction


# ============================================================
# 7. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 75)
print("FINAL FEATURE ABLATION COMPARISON")
print("=" * 75)


print(
    results_df.round(6).to_string(
        index=False
    )
)


# ============================================================
# 8. BASELINE COMPARISON
# ============================================================

baseline = results_df.iloc[0]

baseline_f1 = baseline["F1_Score"]
baseline_pr_auc = baseline["PR_AUC"]


print("\n" + "=" * 75)
print("FEATURE ENGINEERING IMPROVEMENT")
print("=" * 75)


for _, row in results_df.iterrows():

    f1_change = (
        row["F1_Score"]
        - baseline_f1
    )

    pr_auc_change = (
        row["PR_AUC"]
        - baseline_pr_auc
    )


    print(
        f"\n{row['Experiment']}"
    )

    print(
        f"F1 change : {f1_change:+.6f}"
    )

    print(
        f"PR-AUC change : {pr_auc_change:+.6f}"
    )

    print(
        f"False Positives : "
        f"{int(row['False_Positives'])}"
    )

    print(
        f"False Negatives : "
        f"{int(row['False_Negatives'])}"
    )


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "feature_ablation_results.csv",
    index=False
)


print(
    "\nResults saved:"
)

print(
    "feature_ablation_results.csv"
)


# ============================================================
# 10. GRAPH 1 — F1 SCORE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    results_df["Experiment"],
    results_df["F1_Score"],
    marker="o"
)

plt.xlabel(
    "Feature Configuration"
)

plt.ylabel(
    "F1-Score"
)

plt.title(
    "Effect of Feature Engineering on Fraud Detection"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "feature_ablation_f1.png",
    dpi=300
)

plt.close()


print(
    "Saved:"
)

print(
    "feature_ablation_f1.png"
)


# ============================================================
# 11. GRAPH 2 — PR-AUC
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    results_df["Experiment"],
    results_df["PR_AUC"],
    marker="o"
)

plt.xlabel(
    "Feature Configuration"
)

plt.ylabel(
    "PR-AUC"
)

plt.title(
    "Effect of Feature Engineering on PR-AUC"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "feature_ablation_pr_auc.png",
    dpi=300
)

plt.close()


print(
    "Saved:"
)

print(
    "feature_ablation_pr_auc.png"
)


# ============================================================
# 12. GRAPH 3 — FORENSIC ERRORS
# ============================================================

x = np.arange(
    len(results_df)
)

width = 0.35


plt.figure(
    figsize=(10, 6)
)

plt.bar(
    x - width / 2,
    results_df["False_Positives"],
    width,
    label="False Positives"
)

plt.bar(
    x + width / 2,
    results_df["False_Negatives"],
    width,
    label="False Negatives"
)

plt.xlabel(
    "Feature Configuration"
)

plt.ylabel(
    "Number of Cases"
)

plt.title(
    "Forensic Errors Across Feature Configurations"
)

plt.xticks(
    x,
    results_df["Experiment"],
    rotation=20,
    ha="right"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "feature_ablation_errors.png",
    dpi=300
)

plt.close()


print(
    "Saved:"
)

print(
    "feature_ablation_errors.png"
)


# ============================================================
# 13. FORENSIC INTERPRETATION
# ============================================================

print("\n" + "=" * 75)
print("FORENSIC INTERPRETATION")
print("=" * 75)


print(
    """
The feature ablation study evaluates whether engineered
transaction and balance features contribute meaningful
information to fraud detection.

Four feature configurations were evaluated using the same
chronological validation period and controlled training sample.

The first configuration uses the original transaction
features and acts as the baseline.

The second configuration adds balance consistency features.

The third configuration adds ratio-based features.

The fourth configuration uses the complete engineered
feature set.

Performance is compared using precision, recall, F1-score,
PR-AUC and forensic error counts.

If the engineered feature configurations improve model
performance or reduce false-positive and false-negative
errors, this provides evidence that the feature engineering
stage contributes meaningful information to fraud detection.

The purpose of this experiment is therefore not only to
identify a high-performing model, but also to determine
whether the engineered transaction characteristics provide
additional forensic value.
"""
)


print("\n" + "=" * 75)
print("FEATURE ABLATION STUDY COMPLETED SUCCESSFULLY")
print("=" * 75)