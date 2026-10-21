import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

print("=" * 60)
print("GLOBAL SHAP FORENSIC FEATURE ANALYSIS")
print("=" * 60)

# --------------------------------------------------
# LOAD MODEL AND TEST DATA
# --------------------------------------------------

print("\nLoading trained model and test data...")

rf_model = joblib.load("fraud_rf_model.pkl")

X_test = pd.read_csv("X_test.csv")

y_test = pd.read_csv("y_test.csv").squeeze("columns")

print("Model loaded successfully.")
print("Test data loaded successfully.")

# --------------------------------------------------
# SAMPLE TEST DATA
# --------------------------------------------------

# SHAP on the full 954k test rows may take a long time.
# Use a representative sample.

sample_size = min(5000, len(X_test))

X_sample = X_test.sample(
    n=sample_size,
    random_state=42
)

print(f"\nSHAP Sample Size: {sample_size}")

# --------------------------------------------------
# CREATE SHAP EXPLAINER
# --------------------------------------------------

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(rf_model)

print("SHAP explainer created successfully.")

# --------------------------------------------------
# CALCULATE SHAP VALUES
# --------------------------------------------------

print("\nCalculating global SHAP values...")

shap_values = explainer.shap_values(X_sample)

# Handle different SHAP output formats
if isinstance(shap_values, list):
    shap_values_fraud = shap_values[1]

elif len(shap_values.shape) == 3:
    shap_values_fraud = shap_values[:, :, 1]

else:
    shap_values_fraud = shap_values

print("SHAP values calculated successfully.")

# --------------------------------------------------
# GLOBAL FEATURE IMPORTANCE
# --------------------------------------------------

print("\n" + "=" * 60)
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 60)

mean_abs_shap = np.abs(
    shap_values_fraud
).mean(axis=0)

global_importance = pd.DataFrame({
    "Feature": X_sample.columns,
    "Mean_Absolute_SHAP": mean_abs_shap
})

global_importance = global_importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

print("\nTop 15 Global Feature Contributions:")

print(
    global_importance.head(15).to_string(
        index=False
    )
)

# Save CSV
global_importance.to_csv(
    "shap_global_feature_importance.csv",
    index=False
)

print(
    "\nFeature importance saved as:"
)
print("shap_global_feature_importance.csv")

# --------------------------------------------------
# GLOBAL SHAP BAR GRAPH
# --------------------------------------------------

print("\nGenerating global SHAP feature importance graph...")

plt.figure()

shap.summary_plot(
    shap_values_fraud,
    X_sample,
    plot_type="bar",
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_global_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print("shap_global_feature_importance.png")

# --------------------------------------------------
# SHAP SUMMARY PLOT
# --------------------------------------------------

print("\nGenerating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values_fraud,
    X_sample,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_global_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print("shap_global_summary_plot.png")

# --------------------------------------------------
# FORENSIC INTERPRETATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("FORENSIC INTERPRETATION")
print("=" * 60)

top_features = global_importance.head(5)

print(
    "\nThe global SHAP analysis identifies the features that "
    "have the greatest overall influence on fraud likelihood "
    "across the sampled test transactions."
)

print("\nTop 5 Globally Important Features:")

for i, row in enumerate(
    top_features.itertuples(index=False),
    start=1
):
    print(
        f"{i}. {row.Feature} "
        f"(Mean |SHAP| = {row.Mean_Absolute_SHAP:.6f})"
    )

print(
    "\nThe bar plot shows the overall magnitude of each "
    "feature's contribution, while the SHAP summary plot shows "
    "both the magnitude and direction of feature effects."
)

print("\n" + "=" * 60)
print("GLOBAL SHAP ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)