import pandas as pd
import numpy as np

print("=" * 60)
print("PREPARING DATA FOR MODEL COMPARISON")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD ONLY REQUIRED COLUMNS
# --------------------------------------------------

print("\nLoading PaySim dataset...")

required_columns = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud"
]

df = pd.read_csv(
    "paysim.csv",
    usecols=required_columns
)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. CREATE BALANCE FEATURES
# --------------------------------------------------

print("\nCreating balance features...")

df["orig_balance_change"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

df["dest_balance_change"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)


# --------------------------------------------------
# 3. BALANCE CONSISTENCY FEATURES
# --------------------------------------------------

print("Creating balance consistency features...")

df["expected_newbalanceOrig"] = (
    df["oldbalanceOrg"] - df["amount"]
)

df["orig_balance_error"] = (
    df["newbalanceOrig"] -
    df["expected_newbalanceOrig"]
)

df["expected_newbalanceDest"] = (
    df["oldbalanceDest"] + df["amount"]
)

df["dest_balance_error"] = (
    df["newbalanceDest"] -
    df["expected_newbalanceDest"]
)

df["abs_orig_balance_error"] = (
    df["orig_balance_error"].abs()
)

df["abs_dest_balance_error"] = (
    df["dest_balance_error"].abs()
)


# --------------------------------------------------
# 4. RATIO FEATURES
# --------------------------------------------------

print("Creating ratio features...")

safe_oldbalance = (
    df["oldbalanceOrg"].replace(0, np.nan)
)

df["amount_to_orig_balance_ratio"] = (
    df["amount"] / safe_oldbalance
)

df["orig_balance_depletion_ratio"] = (
    (
        df["oldbalanceOrg"] -
        df["newbalanceOrig"]
    )
    / safe_oldbalance
)

# Replace invalid values
df["amount_to_orig_balance_ratio"] = (
    df["amount_to_orig_balance_ratio"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

df["orig_balance_depletion_ratio"] = (
    df["orig_balance_depletion_ratio"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# --------------------------------------------------
# 5. SORT CHRONOLOGICALLY
# --------------------------------------------------

print("\nSorting transactions by step...")

df = (
    df.sort_values("step")
    .reset_index(drop=True)
)

print("Chronological sorting completed.")


# --------------------------------------------------
# 6. SELECT MODEL FEATURES
# --------------------------------------------------

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


# --------------------------------------------------
# 7. ONE-HOT ENCODE TRANSACTION TYPE
# --------------------------------------------------

print("\nEncoding transaction type...")

X = pd.get_dummies(
    X,
    columns=["type"],
    drop_first=False,
    dtype=np.int8
)

print("Feature matrix shape:", X.shape)


# --------------------------------------------------
# 8. CHRONOLOGICAL SPLIT
# --------------------------------------------------

print("\nCreating chronological train/validation/test split...")

n = len(X)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end].copy()
y_train = y.iloc[:train_end].copy()

X_val = X.iloc[train_end:val_end].copy()
y_val = y.iloc[train_end:val_end].copy()

X_test = X.iloc[val_end:].copy()
y_test = y.iloc[val_end:].copy()


# --------------------------------------------------
# 9. DISPLAY SPLIT INFORMATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("CHRONOLOGICAL SPLIT RESULTS")
print("=" * 60)

print("\nTraining:")
print("Shape:", X_train.shape)
print("Fraud cases:", int(y_train.sum()))
print("Genuine cases:", int((y_train == 0).sum()))

print("\nValidation:")
print("Shape:", X_val.shape)
print("Fraud cases:", int(y_val.sum()))
print("Genuine cases:", int((y_val == 0).sum()))

print("\nTest:")
print("Shape:", X_test.shape)
print("Fraud cases:", int(y_test.sum()))
print("Genuine cases:", int((y_test == 0).sum()))


# --------------------------------------------------
# 10. SAVE TRAINING DATA
# --------------------------------------------------

print("\nSaving training data...")

X_train.to_csv(
    "X_train.csv",
    index=False
)

y_train.to_csv(
    "y_train.csv",
    index=False
)

print("X_train.csv saved.")
print("y_train.csv saved.")


# --------------------------------------------------
# 11. SAVE VALIDATION DATA
# --------------------------------------------------

print("\nSaving validation data...")

X_val.to_csv(
    "X_val.csv",
    index=False
)

y_val.to_csv(
    "y_val.csv",
    index=False
)

print("X_val.csv saved.")
print("y_val.csv saved.")


# --------------------------------------------------
# 12. VERIFY TEST DATA
# --------------------------------------------------

print("\nChecking existing test files...")

try:

    existing_X_test = pd.read_csv(
        "X_test.csv",
        nrows=5
    )

    existing_y_test = pd.read_csv(
        "y_test.csv",
        nrows=5
    )

    print("Existing X_test.csv found.")
    print("Existing y_test.csv found.")

except FileNotFoundError:

    print("Existing test files not found.")
    print("Saving newly created test files...")

    X_test.to_csv(
        "X_test.csv",
        index=False
    )

    y_test.to_csv(
        "y_test.csv",
        index=False
    )


# --------------------------------------------------
# 13. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL DATA PREPARATION COMPLETED")
print("=" * 60)

print("\nCreated / available files:")

print("1. X_train.csv")
print("2. y_train.csv")
print("3. X_val.csv")
print("4. y_val.csv")
print("5. X_test.csv")
print("6. y_test.csv")

print("\nForensic Design:")
print(
    "The data is split chronologically so that earlier transactions "
    "are used for model training and later transactions are used for "
    "validation and final testing. This prevents future transaction "
    "information from being used to train the model."
)

print("\nNext step:")
print(
    "Train and compare Random Forest, XGBoost and LightGBM "
    "using the same training and validation data."
)