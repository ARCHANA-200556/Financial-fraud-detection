import pandas as pd
import numpy as np


print("=" * 75)
print("FORENSIC EVIDENCE ANALYSIS")
print("TRANSACTION-LEVEL INVESTIGATION SUPPORT")
print("=" * 75)


# ============================================================
# 1. LOAD FORENSIC INVESTIGATION DATA
# ============================================================

print("\nLoading forensic investigation results...")

df = pd.read_csv(
    "forensic_investigation_results.csv"
)

print("Forensic data loaded successfully.")

print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# 2. LOAD CALIBRATED TEST PROBABILITIES IF AVAILABLE
# ============================================================

print(
    "\nChecking calibrated probability results..."
)

calibrated_file = "calibrated_test_results.csv"

try:

    calibrated = pd.read_csv(
        calibrated_file
    )

    print(
        "Calibrated probability file loaded."
    )

    print(
        "Calibrated dataset shape:",
        calibrated.shape
    )

except FileNotFoundError:

    calibrated = None

    print(
        "Calibrated probability file not found."
    )

    print(
        "Existing fraud probability will be retained."
    )


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [

    "step",
    "amount",

    "actual_fraud",
    "fraud_probability",

    "behavioral_risk_score",
    "behavioral_risk_level",

    "orig_amount_deviation_ratio",
    "orig_previous_transaction_count",
    "orig_step_gap",
    "orig_unique_destinations",

    "forensic_priority_score",
    "forensic_priority_level"

]


missing_columns = [

    column
    for column in required_columns
    if column not in df.columns

]


if missing_columns:

    print(
        "\nMissing required columns:"
    )

    print(
        missing_columns
    )

    raise SystemExit


# ============================================================
# 4. FINAL DECISION THRESHOLD
# ============================================================

FINAL_THRESHOLD = 0.75


print(
    "\nFinal investigation threshold:",
    FINAL_THRESHOLD
)


# ============================================================
# 5. FRAUD LIKELIHOOD CATEGORY
# ============================================================

print(
    "\nCreating fraud likelihood categories..."
)


def probability_category(probability):

    if probability >= 0.90:

        return "VERY HIGH"

    elif probability >= 0.75:

        return "HIGH"

    elif probability >= 0.50:

        return "MEDIUM"

    elif probability >= 0.25:

        return "LOW"

    else:

        return "VERY LOW"


df[
    "fraud_likelihood_category"
] = (

    df[
        "fraud_probability"
    ]
    .apply(
        probability_category
    )

)


# ============================================================
# 6. BEHAVIORAL EVIDENCE
# ============================================================

print(
    "\nCreating behavioral evidence classification..."
)


def behavioral_evidence(row):

    score = row[
        "behavioral_risk_score"
    ]

    level = row[
        "behavioral_risk_level"
    ]


    if level == "HIGH":

        return "STRONG BEHAVIORAL EVIDENCE"

    elif score >= 30:

        return "MODERATE BEHAVIORAL EVIDENCE"

    else:

        return "LIMITED BEHAVIORAL EVIDENCE"


df[
    "behavioral_evidence"
] = (

    df.apply(
        behavioral_evidence,
        axis=1
    )

)


# ============================================================
# 7. TRANSACTION AMOUNT EVIDENCE
# ============================================================

print(
    "\nAnalyzing transaction amount evidence..."
)


amount_75 = df[
    "amount"
].quantile(0.75)


amount_90 = df[
    "amount"
].quantile(0.90)


amount_95 = df[
    "amount"
].quantile(0.95)


print(
    f"75th percentile amount : {amount_75:,.2f}"
)

print(
    f"90th percentile amount : {amount_90:,.2f}"
)

print(
    f"95th percentile amount : {amount_95:,.2f}"
)


def amount_evidence(amount):

    if amount >= amount_95:

        return "EXTREMELY HIGH AMOUNT"

    elif amount >= amount_90:

        return "VERY HIGH AMOUNT"

    elif amount >= amount_75:

        return "HIGH AMOUNT"

    else:

        return "NORMAL RANGE"


df[
    "amount_evidence"
] = (

    df[
        "amount"
    ]
    .apply(
        amount_evidence
    )

)


# ============================================================
# 8. BEHAVIORAL ANOMALIES
# ============================================================

print(
    "\nIdentifying behavioral anomalies..."
)


def identify_anomalies(row):

    anomalies = []


    # --------------------------------------------------------
    # Unusual transaction amount
    # --------------------------------------------------------

    if row[
        "orig_amount_deviation_ratio"
    ] >= 5:

        anomalies.append(
            "Unusual transaction amount"
        )


    # --------------------------------------------------------
    # New / limited sender history
    # --------------------------------------------------------

    if row[
        "orig_previous_transaction_count"
    ] < 2:

        anomalies.append(
            "Limited sender history"
        )


    # --------------------------------------------------------
    # Rapid transaction activity
    # --------------------------------------------------------

    step_gap = row[
        "orig_step_gap"
    ]

    if (

        step_gap >= 0
        and
        step_gap <= 1

    ):

        anomalies.append(
            "Rapid transaction activity"
        )


    # --------------------------------------------------------
    # Destination diversity
    # --------------------------------------------------------

    if row[
        "orig_unique_destinations"
    ] >= 5:

        anomalies.append(
            "Multiple destinations"
        )


    if len(anomalies) == 0:

        return "No strong behavioral anomaly detected"


    return "; ".join(
        anomalies
    )


df[
    "behavioral_anomalies"
] = (

    df.apply(
        identify_anomalies,
        axis=1
    )

)


# ============================================================
# 9. INDEPENDENT FORENSIC EVIDENCE SCORE
# ============================================================

print(
    "\nCalculating forensic evidence score..."
)


# Model evidence
#
# Maximum contribution = 50

probability_component = (

    df[
        "fraud_probability"
    ]
    .clip(
        0,
        1
    )
    * 50

)


# Behavioral evidence
#
# Maximum contribution = 30

behavior_component = (

    df[
        "behavioral_risk_score"
    ]
    .clip(
        0,
        100
    )
    * 0.30

)


# Financial evidence
#
# Maximum contribution = 20

amount_component = np.where(

    df[
        "amount"
    ] >= amount_95,

    20,

    np.where(

        df[
            "amount"
        ] >= amount_90,

        15,

        np.where(

            df[
                "amount"
            ] >= amount_75,

            10,

            0

        )

    )

)


df[
    "forensic_evidence_score"
] = (

    probability_component
    +
    behavior_component
    +
    amount_component

)


df[
    "forensic_evidence_score"
] = (

    df[
        "forensic_evidence_score"
    ]
    .clip(
        0,
        100
    )

)


# ============================================================
# 10. EVIDENCE STRENGTH
# ============================================================

print(
    "\nClassifying evidence strength..."
)


def evidence_strength(score):

    if score >= 80:

        return "VERY STRONG"

    elif score >= 60:

        return "STRONG"

    elif score >= 40:

        return "MODERATE"

    elif score >= 20:

        return "WEAK"

    else:

        return "MINIMAL"


df[
    "evidence_strength"
] = (

    df[
        "forensic_evidence_score"
    ]
    .apply(
        evidence_strength
    )

)


# ============================================================
# 11. INVESTIGATION DECISION
# ============================================================

print(
    "\nGenerating investigation decisions..."
)


def investigation_decision(row):

    probability = row[
        "fraud_probability"
    ]

    score = row[
        "forensic_evidence_score"
    ]

    behavioral = row[
        "behavioral_risk_level"
    ]


    if (

        probability >= 0.90
        and
        score >= 70

    ):

        return "IMMEDIATE INVESTIGATION"


    elif (

        probability >= FINAL_THRESHOLD
        and
        behavioral == "HIGH"

    ):

        return "PRIORITY INVESTIGATION"


    elif probability >= FINAL_THRESHOLD:

        return "INVESTIGATE"


    elif probability >= 0.50:

        return "ENHANCED REVIEW"


    else:

        return "ROUTINE MONITORING"


df[
    "investigation_decision"
] = (

    df.apply(
        investigation_decision,
        axis=1
    )

)


# ============================================================
# 12. RECOMMENDED INVESTIGATOR ACTION
# ============================================================

print(
    "\nGenerating recommended investigator actions..."
)


def recommended_action(decision):

    if decision == "IMMEDIATE INVESTIGATION":

        return (
            "Immediately review transaction, "
            "sender history and destination activity."
        )


    elif decision == "PRIORITY INVESTIGATION":

        return (
            "Prioritize investigator review and "
            "examine supporting behavioral evidence."
        )


    elif decision == "INVESTIGATE":

        return (
            "Review transaction details and "
            "supporting account evidence."
        )


    elif decision == "ENHANCED REVIEW":

        return (
            "Perform additional transaction and "
            "account-level review."
        )


    else:

        return (
            "Retain for routine monitoring; "
            "no immediate investigation required."
        )


df[
    "recommended_action"
] = (

    df[
        "investigation_decision"
    ]
    .apply(
        recommended_action
    )

)


# ============================================================
# 13. FORENSIC EVIDENCE SUMMARY
# ============================================================

print(
    "\nGenerating forensic evidence summaries..."
)


def evidence_summary(row):

    return (

        "Fraud likelihood = "
        +
        f"{row['fraud_probability']:.4f}"
        +
        " | Behavioral risk = "
        +
        str(
            row[
                "behavioral_risk_level"
            ]
        )
        +
        " | Amount evidence = "
        +
        str(
            row[
                "amount_evidence"
            ]
        )
        +
        " | Behavioral anomalies = "
        +
        str(
            row[
                "behavioral_anomalies"
            ]
        )

    )


df[
    "forensic_evidence_summary"
] = (

    df.apply(
        evidence_summary,
        axis=1
    )

)


# ============================================================
# 14. FINAL CASE CLASSIFICATION
# ============================================================

print(
    "\nCreating final forensic case classification..."
)


def final_case_classification(row):

    actual = row[
        "actual_fraud"
    ]

    decision = row[
        "investigation_decision"
    ]


    if actual == 1:

        if decision in [

            "IMMEDIATE INVESTIGATION",
            "PRIORITY INVESTIGATION",
            "INVESTIGATE"

        ]:

            return "CONFIRMED FRAUD — CORRECTLY PRIORITIZED"

        else:

            return "CONFIRMED FRAUD — REVIEW REQUIRED"


    else:

        if decision in [

            "IMMEDIATE INVESTIGATION",
            "PRIORITY INVESTIGATION",
            "INVESTIGATE"

        ]:

            return "GENUINE — INVESTIGATION ALERT"

        else:

            return "GENUINE — NOT PRIORITIZED"


df[
    "final_case_classification"
] = (

    df.apply(
        final_case_classification,
        axis=1
    )

)


# ============================================================
# 15. SUMMARY
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "FORENSIC EVIDENCE SUMMARY"
)

print(
    "=" * 75
)


print(
    "\nTotal transactions:",
    f"{len(df):,}"
)


print(
    "\nConfirmed fraud:",
    f"{df['actual_fraud'].sum():,}"
)


print(
    "\nInvestigation decision distribution:"
)


print(
    df[
        "investigation_decision"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nEvidence strength distribution:"
)


print(
    df[
        "evidence_strength"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# 16. FRAUD RATE BY INVESTIGATION DECISION
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "FRAUD RATE BY INVESTIGATION DECISION"
)

print(
    "=" * 75
)


decision_summary = (

    df.groupby(
        "investigation_decision"
    )[
        "actual_fraud"
    ]
    .agg(
        total_cases="count",
        confirmed_fraud="sum",
        fraud_rate="mean"
    )

)


decision_summary[
    "fraud_rate"
] *= 100


print(
    decision_summary
    .sort_values(
        "fraud_rate",
        ascending=False
    )
    .round(4)
    .to_string()
)


# ============================================================
# 17. FINAL FORENSIC CASES
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "TOP FORENSIC INVESTIGATION CASES"
)

print(
    "=" * 75
)


top_cases = (

    df.sort_values(
        [
            "forensic_evidence_score",
            "fraud_probability"
        ],
        ascending=False
    )
    .head(20)

)


top_columns = [

    "step",

    "amount",

    "actual_fraud",

    "fraud_probability",

    "fraud_likelihood_category",

    "behavioral_risk_score",

    "behavioral_risk_level",

    "orig_amount_deviation_ratio",

    "orig_previous_transaction_count",

    "orig_step_gap",

    "orig_unique_destinations",

    "amount_evidence",

    "behavioral_anomalies",

    "forensic_evidence_score",

    "evidence_strength",

    "investigation_decision",

    "recommended_action"

]


print(
    top_cases[
        top_columns
    ]
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 18. CONFIRMED FRAUD ANALYSIS
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "CONFIRMED FRAUD ANALYSIS"
)

print(
    "=" * 75
)


fraud_cases = df[
    df[
        "actual_fraud"
    ] == 1
]


print(
    "\nConfirmed fraud cases:",
    len(
        fraud_cases
    )
)


print(
    "Average fraud likelihood:",
    f"{fraud_cases['fraud_probability'].mean():.6f}"
)


print(
    "Average forensic evidence score:",
    f"{fraud_cases['forensic_evidence_score'].mean():.4f}"
)


print(
    "\nFraud cases by behavioral risk:"
)


print(
    fraud_cases[
        "behavioral_risk_level"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# 19. MISSED FRAUD ANALYSIS
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "MISSED FRAUD ANALYSIS"
)

print(
    "=" * 75
)


missed_fraud = df[
    (
        df[
            "actual_fraud"
        ] == 1
    )
    &
    (
        df[
            "fraud_probability"
        ] < FINAL_THRESHOLD
    )
]


print(
    "\nMissed fraud cases:",
    len(
        missed_fraud
    )
)


if len(missed_fraud) > 0:

    print(
        "\nMissed fraud details:"
    )

    print(
        missed_fraud[
            [
                "amount",
                "fraud_probability",
                "behavioral_risk_score",
                "behavioral_risk_level",
                "forensic_evidence_score"
            ]
        ]
        .round(6)
        .to_string(
            index=False
        )
    )

else:

    print(
        "\nNo missed fraud cases found."
    )


# ============================================================
# 20. SAVE COMPLETE FORENSIC DATA
# ============================================================

print(
    "\nSaving complete forensic evidence dataset..."
)


output_columns = [

    "step",

    "amount",

    "actual_fraud",

    "fraud_probability",

    "fraud_likelihood_category",

    "behavioral_risk_score",

    "behavioral_risk_level",

    "behavioral_evidence",

    "orig_amount_deviation_ratio",

    "orig_previous_transaction_count",

    "orig_step_gap",

    "orig_unique_destinations",

    "amount_evidence",

    "behavioral_anomalies",

    "forensic_evidence_score",

    "evidence_strength",

    "forensic_priority_score",

    "forensic_priority_level",

    "investigation_decision",

    "recommended_action",

    "forensic_evidence_summary",

    "final_case_classification"

]


df[
    output_columns
].to_csv(
    "forensic_evidence_analysis.csv",
    index=False
)


# ============================================================
# 21. SAVE SUMMARY
# ============================================================

decision_summary.reset_index().to_csv(
    "forensic_decision_summary.csv",
    index=False
)


print(
    "\nFiles saved successfully:"
)

print(
    "1. forensic_evidence_analysis.csv"
)

print(
    "2. forensic_decision_summary.csv"
)


# ============================================================
# 22. FINAL INTERPRETATION
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "FORENSIC INTERPRETATION"
)

print(
    "=" * 75
)


print(
    """
The forensic evidence analysis extends the machine-learning
fraud prediction into a transaction-level investigation-support
framework.

The fraud likelihood represents the model's assessment of the
transaction's likelihood of being fraudulent.

Behavioral evidence provides additional context based on
historical transaction activity, transaction amount deviation,
sender history, transaction timing and destination diversity.

Financial evidence provides additional context based on the
relative transaction amount within the test population.

These evidence components are combined into a forensic evidence
score and translated into an investigation decision.

The purpose of the evidence score is prioritization rather than
automatic determination of guilt.

A high model likelihood combined with strong behavioral and
financial evidence produces a stronger case for immediate
investigation.

Conversely, a transaction with lower likelihood or limited
supporting evidence may be assigned a lower investigation
priority.

This allows the system to move beyond simple classification
and provide an interpretable forensic decision-support layer.

The final investigation recommendation should be treated as
analytical support for a human investigator rather than as
conclusive proof of fraudulent activity.
"""
)


print(
    "\n" + "=" * 75
)

print(
    "FORENSIC EVIDENCE ANALYSIS COMPLETED SUCCESSFULLY"
)

print(
    "=" * 75
)