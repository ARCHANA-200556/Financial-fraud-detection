import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Fraud Forensic Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE CHECK
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))


def file_path(filename):
    return os.path.join(BASE, filename)


def exists(filename):
    return os.path.exists(file_path(filename))


# ============================================================
# SMALL FILE LOADER
# ============================================================

@st.cache_data
def load_small_csv(filename):

    path = file_path(filename)

    if not os.path.exists(path):
        return None

    return pd.read_csv(path)


# ============================================================
# LARGE FILE LOADER
# IMPORTANT:
# DO NOT CACHE THE 954K ROW DATAFRAME
# ============================================================

def load_forensic_columns(columns=None, nrows=None):

    path = file_path(
        "forensic_investigation_results.csv"
    )

    if not os.path.exists(path):
        return None

    return pd.read_csv(
        path,
        usecols=columns,
        nrows=nrows
    )


def load_evidence_columns(columns=None, nrows=None):

    path = file_path(
        "forensic_evidence_analysis.csv"
    )

    if not os.path.exists(path):
        return None

    return pd.read_csv(
        path,
        usecols=columns,
        nrows=nrows
    )


# ============================================================
# LOAD ONLY SMALL FILES AT STARTUP
# ============================================================

model_comparison = load_small_csv(
    "model_comparison_results.csv"
)

feature_ablation = load_small_csv(
    "feature_ablation_results.csv"
)

temporal = load_small_csv(
    "temporal_fraud_analysis.csv"
)

temporal_type = load_small_csv(
    "temporal_transaction_type_analysis.csv"
)

shap = load_small_csv(
    "shap_global_feature_importance.csv"
)

threshold = load_small_csv(
    "threshold_cost_analysis.csv"
)

decision_summary = load_small_csv(
    "forensic_decision_summary.csv"
)

calibrated_results = load_small_csv(
    "calibrated_test_results.csv"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🔎 Forensic Intelligence"
)

st.sidebar.markdown(
    """
    ### Financial Fraud Detection
    """
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Executive Dashboard",
        "📝 Manual Transaction Prediction",
        "🤖 Model Comparison",
        "🧬 Behavioral Profiling",
        "🎯 Probability Calibration",
        "⚖️ Threshold & Cost Analysis",
        "📈 Temporal & Concept Drift",
        "🧪 Feature Ablation",
        "🧠 Global SHAP Analysis",
        "🔬 Individual Case Investigation",
        "🚨 Forensic Decision Support",
        "📁 Forensic Case Explorer",
        "📄 Investigation Report"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    """
    This system provides analytical decision support.

    A high risk score does not constitute
    conclusive proof of fraud.
    """
)


# ============================================================
# PAGE 1
# EXECUTIVE DASHBOARD
# ============================================================

if page == "🏠 Executive Dashboard":

    st.title(
        "🔎 Financial Fraud Detection & Forensic Intelligence"
    )

    st.markdown(
        """
        ## Integrated Forensic Analytics System

        This project goes beyond simple fraud classification.

        It combines:

        **Machine Learning → Probability Calibration →
        Behavioral Profiling → Threshold Analysis →
        Temporal Analysis → SHAP Explainability →
        Forensic Evidence → Investigation Priority**
        """
    )

    st.divider()

    # --------------------------------------------------------
    # MAIN TEST DATA
    # --------------------------------------------------------

    dashboard_columns = [
        "actual_fraud",
        "fraud_probability",
        "behavioral_risk_level",
        "forensic_priority_level",
        "forensic_priority_score",
        "amount"
    ]

    with st.spinner(
        "Loading dashboard statistics..."
    ):

        data = load_forensic_columns(
            dashboard_columns
        )

    if data is None:

        st.error(
            "forensic_investigation_results.csv not found."
        )

        st.stop()

    total_transactions = len(data)

    total_fraud = int(
        data["actual_fraud"].sum()
    )

    fraud_rate = (
        total_fraud /
        total_transactions
    ) * 100

    critical = int(
        (
            data[
                "forensic_priority_level"
            ]
            == "CRITICAL"
        ).sum()
    )

    high = int(
        (
            data[
                "forensic_priority_level"
            ]
            == "HIGH"
        ).sum()
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Test Transactions",
        f"{total_transactions:,}"
    )

    c2.metric(
        "Confirmed Fraud",
        f"{total_fraud:,}"
    )

    c3.metric(
        "Fraud Rate",
        f"{fraud_rate:.4f}%"
    )

    c4.metric(
        "Critical Cases",
        f"{critical:,}"
    )

    c5.metric(
        "High Cases",
        f"{high:,}"
    )

    st.divider()

    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    st.subheader(
        "Final Model Configuration"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Final Model",
        "Random Forest"
    )

    c2.metric(
        "Decision Threshold",
        "0.75"
    )

    c3.metric(
        "Framework",
        "Forensic Decision Support"
    )

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "Final Test Performance"
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Accuracy",
        "99.9998%"
    )

    c2.metric(
        "Precision",
        "100.00%"
    )

    c3.metric(
        "Recall",
        "99.9501%"
    )

    c4.metric(
        "F1-Score",
        "99.9750%"
    )

    c5.metric(
        "ROC-AUC",
        "1.0000"
    )

    c6.metric(
        "PR-AUC",
        "0.999999"
    )

    st.divider()

    # --------------------------------------------------------
    # PRIORITY DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Forensic Priority Distribution"
    )

    priority_counts = (
        data[
            "forensic_priority_level"
        ]
        .value_counts()
    )

    st.bar_chart(
        priority_counts
    )

    st.dataframe(
        priority_counts.rename(
            "Number of Transactions"
        ),
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # TOP CASES
    # --------------------------------------------------------

    st.subheader(
        "Top Forensic Investigation Cases"
    )

    top_cases = (
        data
        .sort_values(
            "forensic_priority_score",
            ascending=False
        )
        .head(15)
    )

    st.dataframe(
        top_cases,
        use_container_width=True
    )


# ============================================================
# PAGE 1B
# MANUAL TRANSACTION PREDICTION
# ============================================================

elif page == "📝 Manual Transaction Prediction":

    st.title(
        "📝 Manual Transaction Prediction"
    )

    st.markdown(
        """
        Enter the transaction details below to estimate
        the fraud probability using the trained Random Forest model.
        """
    )

    model_path = file_path(
        "fraud_rf_model.pkl"
    )

    if not os.path.exists(model_path):

        st.error(
            "fraud_rf_model.pkl not found in the application folder."
        )

        st.stop()

    try:

        model = joblib.load(
            model_path
        )

    except Exception as e:

        st.error(
            f"Unable to load the fraud model: {e}"
        )

        st.stop()

    if hasattr(
        model,
        "feature_names_in_"
    ):

        feature_order = list(
            model.feature_names_in_
        )

    elif hasattr(
        model,
        "feature_names"
    ):

        feature_order = list(
            model.feature_names
        )

    else:

        st.error(
            "The model does not contain feature names. "
            "The training feature order is required."
        )

        st.stop()

    st.subheader(
        "Transaction Features"
    )

    defaults = {
        "amount": 1000.0,
        "oldbalanceorg": 1000.0,
        "newbalanceorig": 0.0,
        "oldbalancedest": 0.0,
        "newbalancedest": 1000.0
    }

    with st.form(
        "manual_prediction_form"
    ):

        values = {}

        columns = st.columns(2)

        for i, feature in enumerate(
            feature_order
        ):

            feature_name = str(
                feature
            )

            default_value = defaults.get(
                feature_name.lower(),
                0.0
            )

            with columns[i % 2]:

                values[feature] = st.number_input(
                    feature_name,
                    value=float(
                        default_value
                    ),
                    format="%.6f",
                    key=f"manual_feature_{i}"
                )

        submitted = st.form_submit_button(
            "🔍 Predict Transaction"
        )

    if submitted:

        input_df = pd.DataFrame(
            [values]
        )

        input_df = input_df.reindex(
            columns=feature_order,
            fill_value=0
        )

        try:

            if hasattr(
                model,
                "predict_proba"
            ):

                probability = float(
                    model.predict_proba(
                        input_df
                    )[0][1]
                )

            else:

                prediction_value = int(
                    model.predict(
                        input_df
                    )[0]
                )

                probability = float(
                    prediction_value
                )

            threshold_value = 0.75

            prediction = int(
                probability >= threshold_value
            )

            st.divider()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Fraud Probability",
                f"{probability:.2%}"
            )

            c2.metric(
                "Decision Threshold",
                f"{threshold_value:.2f}"
            )

            c3.metric(
                "Prediction",
                "Fraud" if prediction == 1 else "Genuine"
            )

            if prediction == 1:

                st.error(
                    f"""
                    HIGH FRAUD LIKELIHOOD

                    Fraud probability: {probability:.2%}

                    This transaction should be reviewed by
                    an investigator with supporting evidence.
                    """
                )

            else:

                st.success(
                    f"""
                    LOW FRAUD LIKELIHOOD

                    Fraud probability: {probability:.2%}

                    Continue reviewing the available
                    transaction evidence.
                    """
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# PAGE 2
# MODEL COMPARISON
# ============================================================

elif page == "🤖 Model Comparison":

    st.title(
        "🤖 Machine Learning Model Comparison"
    )

    st.markdown(
        """
        Five algorithms were evaluated using the same
        chronological validation framework.

        **Random Forest, XGBoost, LightGBM, CatBoost
        and Extra Trees**
        """
    )

    if model_comparison is None:

        st.error(
            "model_comparison_results.csv not found."
        )

        st.stop()

    st.subheader(
        "Complete Comparison Table"
    )

    st.dataframe(
        model_comparison,
        use_container_width=True
    )

    st.divider()

    # F1
    st.subheader(
        "F1-Score Comparison"
    )

    if {
        "Model",
        "F1_Score"
    }.issubset(
        model_comparison.columns
    ):

        chart = (
            model_comparison
            .set_index("Model")
            ["F1_Score"]
        )

        st.bar_chart(
            chart
        )

    # Precision
    st.subheader(
        "Precision Comparison"
    )

    if {
        "Model",
        "Precision"
    }.issubset(
        model_comparison.columns
    ):

        chart = (
            model_comparison
            .set_index("Model")
            ["Precision"]
        )

        st.bar_chart(
            chart
        )

    # Recall
    st.subheader(
        "Recall Comparison"
    )

    if {
        "Model",
        "Recall"
    }.issubset(
        model_comparison.columns
    ):

        chart = (
            model_comparison
            .set_index("Model")
            ["Recall"]
        )

        st.bar_chart(
            chart
        )

    # PR AUC
    st.subheader(
        "PR-AUC Comparison"
    )

    if {
        "Model",
        "PR_AUC"
    }.issubset(
        model_comparison.columns
    ):

        chart = (
            model_comparison
            .set_index("Model")
            ["PR_AUC"]
        )

        st.bar_chart(
            chart
        )

    # Errors
    st.subheader(
        "False Positive Comparison"
    )

    if {
        "Model",
        "False_Positives"
    }.issubset(
        model_comparison.columns
    ):

        chart = (
            model_comparison
            .set_index("Model")
            ["False_Positives"]
        )

        st.bar_chart(
            chart
        )

    st.subheader(
        "False Negative Comparison"
    )

    if {
        "Model",
        "False_Negatives"
    }.issubset(
        model_comparison.columns
    ):

        chart = (
            model_comparison
            .set_index("Model")
            ["False_Negatives"]
        )

        st.bar_chart(
            chart
        )

    st.divider()

    st.success(
        """
        FINAL MODEL: RANDOM FOREST

        Final test results:

        Precision = 100%

        Recall = 99.9501%

        F1-Score = 99.9750%

        False Positives = 0

        False Negatives = 2
        """
    )


# ============================================================
# PAGE 3
# BEHAVIORAL PROFILING
# ============================================================

elif page == "🧬 Behavioral Profiling":

    st.title(
        "🧬 Behavioral Risk Profiling"
    )

    st.markdown(
        """
        ### Why Behavioral Profiling?

        A fraud detector should not rely only on the current
        transaction.

        Behavioral profiling provides contextual evidence from
        transaction history.

        The system considers:

        - Previous transaction count
        - Amount deviation
        - Transaction timing
        - Destination diversity
        - Sender transaction behavior
        """
    )

    behavior_columns = [
        "actual_fraud",
        "behavioral_risk_score",
        "behavioral_risk_level",
        "orig_amount_deviation_ratio",
        "orig_previous_transaction_count",
        "orig_step_gap",
        "orig_unique_destinations",
        "amount"
    ]

    with st.spinner(
        "Loading behavioral profiles..."
    ):

        behavior = load_forensic_columns(
            behavior_columns
        )

    if behavior is None:

        st.error(
            "Behavioral data unavailable."
        )

        st.stop()

    st.subheader(
        "Behavioral Risk Distribution"
    )

    risk_counts = (
        behavior[
            "behavioral_risk_level"
        ]
        .value_counts()
    )

    st.bar_chart(
        risk_counts
    )

    st.dataframe(
        risk_counts.rename(
            "Transactions"
        ),
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "Fraud Rate by Behavioral Risk"
    )

    risk_table = (
        behavior
        .groupby(
            "behavioral_risk_level"
        )["actual_fraud"]
        .agg(
            total_transactions="count",
            fraud_transactions="sum",
            fraud_rate="mean"
        )
    )

    risk_table[
        "fraud_rate"
    ] *= 100

    st.dataframe(
        risk_table.round(4),
        use_container_width=True
    )

    st.bar_chart(
        risk_table[
            "fraud_rate"
        ]
    )

    st.divider()

    st.subheader(
        "Behavioral Feature Statistics"
    )

    behavioral_features = [
        "behavioral_risk_score",
        "orig_amount_deviation_ratio",
        "orig_previous_transaction_count",
        "orig_step_gap",
        "orig_unique_destinations"
    ]

    st.dataframe(
        behavior[
            behavioral_features
        ]
        .describe()
        .T,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "High Behavioral Risk Cases"
    )

    high_risk = behavior[
        behavior[
            "behavioral_risk_level"
        ] == "HIGH"
    ]

    high_risk = (
        high_risk
        .sort_values(
            "behavioral_risk_score",
            ascending=False
        )
        .head(50)
    )

    st.dataframe(
        high_risk,
        use_container_width=True,
        height=500
    )

    st.info(
        """
        Behavioral risk is supporting evidence.

        It does not independently establish fraud.
        """
    )


# ============================================================
# PAGE 4
# PROBABILITY CALIBRATION
# ============================================================

elif page == "🎯 Probability Calibration":

    st.title(
        "🎯 Fraud Probability Calibration"
    )

    st.markdown(
        """
        Random Forest produces fraud likelihood scores.

        A calibration stage was added so that these scores can
        be interpreted more defensibly as probabilities.
        """
    )

    st.subheader(
        "Calibration Curve"
    )

    show_image_path = file_path(
        "probability_calibration_curve.png"
    )

    if os.path.exists(
        show_image_path
    ):

        st.image(
            show_image_path,
            caption="Probability Calibration Curve",
            use_container_width=True
        )

    else:

        st.warning(
            "Calibration curve not found."
        )

    st.divider()

    calibration_table = pd.DataFrame({
        "Dataset": [
            "Validation",
            "Validation",
            "Test",
            "Test"
        ],
        "Stage": [
            "Before",
            "After",
            "Before",
            "After"
        ],
        "Brier Score": [
            0.00000796,
            0.00000043,
            0.00000680,
            0.00000219
        ]
    })

    st.subheader(
        "Brier Score Comparison"
    )

    st.dataframe(
        calibration_table,
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Mean Fraud Probability",
        "0.996677"
    )

    c2.metric(
        "Mean Genuine Probability",
        "0.000001"
    )


# ============================================================
# PAGE 5
# THRESHOLD
# ============================================================

elif page == "⚖️ Threshold & Cost Analysis":

    st.title(
        "⚖️ Forensic Threshold & Cost Analysis"
    )

    st.markdown(
        """
        The fraud threshold was selected using the validation
        period.

        The cost of missing fraud was treated as greater than
        the cost of generating an investigation alert.
        """
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Selected Threshold",
        "0.75"
    )

    c2.metric(
        "False Negative Cost",
        "100"
    )

    c3.metric(
        "False Positive Cost",
        "1"
    )

    st.divider()

    for image, caption in [
        (
            "threshold_performance.png",
            "Threshold Performance"
        ),
        (
            "threshold_error_analysis.png",
            "Threshold Error Analysis"
        ),
        (
            "threshold_forensic_cost.png",
            "Forensic Cost Analysis"
        )
    ]:

        path = file_path(
            image
        )

        if os.path.exists(
            path
        ):

            st.subheader(
                caption
            )

            st.image(
                path,
                caption=caption,
                use_container_width=True
            )

    if threshold is not None:

        st.subheader(
            "Threshold Analysis Table"
        )

        st.dataframe(
            threshold,
            use_container_width=True
        )


# ============================================================
# PAGE 6
# TEMPORAL ANALYSIS
# ============================================================

elif page == "📈 Temporal & Concept Drift":

    st.title(
        "📈 Temporal Fraud Behavior & Concept Drift"
    )

    st.markdown(
        """
        Fraud behavior can change over time.

        The project therefore analyzes:

        - Fraud prevalence
        - Fraud transaction count
        - Transaction amount
        - Transaction type
        """
    )

    images = [
        (
            "temporal_fraud_rate.png",
            "Temporal Fraud Rate"
        ),
        (
            "temporal_fraud_count.png",
            "Fraud Transaction Count"
        ),
        (
            "temporal_transaction_amount.png",
            "Transaction Amount Over Time"
        )
    ]

    for image, caption in images:

        path = file_path(
            image
        )

        if os.path.exists(
            path
        ):

            st.subheader(
                caption
            )

            st.image(
                path,
                caption=caption,
                use_container_width=True
            )

    if temporal is not None:

        st.subheader(
            "Temporal Fraud Statistics"
        )

        st.dataframe(
            temporal,
            use_container_width=True
        )

    if temporal_type is not None:

        st.subheader(
            "Fraud by Transaction Type"
        )

        st.dataframe(
            temporal_type,
            use_container_width=True
        )

    st.warning(
        """
        The increase in fraud prevalence over later periods
        suggests that the transaction environment may change.

        This indicates the importance of continued model
        monitoring.
        """
    )


# ============================================================
# PAGE 7
# FEATURE ABLATION
# ============================================================

elif page == "🧪 Feature Ablation":

    st.title(
        "🧪 Feature Ablation Study"
    )

    st.markdown(
        """
        Feature ablation tests whether engineered features
        actually contribute to fraud detection.
        """
    )

    images = [
        (
            "feature_ablation_f1.png",
            "F1-Score Comparison"
        ),
        (
            "feature_ablation_pr_auc.png",
            "PR-AUC Comparison"
        ),
        (
            "feature_ablation_errors.png",
            "Forensic Error Comparison"
        )
    ]

    for image, caption in images:

        path = file_path(
            image
        )

        if os.path.exists(
            path
        ):

            st.subheader(
                caption
            )

            st.image(
                path,
                caption=caption,
                use_container_width=True
            )

    if feature_ablation is not None:

        st.subheader(
            "Feature Ablation Results"
        )

        st.dataframe(
            feature_ablation,
            use_container_width=True
        )

    st.info(
        """
        Important result:

        Original features:

        F1 = 0.787004

        PR-AUC = 0.870983

        After adding balance features:

        F1 = 0.999111

        PR-AUC = 1.000000

        False negatives decreased from 126 to 0.

        This provides experimental evidence that feature
        engineering contributed significant predictive value.
        """
    )


# ============================================================
# PAGE 8
# SHAP
# ============================================================

elif page == "🧠 Global SHAP Analysis":

    st.title(
        "🧠 Global SHAP Explainable AI"
    )

    st.markdown(
        """
        SHAP explains which features contribute most strongly
        to the model's predictions.
        """
    )

    for image, caption in [
        (
            "shap_global_feature_importance.png",
            "Global SHAP Feature Importance"
        ),
        (
            "shap_global_summary_plot.png",
            "Global SHAP Summary Plot"
        )
    ]:

        path = file_path(
            image
        )

        if os.path.exists(
            path
        ):

            st.subheader(
                caption
            )

            st.image(
                path,
                caption=caption,
                use_container_width=True
            )

    if shap is not None:

        st.subheader(
            "SHAP Feature Importance Table"
        )

        st.dataframe(
            shap,
            use_container_width=True
        )


# ============================================================
# PAGE 9
# INDIVIDUAL CASE
# ============================================================

elif page == "🔬 Individual Case Investigation":

    st.title(
        "🔬 Individual Forensic Case Investigation"
    )

    case_columns = [
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

    with st.spinner(
        "Loading investigation cases..."
    ):

        cases = load_forensic_columns(
            case_columns
        )

    if cases is None:

        st.error(
            "Investigation data unavailable."
        )

        st.stop()

    case_type = st.selectbox(
        "Select Case Type",
        [
            "Critical Fraud Case",
            "High Priority Case",
            "Missed Fraud Case",
            "Genuine Transaction"
        ]
    )

    if case_type == "Critical Fraud Case":

        filtered = cases[
            cases[
                "forensic_priority_level"
            ] == "CRITICAL"
        ]

    elif case_type == "High Priority Case":

        filtered = cases[
            cases[
                "forensic_priority_level"
            ].isin(
                [
                    "CRITICAL",
                    "HIGH"
                ]
            )
        ]

    elif case_type == "Missed Fraud Case":

        filtered = cases[
            (
                cases[
                    "actual_fraud"
                ] == 1
            )
            &
            (
                cases[
                    "fraud_probability"
                ] < 0.75
            )
        ]

    else:

        filtered = cases[
            cases[
                "actual_fraud"
            ] == 0
        ]

    if len(filtered) == 0:

        st.warning(
            "No cases found."
        )

    else:

        selected = st.selectbox(
            "Select Transaction",
            filtered.index.tolist()
        )

        case = filtered.loc[
            selected
        ]

        st.divider()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Fraud Probability",
            f"{case['fraud_probability']:.4%}"
        )

        c2.metric(
            "Behavioral Risk",
            str(
                case[
                    "behavioral_risk_level"
                ]
            )
        )

        c3.metric(
            "Priority Score",
            f"{case['forensic_priority_score']:.4f}"
        )

        st.subheader(
            "Transaction Evidence"
        )

        st.dataframe(
            case.to_frame(
                "Value"
            ),
            use_container_width=True
        )

        probability = float(
            case[
                "fraud_probability"
            ]
        )

        if probability >= 0.75:

            st.error(
                f"""
                HIGH FRAUD LIKELIHOOD

                Model likelihood:
                {probability:.2%}

                This transaction should be reviewed by
                an investigator together with supporting
                behavioral and financial evidence.
                """
            )

        else:

            st.info(
                f"""
                The transaction has a model fraud likelihood
                of {probability:.2%}.

                Additional evidence should be considered
                before assigning investigation priority.
                """
            )

        st.subheader(
            "Individual SHAP Explanations"
        )

        for image, caption in [
            (
                "shap_case1_detected_fraud.png",
                "Detected Fraud"
            ),
            (
                "shap_case2_missed_fraud.png",
                "Missed Fraud"
            ),
            (
                "shap_case3_genuine.png",
                "Genuine Transaction"
            )
        ]:

            path = file_path(
                image
            )

            if os.path.exists(
                path
            ):

                st.image(
                    path,
                    caption=caption,
                    use_container_width=True
                )


# ============================================================
# PAGE 10
# FORENSIC DECISION SUPPORT
# ============================================================

elif page == "🚨 Forensic Decision Support":

    st.title(
        "🚨 Forensic Decision Support"
    )

    st.markdown(
        """
        ## From Prediction to Investigation

        The system does not stop after predicting fraud.

        It combines:

        ### 1. Fraud Likelihood

        Machine-learning assessment.

        ### 2. Behavioral Evidence

        Historical transaction behavior.

        ### 3. Financial Evidence

        Transaction amount and amount deviation.

        ### 4. Forensic Evidence Score

        Combined evidence for prioritization.

        ### 5. Investigation Decision

        Converts analytical evidence into an actionable
        investigation category.
        """
    )

    decision_columns = [
        "actual_fraud",
        "fraud_probability",
        "behavioral_risk_score",
        "behavioral_risk_level",
        "amount_evidence",
        "behavioral_anomalies",
        "forensic_evidence_score",
        "evidence_strength",
        "investigation_decision",
        "recommended_action"
    ]

    with st.spinner(
        "Loading forensic evidence..."
    ):

        evidence_data = load_evidence_columns(
            decision_columns
        )

    if evidence_data is None:

        st.error(
            "forensic_evidence_analysis.csv not found."
        )

        st.stop()

    total = len(
        evidence_data
    )

    fraud = int(
        evidence_data[
            "actual_fraud"
        ].sum()
    )

    immediate = int(
        (
            evidence_data[
                "investigation_decision"
            ]
            ==
            "IMMEDIATE INVESTIGATION"
        ).sum()
    )

    strong = int(
        evidence_data[
            "evidence_strength"
        ]
        .isin(
            [
                "STRONG",
                "VERY STRONG"
            ]
        )
        .sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Transactions",
        f"{total:,}"
    )

    c2.metric(
        "Confirmed Fraud",
        f"{fraud:,}"
    )

    c3.metric(
        "Immediate Investigation",
        f"{immediate:,}"
    )

    c4.metric(
        "Strong Evidence",
        f"{strong:,}"
    )

    st.divider()

    st.subheader(
        "Investigation Decision Distribution"
    )

    decision_counts = (
        evidence_data[
            "investigation_decision"
        ]
        .value_counts()
    )

    st.bar_chart(
        decision_counts
    )

    st.dataframe(
        decision_counts.rename(
            "Cases"
        ),
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "Evidence Strength Distribution"
    )

    strength_counts = (
        evidence_data[
            "evidence_strength"
        ]
        .value_counts()
    )

    st.bar_chart(
        strength_counts
    )

    st.dataframe(
        strength_counts.rename(
            "Cases"
        ),
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "Fraud Rate by Investigation Decision"
    )

    fraud_decision = (
        evidence_data
        .groupby(
            "investigation_decision"
        )["actual_fraud"]
        .agg(
            total_cases="count",
            confirmed_fraud="sum",
            fraud_rate="mean"
        )
    )

    fraud_decision[
        "fraud_rate"
    ] *= 100

    st.dataframe(
        fraud_decision.round(4),
        use_container_width=True
    )

    st.bar_chart(
        fraud_decision[
            "fraud_rate"
        ]
    )

    st.divider()

    st.subheader(
        "Top Forensic Evidence Cases"
    )

    top = (
        evidence_data
        .sort_values(
            "forensic_evidence_score",
            ascending=False
        )
        .head(25)
    )

    st.dataframe(
        top,
        use_container_width=True,
        height=600
    )

    st.success(
        """
        The forensic decision-support layer transforms
        machine-learning output into investigation-oriented
        evidence and recommended actions.

        It supports human investigators rather than
        automatically declaring guilt.
        """
    )


# ============================================================
# PAGE 11
# CASE EXPLORER
# ============================================================

elif page == "📁 Forensic Case Explorer":

    st.title(
        "📁 Forensic Case Explorer"
    )

    explorer_columns = [
        "step",
        "amount",
        "actual_fraud",
        "fraud_probability",
        "behavioral_risk_score",
        "behavioral_risk_level",
        "amount_evidence",
        "behavioral_anomalies",
        "forensic_evidence_score",
        "evidence_strength",
        "investigation_decision",
        "recommended_action"
    ]

    with st.spinner(
        "Loading forensic cases..."
    ):

        explorer = load_evidence_columns(
            explorer_columns
        )

    if explorer is None:

        st.error(
            "Forensic evidence data unavailable."
        )

        st.stop()

    decisions = sorted(
        explorer[
            "investigation_decision"
        ]
        .dropna()
        .unique()
    )

    strengths = sorted(
        explorer[
            "evidence_strength"
        ]
        .dropna()
        .unique()
    )

    c1, c2, c3 = st.columns(3)

    selected_decisions = c1.multiselect(
        "Investigation Decision",
        decisions,
        default=decisions
    )

    selected_strengths = c2.multiselect(
        "Evidence Strength",
        strengths,
        default=strengths
    )

    min_probability = c3.slider(
        "Minimum Fraud Probability",
        0.0,
        1.0,
        0.0,
        0.01
    )

    filtered = explorer.copy()

    if selected_decisions:

        filtered = filtered[
            filtered[
                "investigation_decision"
            ].isin(
                selected_decisions
            )
        ]

    if selected_strengths:

        filtered = filtered[
            filtered[
                "evidence_strength"
            ].isin(
                selected_strengths
            )
        ]

    filtered = filtered[
        filtered[
            "fraud_probability"
        ]
        >= min_probability
    ]

    st.subheader(
        f"Matching Cases: {len(filtered):,}"
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        height=600
    )

    csv = filtered.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        "⬇️ Download Filtered Cases",
        csv,
        "forensic_cases.csv",
        "text/csv"
    )


# ============================================================
# PAGE 12
# INVESTIGATION REPORT
# ============================================================

elif page == "📄 Investigation Report":

    st.title(
        "📄 Forensic Investigation Report"
    )

    st.markdown(
        """
        # Financial Fraud Detection & Forensic Intelligence

        ## Objective

        Identify potentially fraudulent financial transactions
        and provide interpretable evidence to support human
        forensic investigation.

        ---

        ## Final Model

        **Random Forest**

        The final model was evaluated against:

        - XGBoost
        - LightGBM
        - CatBoost
        - Extra Trees

        ---

        ## Final Test Performance

        **Accuracy:** 99.9998%

        **Precision:** 100%

        **Recall:** 99.9501%

        **F1-Score:** 99.9750%

        **ROC-AUC:** 1.0000

        **PR-AUC:** 0.999999

        **False Positives:** 0

        **False Negatives:** 2

        ---

        ## Behavioral Profiling

        Behavioral profiling adds historical context to the
        current transaction.

        It evaluates:

        - Previous transaction activity
        - Amount deviation
        - Transaction timing
        - Destination diversity
        - Behavioral risk

        ---

        ## Probability Calibration

        Random Forest likelihood scores were calibrated using
        the validation period.

        The calibration stage improved Brier score while
        preserving classification performance.

        ---

        ## Threshold Selection

        A cost-sensitive analysis selected:

        **Threshold = 0.75**

        The threshold was selected on validation data and then
        applied to the unseen chronological test period.

        ---

        ## Temporal Analysis

        Temporal analysis showed that fraud prevalence changes
        across chronological periods.

        This provides evidence that the fraud environment
        should be continuously monitored.

        ---

        ## Feature Ablation

        Feature ablation demonstrated that engineered balance
        features provide substantial predictive value.

        ---

        ## Explainable AI

        SHAP provides:

        **Global explanations**

        and

        **Individual transaction explanations**

        ---

        ## Forensic Decision Support

        The final layer combines:

        **Fraud likelihood**

        +

        **Behavioral evidence**

        +

        **Financial evidence**

        ↓

        **Forensic Evidence Score**

        ↓

        **Evidence Strength**

        ↓

        **Investigation Decision**

        ↓

        **Recommended Investigator Action**

        ---

        ## Final Principle

        The system is a forensic decision-support tool.

        A high score should trigger investigation and evidence
        collection, not automatically establish guilt.
        """
    )

    st.divider()

    st.subheader(
        "Complete Project Architecture"
    )

    st.code(
        """
PaySim Dataset
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Chronological Train / Validation / Test
       ↓
Model Comparison
       ↓
Random Forest Selection
       ↓
Probability Calibration
       ↓
Behavioral Profiling
       ↓
Threshold & Cost Analysis
       ↓
Temporal / Concept Drift Analysis
       ↓
SHAP Explainable AI
       ↓
Forensic Evidence Analysis
       ↓
Evidence Strength
       ↓
Investigation Priority
       ↓
Recommended Investigator Action
       ↓
Interactive Forensic Dashboard
        """,
        language="text"
    )

    st.success(
        "Forensic Intelligence System Completed"
    )