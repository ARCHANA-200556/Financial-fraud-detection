Explainable Financial Fraud Detection and Forensic Intelligence Using Machine Learning
Developed as part of the course on "Forensic Analytics" at Vellore Institute of Technology, Chennai.

by "Veligaram Archana (23MIA1149) ","Yamasani Lahari Sree (23MIA1129) "

Subject Code: CSE4074

Semester: Fall 2026-27

Professor: Dr. S. Johanan Joysingh
1. Project Overview
This project develops an interactive machine-learning workflow to estimate fraud likelihood for financial transactions and support forensic investigation. It uses a Random Forest classifier as the principal model in the application and integrates analysis for model interpretation, behavioural profiling, probability calibration, threshold-cost analysis, temporal exploration, and forensic reporting.
2. Problem Statement
Traditional rule-based systems rely on predefined conditions. Machine-learning systems can learn patterns from historical transaction data, but a prediction alone may not provide enough context for an investigator to examine a suspicious transaction. This project combines fraud-likelihood prediction with supporting analytical and forensic tools in one interactive workflow.
3. Objectives
- Estimate fraud likelihood for financial transactions using machine learning.
- Engineer transaction and balance-related features.
- Compare selected machine-learning classifiers.
- Explore behavioural risk patterns and temporal changes.
- Analyse probability calibration and classification-threshold trade-offs.
- Use SHAP-based analysis to help interpret model behaviour.
- Provide an interactive interface for examining results and supporting forensic investigation.
4. Methodology
1. Load and inspect the PaySim synthetic transaction dataset.
2. Preprocess the data and engineer transaction-related features.
3. Split the data chronologically into training, validation, and test sets.
4. Train and evaluate machine-learning models.
5. Examine prediction scores, model performance, behavioural patterns, and temporal trends.
6. Apply SHAP-based analysis to interpret feature contributions.
7. Integrate the implemented analyses into the Streamlit application.
5. Main Features
Depending on the scripts included in the repository, features may include fraud-likelihood prediction, model comparison, behavioural risk profiling, probability calibration, threshold-cost analysis, temporal fraud analysis, SHAP-based explanations, and forensic case exploration and reporting.
6. Dataset
The project uses the PaySim synthetic financial transaction dataset. The original dataset used in the project contains approximately 6.36 million transactions, with transaction types, amounts, account balances, and fraud labels.

The dataset is synthetic, so results should not automatically be interpreted as equivalent performance on real-world banking transactions. The full dataset is not included by default. Obtain it from a permitted source and place it at the path expected by the project scripts.
7. Technology Stack
Python, Pandas, NumPy, Scikit-learn, SHAP, Streamlit, Matplotlib, Plotly, and model libraries such as XGBoost, LightGBM, or CatBoost where required by the included code. Keep requirements.txt aligned with the actual imports in your project.
8. Installation and Execution
Create and activate a virtual environment (optional but recommended):
python -m venv .venv
.venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
Confirm which file is the actual Streamlit entry point. If it is app.py, run:
streamlit run app.py
If it is main.py, run:
streamlit run main.py
The application may also require the dataset and trained model files to be present at the paths expected by the code.
9. Evaluation
Where implemented, the project evaluates models using precision, recall, F1-score, ROC-AUC, and PR-AUC in addition to accuracy. Fraud is a minority class, so accuracy alone is not sufficient. Report the data split, threshold, confusion matrix, and class distribution alongside results. Check for feature leakage and generalisation limitations, especially when scores are unusually high.
10. Project Contribution
The project does not claim that Random Forest, SHAP, or behavioural analysis is individually new. Its intended contribution is the integration of established techniques into an interactive forensic workflow combining prediction, explanation, behavioural profiling, threshold analysis, temporal exploration, and investigation support.
11. Limitations and Future Scope
PaySim is synthetic and may not represent every real-world fraud pattern. Further validation on independent datasets is needed. Future work may include testing additional datasets, drift monitoring, improved case management, and feedback from domain experts.
12. Academic Details
- Team Members: Veligaram Archana (23MIA1149) ,Yamasani Lahari Sree (23MIA1129)
- Professor: Dr. S. Johanan Joysingh
- Institution: Vellore Institute of Technology, Chennai
- Course: Forensic Analytics
- Subject Code: CSE4074
- Semester: Fall 2026-27