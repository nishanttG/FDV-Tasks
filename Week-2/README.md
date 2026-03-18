# Week 2: Supervised ML Under Constraints (Telco Customer Churn)

This project focuses on building a robust machine learning pipeline for predicting customer churn in a telecommunications dataset, with a strong emphasis on rigorous evaluation, model calibration, cost-sensitive decision-making, explainability, and fairness.

## Table of Contents
1.  [Introduction](#1-introduction)
2.  [Dataset](#2-dataset)
3.  [5-Day Plan Summary](#3-5-day-plan-summary)
    *   [Day 1: Problem Framing & Leakage Audit](#day-1-problem-framing--leakage-audit)
    *   [Day 2: Baselines & Class Imbalance](#day-2-baselines--class-imbalance)
    *   [Day 3: Model Suite & Nested CV](#day-3-model-suite--nested-cv)
    *   [Day 4: Explainability & Calibration](#day-4-explainability--calibration)
    *   [Day 5: Fairness & Deployment-Ready Artifact](#day-5-fairness--deployment-ready-artifact)
4.  [Side Quest: Threshold Picker](#4-side-quest-threshold-picker)
5.  [Deliverables](#5-deliverables)
6.  [House Rules Adherence](#6-house-rules-adherence)
7.  [How to Run](#7-how-to-run)
8.  [Decision Log](#8-decision-log)

---

## 1. Introduction

Customer churn is a critical business problem, especially in competitive industries like telecommunications. This project aims to develop a predictive model to identify customers likely to churn, incorporating real-world business constraints such as imbalanced datasets and varying costs of false positives and false negatives. The focus is on building a reliable and fair model that can inform strategic decisions to retain customers.

## 2. Dataset

The dataset used is the Telco Customer Churn by IBM, available at:
[https://github.com/plotly/datasets/blob/master/telco-customer-churn-by-IBM.csv](https://github.com/plotly/datasets/blob/master/telco-customer-churn-by-IBM.csv)

**Data Dictionary:**

*   **customerID:** Customer ID (identifier).
*   **gender:** Customer's gender (categorical).
*   **SeniorCitizen:** Whether the customer is a senior citizen (0 or 1, numerical).
*   **Partner:** Whether the customer has a partner (Yes/No, categorical).
*   **Dependents:** Whether the customer has dependents (Yes/No, categorical).
*   **tenure:** Number of months the customer has stayed with the company (numerical).
*   **PhoneService:** Whether the customer has phone service (Yes/No, categorical).
*   **MultipleLines:** Whether the customer has multiple lines (Yes/No/No phone service, categorical).
*   **InternetService:** Customer's internet service provider (DSL/Fiber optic/No, categorical).
*   **OnlineSecurity:** Whether the customer has online security (Yes/No/No internet service, categorical).
*   **OnlineBackup:** Whether the customer has online backup (Yes/No/No internet service, categorical).
*   **DeviceProtection:** Whether the customer has device protection (Yes/No/No internet service, categorical).
*   **TechSupport:** Whether the customer has tech support (Yes/No/No internet service, categorical).
*   **StreamingTV:** Whether the customer has streaming TV (Yes/No/No internet service, categorical).
*   **StreamingMovies:** Whether the customer has streaming movies (Yes/No/No internet service, categorical).
*   **Contract:** The customer's contract type (Month-to-month/One year/Two year, categorical).
*   **PaperlessBilling:** Whether the customer has paperless billing (Yes/No, categorical).
*   **PaymentMethod:** The customer's payment method (Categorical).
*   **MonthlyCharges:** The amount charged to the customer monthly (numerical).
*   **TotalCharges:** The total amount charged to the customer (numerical, converted from object).
*   **Churn:** Whether the customer churned (Yes/No, **Target Variable**).

## 3. 5-Day Plan Summary

### Day 1: Problem Framing & Leakage Audit

*   **Binary Target:** The target variable `Churn` is binary (Yes/No). It was mapped to 1/0 for modeling.
*   **Business Costs:** Defined business costs for misclassifications:
    *   `cost_fn` (False Negative - missing a churner): 500 (since missing a churner is considered 5 times worse).
    *   `cost_fp` (False Positive - wrongly predicting churn for a loyal customer): 100.
*   **Feature Pipeline:** An `sklearn` Pipeline was constructed using `ColumnTransformer` for preprocessing:
    *   Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`): Imputed with median, then scaled using `StandardScaler`.
    *   Categorical features: Imputed with most frequent, then one-hot encoded using `OneHotEncoder`.
*   **Data Validation:** `Pandera` was used to define a schema for the DataFrame, ensuring data quality and integrity. This helps in failing fast on schema drift.
*   **Stratified Split:** The dataset was split into training (80%) and testing (20%) sets using `train_test_split` with `stratify=y` to maintain the class distribution in both sets, preventing data leakage.
*   **Leakage Audit:** Confirmed no data leakage by checking feature engineering steps are applied *after* the train-test split and evaluating class distribution consistency. `customerID` was dropped as it holds no predictive power and could lead to leakage.

### Day 2: Baselines & Class Imbalance

*   **Class Imbalance:** Noted significant class imbalance in the `Churn` variable.
*   **Baselines:**
    *   **Majority Baseline:** A model predicting the majority class (`No`) achieved an F1-score of 0.0 and ROC-AUC of 0.5 (as it makes no positive predictions for churn).
    *   **Vanilla Logistic Regression:** Trained a basic Logistic Regression model, establishing a performance benchmark.
*   **Handling Imbalance:**
    *   **Class Weights (`class_weight='balanced'`):** Explored adjusting class weights in Logistic Regression to penalize misclassification of the minority class more heavily.
    *   **SMOTE (Synthetic Minority Over-sampling Technique):** Applied SMOTE to oversample the minority class in the training data, creating synthetic samples.
*   **Justification with PR Curves:** Precision-Recall (PR) curves were used to visualize and compare the performance of models (vanilla, class-weighted, SMOTE-treated), especially crucial for imbalanced datasets where ROC-AUC can be misleading. Logged PR-AUC scores to W&B.

### Day 3: Model Suite & Nested CV

*   **Model Suite:** Implemented and compared three different machine learning models:
    *   Logistic Regression (`LogisticRegression`)
    *   Random Forest (`RandomForestClassifier`)
    *   XGBoost (`XGBClassifier`)
*   **Hyperparameter Tuning:** Defined hyperparameter grids for each model.
*   **Nested Cross-Validation:** Utilized nested cross-validation (outer 5-fold, inner 3-fold stratified KFold) for unbiased model selection and robust performance estimation, mitigating overfitting to the test set.
    *   The outer loop provides an unbiased estimate of the model's performance.
    *   The inner loop is used for hyperparameter tuning.
*   **Metric Logging:** Logged mean ROC-AUC and PR-AUC scores, along with their standard deviations across folds, for each model to W&B. XGBoost showed slightly better performance overall.

### Day 4: Explainability & Calibration

*   **Model Selection:** Based on Day 3 results, XGBoost was selected as the best performing model.
*   **Explainability with SHAP:**
    *   SHAP (SHapley Additive exPlanations) was used to interpret the XGBoost model's predictions.
    *   Generated `shap.summary_plot(shap_values, X_test_processed, plot_type="bar")` to visualize feature importance.
    *   Generated `shap.force_plot` for individual predictions to understand local interpretability.
*   **Probability Calibration:**
    *   Evaluated the uncalibrated predicted probabilities (histogram plotted, Brier score, and ECE calculated).
    *   Applied two calibration techniques: Platt scaling (`CalibratedClassifierCV(method="sigmoid")`) and Isotonic Regression (`CalibratedClassifierCV(method="isotonic")`).
    *   Compared calibrated probabilities using Brier score, Expected Calibration Error (ECE), and reliability diagrams.
*   **Cost-Sensitive Decision Making (Side Quest):**
    *   Swept through a range of decision thresholds (0 to 1).
    *   Calculated the `total_cost` (based on `cost_fn=500`, `cost_fp=100`) for each threshold.
    *   Identified the `best_threshold` that minimized the total cost.
    *   Generated a plot of "Threshold vs Total Cost".
    *   Saved the best threshold and its expected cost to `threshold.json`.
    *   Confirmed chosen threshold leads to lower cost than 0.5 threshold and "predict-all-negative" baseline.

### Day 5: Fairness & Deployment-Ready Artifact

*   **Fairness Analysis (Slice Metrics):**
    *   Segmented the test data (`X_test_df`) by various demographic and service-related features: `Contract`, `tenure_group`, `PaymentMethod`.
    *   Computed ROC-AUC and PR-AUC for each slice (e.g., "Contract = Two year", "tenure_group = 0-12", "PaymentMethod = Electronic check").
    *   Logged these slice-specific metrics to W&B for comparative analysis.
    *   **Discussion on Trade-offs:** Identified segments where the model performs well (e.g., "Month-to-month contracts", "New customers", "Electronic check payments") and problematic segments with large gaps between ROC-AUC and PR-AUC (e.g., "Two-year contracts", "Mailed check payments"). This highlights where the model might be making many false positive "churn" predictions, leading to wasted retention efforts.
*   **Deployment-Ready Artifact:**
    *   Saved the final trained XGBoost model as a `pickle` file (`xgboost_model_day5.pkl`).
    *   Logged this `.pkl` file as a versioned artifact to W&B.
    *   A model card (referencing an external file `model_card.md` if available, or including its content here) would document model details, assumptions, and limitations.

## 4. Side Quest: Threshold Picker

**Purpose:** To transform model probability scores into actionable business decisions by selecting an optimal decision threshold based on defined business costs.

**Inputs:**
*   `preds.csv` (hypothetical, but `y_test` and `y_prob` from the model are used) with columns: `y_true`, `p_hat` (predicted probability for the positive class).
*   Costs: `cost_fn = 500` (False Negative), `cost_fp = 100` (False Positive).

**Do:**
1.  **Threshold Sweep:** Iterate through a range of thresholds from 0 to 1 (e.g., 0.01 increments).
2.  **Expected Cost Computation:** For each threshold, predict labels (`y_pred`) and calculate the total expected cost using the formula: `total_cost = (FN * cost_fn) + (FP * cost_fp)`.
3.  **Optimal Threshold Selection:** Identify the threshold that results in the minimum `total_cost`.
4.  **Visualization:** Generate a plot showing "Decision Threshold" on the x-axis and "Total Cost" on the y-axis, with the optimal threshold highlighted.

**Outputs:**
*   `threshold.json`: A JSON file containing the `best_threshold` and its `expected_cost`.
    *   Example: `{"threshold": 0.150, "expected_cost": 59400}`
*   `plot.png`: An image file of the "Threshold vs Total Cost" plot.

**Pass Checks:**
*   The `threshold.json` file exists and contains a `threshold` value between 0 and 1.
*   The `plot.png` file exists.
*   The `expected_cost` at the `best_threshold` is less than or equal to the cost at the default threshold of 0.5, and less than or equal to the cost of the "predict-all-negative" baseline.

## 5. Deliverables

*   **Reproducible Training Script:** `Week-2.ipynb` (Jupyter notebook) contains all the code for data loading, preprocessing, model training, evaluation, explainability, calibration, and fairness analysis.
*   **Experiment Logs:** All key parameters, metrics (ROC-AUC, PR-AUC, F1, Brier, ECE, slice metrics), and plots are logged to [Weights & Biases (W&B)](https://wandb.ai/nishantg/Week-2).
*   **Comparison Report:** The W&B runs provide a comprehensive comparison report of model performance across different baselines and configurations.
*   **Model Card:** (The `model_card.md` file, if provided separately, would be referenced here. If not, its key elements would be included in this README.)
*   **Serialized Model:** `xgboost_model_day5.pkl` is saved as a W&B artifact for deployment.
*   **Inference Script:** (An inference script would typically accompany the serialized model. For this context, the model can be loaded and used directly from the notebook if a separate script isn't explicitly required.)
*   **`threshold.json` and `plot.png`:** Outputs from the "Threshold Picker" side quest.

## 6. House Rules Adherence

*   **Reproducible Environment:**
    *   `requirements.txt` is generated to lock dependencies using `pip compile`.
    *   Python version (`sys.version`) is printed in the notebook.
    *   Random seeds (`RANDOM_SEED = 42`) are consistently set for reproducibility across NumPy and other libraries.
    *   MD5 checksums for input files are computed and logged to W&B to verify data integrity.
*   **Data Validation:**
    *   `Pandera` schema validation is implemented in Day 1 to ensure data quality and fail fast on schema drift.
*   **Experiment Tracking:**
    *   [Weights & Biases (W&B)](https://wandb.ai/nishantg/Week-2) is used extensively throughout the notebook to log parameters, metrics, and artifacts, providing a clear audit trail of experiments.
*   **Readme:** This `README.md` file itself serves as the project documentation.
*   **Tests:** Unit tests and coverage reports were not explicitly developed within this notebook context. This would be a crucial next step in a full production pipeline to ensure the reliability of data transforms, model components, and metrics.
*   **APIs / Streamlit:** Implementation of FastAPI or Streamlit applications was not part of the notebook's direct output. These would be integrated in a subsequent phase for model deployment as web services.

## 7. How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Week-2
    ```
2.  **Set up the environment (recommended — conda):**

    - Using `conda` (recommended):

    ```bash
    conda create -n churn-week2 python=3.10 -y
    conda activate churn-week2
    pip install pip-tools
    pip-compile requirements.in
    pip install -r requirements.txt
    ```

    - Alternative (virtualenv / venv):

    ```bash
    python -m venv .venv
    # macOS / Linux
    source .venv/bin/activate
    # Windows (PowerShell)
    .venv\\Scripts\\Activate.ps1
    # or Windows (cmd.exe)
    .venv\\Scripts\\activate
    pip install -r requirements.txt
    ```

    *(Note: `pip-compile` comes from `pip-tools` and is used to generate a locked `requirements.txt` from `requirements.in`.)*
3.  **W&B Login:** Ensure you are logged into W&B. If not, run:
    ```bash
    wandb login
    ```
4.  **Run the Jupyter Notebook:** Open and run the `Week-2.ipynb` notebook.
    ```bash
    jupyter notebook
    ```
5. **Run pipeline without W&B:**
    ``` set LOG_WANDB=0 && python -m src.main ```

    **Run pipeline with W&B:**
    ``` set LOG_WANDB=1 && python -m src.main ```
    
    Execute all cells in the notebook to reproduce the results, logs, and artifacts.

6.  **View Results:** Access the experiment logs and artifacts on the W&B dashboard via the links provided in the notebook outputs.

## 8. Decision Log

*   **Dataset Choice:** Telco Customer Churn, due to its common use in churn prediction and clear binary target, making it suitable for cost-sensitive modeling.
*   **Business Costs:** False Negative (missing churner) is weighted higher (500) than False Positive (100) based on typical business impact where losing a customer is more costly than a potentially unnecessary retention effort.
*   **Preprocessing:** Standardized numerical features and one-hot encoded categorical features to prepare data for various ML models. Median imputation for numerical NaNs and most-frequent for categorical were chosen as robust strategies for this dataset.
*   **Split Strategy:** Stratified train-test split was crucial due to class imbalance, ensuring representative subsets for training and evaluation.
*   **Imbalance Handling:** Explored both `class_weight='balanced'` and `SMOTE`. PR curves showed that `class_weight='balanced'` and SMOTE yielded comparable performance, improving recall for the minority class, which is important for churn prediction. SMOTE was applied first as a common oversampling technique to directly address the imbalance in the training data, then class weights were used to fine-tune the logistic regression.
*   **Model Suite:** Logistic Regression (simplicity, interpretability), Random Forest (ensemble, robust to non-linearity), and XGBoost (gradient boosting, high performance) were chosen to cover a range of model complexities and performance characteristics.
*   **Nested CV:** Essential for obtaining a reliable, unbiased estimate of model performance and avoiding hyperparameter tuning bias.
*   **Explainability:** SHAP was chosen for its model-agnostic nature and ability to provide both local and global interpretability, critical for understanding driver of churn.
*   **Calibration:** Platt scaling and Isotonic regression were selected as standard calibration methods, important for ensuring that predicted probabilities accurately reflect true likelihoods, which is vital for cost-sensitive decision-making.
*   **Optimal Threshold:** The cost curve was used to find the optimal threshold. This directly addresses the business problem by minimizing total misclassification costs. The selected threshold is `0.150` with a minimum total cost of `59400`. This outperforms the default `0.5` threshold (`99200`) and the predict-all-negative baseline (`187000`), demonstrating the value of cost-sensitive decision making.
*   **Fairness Slices:** Analyzing metrics across different segments (contract, tenure, payment method) helps uncover potential biases or performance discrepancies, enabling more ethical and targeted interventions.
*   **Deployment Artifact:** Pickling the model is a straightforward way to save the trained model for later use, adhering to common MLOps practices.
