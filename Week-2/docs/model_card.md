# Model Card: Telco Customer Churn Prediction

## 1. Model Details

*   **Model Name**: XGBoost Classifier for Telco Churn
*   **Version**: XGBoost 3.1.2
*   **Date**: November 29, 2025
*   **Owner**: nishanttg (nishantg)
*   **Model Type**: Binary Classification

## 2. Intended Use

*   **Primary Intended Users**: Telecommunication companies and their customer retention strategists.
*   **Primary Intended Uses**:
    *   To predict whether a customer is likely to churn (cancel their service).
    *   To identify key features influencing churn predictions for individual customers.
    *   To inform targeted customer retention campaigns by focusing on high-risk, high-impact customers or segments.
    *   To quantify the uncertainty of churn predictions, enabling more robust decision-making.
*   **Out-of-Scope Uses**:
    *   Predicting churn for industries other than telecommunications.
    *   Automated decision-making without human oversight.
    *   Making predictions on data with significantly different distributions or feature sets than the training data.

## 3. Training Data

*   **Dataset Name**: `telco-customer-churn-by-IBM.csv`
*   **Source**: IBM (commonly used public dataset for telco churn).
*   **Size**: 5634 samples (80% of original 7043 samples) for training after initial split.
*   **Data Preprocessing**:
    *   The `customerID` column was dropped.
    *   `TotalCharges` column was converted to a numeric type, with `coerce` errors (introducing 8 missing values in the training set).
    *   **Train/Test Split**: The dataset was split into 80% training and 20% testing data using `train_test_split` with `stratify=y` to maintain the class distribution in both sets, preventing data leakage.
    *   **Feature Engineering/Transformation Pipeline**:
        *   **Numerical Features** (`tenure`, `MonthlyCharges`, `TotalCharges`):
            *   Missing values imputed with the median.
            *   Scaled using `StandardScaler`.
        *   **Categorical Features** (`gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`):
            *   Missing values imputed with the most frequent value.
            *   One-hot encoded using `OneHotEncoder(handle_unknown='ignore')`.
*   **Target Variable**: `Churn` (binary: `Yes` mapped to 1, `No` mapped to 0).
*   **Class Imbalance**: The training dataset exhibits class imbalance:
    *   `No Churn`: 73.46%
    *   `Churn`: 26.54%
    *   The Logistic Regression model used a `class_weight='balanced'` parameter to mitigate this, and SMOTE was explored for data balancing. The XGBoost model was trained on the preprocessed (but not oversampled) data directly, relying on its internal handling of imbalanced data through the `eval_metric="logloss"` (or potentially `scale_pos_weight` for explicit imbalance handling, though not explicitly shown in the final XGBoost config).

## 4. Evaluation Data

*   **Dataset**: The remaining 20% of the `telco-customer-churn-by-IBM.csv` dataset, stratified by the `Churn` variable.
*   **Size**: 1409 samples.
*   **Preprocessing**: Applied the same `ColumnTransformer` (fitted on the training data) to the test set to ensure consistency.
*   **Tenure-Based Split Evaluation**: A separate evaluation was performed where the training data (80% of original) was sorted by `tenure`, and then split again into 80% (older customers) for training and 20% (newer customers) for testing. This evaluates model performance on different customer lifecycle stages.

## 5. Metrics

*   **Primary Evaluation Metrics**:
    *   **F1-score**: Important for imbalanced datasets, as it balances precision and recall.
    *   **ROC-AUC (Receiver Operating Characteristic Area Under the Curve)**: Measures the model's ability to distinguish between classes across all possible thresholds.
    *   **PR-AUC (Precision-Recall Area Under the Curve) / Average Precision**: Crucial for imbalanced datasets, as it is more sensitive to changes in the minority class than ROC-AUC.
*   **Additional Metrics**:
    *   **Accuracy**: Overall correctness of predictions.
    *   **Precision**: Proportion of positive identifications that were actually correct.
    *   **Recall**: Proportion of actual positives that were identified correctly.
    *   **Confusion Matrix**: Provides a detailed breakdown of true positives, true negatives, false positives, and false negatives.
    *   **Brier Score**: Measures the accuracy of probabilistic predictions. Lower values indicate better calibration.
    *   **Expected Calibration Error (ECE)**: Quantifies how well predicted probabilities align with observed frequencies. Lower values indicate better calibration.
    *   **Custom Total Cost**: A business-centric metric calculated as `(false_negatives * cost_fn) + (false_positives * cost_fp)`. This helps find the optimal decision threshold based on specific business costs (e.g., cost of losing a customer vs. cost of a retention offer).
*   **Justification**: The emphasis on F1-score, PR-AUC, Brier Score, ECE, and the custom cost metric is due to the imbalanced nature of churn data and the business implications of misclassifications.

## 6. Model Performance

### 6.1. Nested Cross-Validation (on training data)

A nested cross-validation approach was used to robustly evaluate different models and their hyperparameter tuning, preventing overfitting to the validation set.

*   **Logistic Regression**:
    *   Mean ROC-AUC: 0.846 (Std: 0.013)
    *   Mean PR-AUC: 0.658 (Std: 0.017)
    *   Best Params: `{'C': 10, 'max_iter': 200, 'penalty': 'l2', 'solver': 'liblinear'}` (for ROC-AUC)
    *   Best Params: `{'C': 0.1, 'max_iter': 200, 'penalty': 'l1', 'solver': 'liblinear'}` (for PR-AUC)
*   **RandomForest Classifier**:
    *   Mean ROC-AUC: 0.844 (Std: 0.008)
    *   Mean PR-AUC: 0.656 (Std: 0.015)
    *   Best Params: `{'max_depth': 5, 'min_samples_split': 10, 'n_estimators': 600}` (for both ROC-AUC and PR-AUC)
*   **XGBoost Classifier**:
    *   Mean ROC-AUC: 0.849 (Std: 0.011)
    *   Mean PR-AUC: 0.669 (Std: 0.021)
    *   Best Params: `{'gamma': 1, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500}` (for ROC-AUC)
    *   Best Params: `{'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200}` (for PR-AUC)

XGBoost generally showed slightly better performance in terms of both ROC-AUC and PR-AUC.

### 6.2. Final XGBoost Model Performance (on test data)

The XGBoost model (with parameters `eval_metric="logloss", n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42`) was fitted on the entire `X_train_processed` and evaluated on `X_test_processed`.

*   **Accuracy**: 0.8013
*   **Precision**: 0.6577
*   **Recall**: 0.5241
*   **F1-score**: 0.5833
*   **ROC-AUC**: 0.8431
*   **PR-AUC**: 0.6625
*   **Confusion Matrix**:
    ```
    [[933 102]
     [178 196]]
    ```
    *   True Negatives (Correctly predicted no churn): 933
    *   False Positives (Incorrectly predicted churn): 102
    *   False Negatives (Incorrectly predicted no churn): 178
    *   True Positives (Correctly predicted churn): 196

### 6.3. Calibration Performance (on test data)

The model's predicted probabilities were evaluated for calibration using Brier Score and Expected Calibration Error (ECE). Sigmoid and Isotonic calibration methods were also applied.

*   **Brier Score**:
    *   Uncalibrated: 0.1361
    *   Sigmoid Calibrated: 0.1362 (Slightly worse)
    *   Isotonic Calibrated: 0.1358 (Slightly better)
*   **Expected Calibration Error (ECE)**:
    *   Uncalibrated: 0.0231
    *   Sigmoid Calibrated: 0.0325 (Worse)
    *   Isotonic Calibrated: 0.0238 (Slightly worse)

Isotonic calibration provided a marginal improvement in Brier Score but slightly worsened ECE. The uncalibrated model's probabilities are already fairly well-calibrated.

### 6.4. Cost-Benefit Analysis

A custom cost function was used to find an optimal decision threshold, assuming:
*   `cost_fn = 500` (cost of a False Negative - missing a churner)
*   `cost_fp = 100` (cost of a False Positive - offering retention to a non-churner)

*   **Minimum Total Cost**: 59400
*   **Optimal Threshold**: 0.150 (This threshold significantly reduces cost compared to the default 0.5).
*   **Cost at Threshold 0.5**: 99200
*   **Cost for All-Negative Baseline**: 187000 (predicting no churn for all customers).

### 6.5. Slicing Metrics (Performance on Subgroups)

Metrics were analyzed across different customer segments (slices) to identify areas of strong and weak performance.

*   **Contract Type**:
    *   `Contract = Month-to-month`: ROC-AUC = 0.755, PR-AUC = 0.699 (**Good segment**)
    *   `Contract = Two year`: ROC-AUC = 0.724, PR-AUC = 0.075 (**Problem segment - high false positives**)
    *   `Contract = One year`: ROC-AUC = 0.724, PR-AUC = 0.256
*   **Tenure Group**:
    *   `tenure_group = 0-12` (New customers): ROC-AUC = 0.767, PR-AUC = 0.750 (**Good segment - best performance**)
    *   `tenure_group = 12-24`: ROC-AUC = 0.801, PR-AUC = 0.599
    *   `tenure_group = 24+` (Long-term customers): ROC-AUC = 0.825, PR-AUC = 0.442
*   **Payment Method**:
    *   `PaymentMethod = Electronic check`: ROC-AUC = 0.782, PR-AUC = 0.733 (**Good segment**)
    *   `PaymentMethod = Credit card (automatic)`: ROC-AUC = 0.853, PR-AUC = 0.566
    *   `PaymentMethod = Bank transfer (automatic)`: ROC-AUC = 0.867, PR-AUC = 0.643
    *   `PaymentMethod = Mailed check`: ROC-AUC = 0.816, PR-AUC = 0.508 (**Problem segment - too many wrong "churn" predictions**)

### 6.6. Tenure-Based Split Evaluation

When the model was trained on data from earlier tenure periods and tested on later tenure periods (simulating a "time-based" split on tenure), the performance significantly dropped:

*   **Accuracy**: 0.9228 (High, but misleading due to class imbalance and very low TP rate)
*   **Precision**: 0.3333
*   **Recall**: 0.1299
*   **F1-score**: 0.1869
*   **ROC-AUC**: 0.7807
*   **PR-AUC**: 0.2242

This indicates that the model struggles to generalize to customers in different stages of their tenure if trained purely on historical data from earlier stages. This highlights a limitation when applying the model to new or evolving customer populations.

## 7. Ethical Considerations and Limitations

*   **Bias**: The model's performance varies across different demographic (e.g., `gender`, `SeniorCitizen`) and behavioral (e.g., `Contract`, `PaymentMethod`) segments. It's crucial to acknowledge and investigate these disparities to ensure fairness and prevent discriminatory outcomes in retention strategies. The slicing analysis is a good first step.
*   **Interpretability**: While SHAP values provide local (individual predictions) and global (overall feature importance) interpretability, these explanations should be used in conjunction with domain expertise.
*   **Generalizability**: The model's performance, especially when evaluated with a tenure-based split, suggests potential issues with generalizing to different customer lifecycle stages or if customer behavior changes over time (concept drift). Continuous monitoring and retraining are essential.
*   **Data Quality**: The model's effectiveness is dependent on the quality and representativeness of the `telco-customer-churn-by-IBM.csv` dataset. Missing values in `TotalCharges` were handled but could indicate underlying data issues.
*   **Actionable Insights**: The cost analysis and slicing provide valuable, actionable insights for retention campaigns. However, the model outputs should guide, not replace, human decision-making and ethical considerations.
*   **Dynamic Thresholds**: The optimal decision threshold is derived from a specific cost matrix and may change as business priorities or actual costs evolve. This necessitates regular re-evaluation of the threshold.
*   **Conformal Prediction**: While providing a measure of uncertainty, the generated intervals can sometimes be wide, indicating significant uncertainty. This means the model might not be able to provide confident predictions for all customers, and actions should be adjusted accordingly.

## 8. Explainability (SHAP)

SHAP (SHapley Additive exPlanations) was used to understand feature contributions to the model's predictions.

*   **Global Feature Importance (Bar Plot)**:
    *   A SHAP bar plot (`shap_values.abs.mean(0)`) indicates the average magnitude of impact each feature has on the model output across the dataset. This helps identify the most influential factors contributing to churn decisions overall.
*   **Local Explanations (Force Plot)**:
    *   A SHAP force plot (`shap.force_plot`) for individual instances shows how each feature's value contributes to pushing the model's output from the `base_value` to the final predicted `output_value`. Red indicates features pushing the prediction higher (towards churn in this context), and blue indicates features pushing it lower (towards no churn).

## 9. Model Artifacts

*   **`xgboost_model_day5.pkl`**: A pickled file of the trained XGBoost classifier, saved and logged as a W&B artifact.
*   **`threshold.json`**: A JSON file storing the `best_threshold` and `expected_cost` derived from the cost-benefit analysis.
*   **`plot.png`**: An image of the "Threshold vs Total Cost" plot, showing the optimal threshold.
*   **W&B Project**: "Week-2" (https://wandb.ai/nishantg/Week-2) contains logs for model comparisons, explainability plots, calibration results, and slicing metrics.

## 10. Limitations and Recommendations

*   **Data Drift:** The model's performance may degrade over time if customer behavior or market conditions change. Continuous monitoring of data drift and model performance is recommended.
*   **Feature Engineering:** Further feature engineering (e.g., interaction terms, polynomial features) could potentially improve model performance.
*   **Model Complexity:** While XGBoost performs well, exploring simpler, more interpretable models for certain low-stakes decisions could be beneficial if performance trade-offs are acceptable.
*   **Retention Strategies:** This model identifies churn risk. The effectiveness of different retention strategies is a separate, but critical, area for A/B testing and analysis.
*   **Fairness Trade-offs:** Business decisions should consider the trade-offs between overall model performance and fairness across different segments. For example, while the model is less reliable for "Two-year contracts," it might still be capturing some true churners.

