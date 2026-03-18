# Week 2 Day 1: Leakage Checklist – Telco Churn Prediction

**Project:** Supervised ML Under Constraints (Churn)  
**Date:** 24 Nov, 2025 (Monday) 
**Prepared by:** Nishant Ghimire


## 1. Data Split
- [x] Dataset is split into `X_train`, `X_test`, `y_train`, `y_test` **before any preprocessing**.
- [x] Target variable `Churn` is removed from features before model training.
- [x] Stratified split used to maintain target distribution (`stratify=y`).


## 2. Feature Preprocessing
- [x] Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) are imputed using **training set median** only.
- [x] Categorical features (e.g., `gender`, `Partner`, `Dependents`, `Contract`, etc.) are imputed using **training set most frequent value** only.
- [x] Categorical features are one-hot encoded **after split**, with unknown categories in test handled safely (`handle_unknown='ignore'`).


## 3. Target Variable
- [x] `Churn` is mapped to `0/1` only in training/test sets.
- [x] Target is **never included in feature set** during preprocessing.


## 4. Scaling
- [x] Standard scaling applied **only on training numerical features**.
- [x] Same scaling applied to test features after training scaler is fit.


## 5. Feature Engineering / Leakage Check
- [x] No future information is used in features.
- [x] Identifiers (`customerID`) are excluded from model input.
- [x] `TotalCharges` converted to numeric **after train/test split**, and missing values filled using **training median**.

## 6. Validation
- [x] Checked transformed `X_train_processed` and `X_test_processed` have no NaNs.
- [x] Checked that train/test columns match exactly.
- [x] Sample rows verified to ensure **target variable not present in features**.


## 7. Random Seed
- [x] Random seed set (`RANDOM_SEED = 42`) for reproducibility.
- [x] All splits and transformations are deterministic.


## 8. Notes / Decisions
- Missing `TotalCharges` values are filled using **median from training set**.
- Categorical encoding uses **OneHotEncoder with `handle_unknown='ignore'`** to avoid errors on unseen test categories.
- This checklist ensures **leakage-proof preprocessing** for baseline and future models.
