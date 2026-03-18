# src/main.py
import os
from src.data.preprocessing import load_and_validate, compute_checksums, stratified_split
from src.features.pipeline import build_preprocessor, transform_to_df, process_and_save
from src.models.train import nested_cv_select, train_final_xgboost
from src.models.evaluate import compute_classification_metrics, sweep_thresholds_cost
from src.utils.wandb_utils import init_wandb, finish_wandb
import pandas as pd

DATA_CSV = os.path.join("data", "raw", "telco-customer-churn-by-IBM.csv")

def main():
    print("1) Loading & validating data")
    df = load_and_validate(DATA_CSV)

    print("2) Compute checksums for raw data (for reproducibility)")
    cs = compute_checksums(os.path.join("data", "raw"))
    print("Checksums:", cs)

    print("3) Splitting")
    X_train, X_test, y_train, y_test = stratified_split(df)

    # Drop non-feature column 'customerID' before preprocessing
    X_train_feat = X_train.drop(columns=['customerID'], errors='ignore')
    X_test_feat = X_test.drop(columns=['customerID'], errors='ignore')

    # build feature preprocessor and save processed data
    print("4) Build preprocessor")
    X_train_processed, X_test_processed, preprocessor = process_and_save(
        X_train_feat, X_test_feat, y_train, y_test
    )

    # convert y to arrays
    y_train_arr = y_train.values
    y_test_arr = y_test.values

    # init wandb (toggle via env LOG_WANDB)
    run = init_wandb(project="Week-2", name="main_workflow")

    print("5) Nested CV selection (may take time)")
    cv_results = nested_cv_select(X_train_processed, y_train_arr)
    print("CV results:", {k: {"mean_roc_auc": v["mean_roc_auc"], "best_params": v["best_params"]} for k, v in cv_results.items()})

    print("6) Train final XGBoost on full training processed data")
    best_model, model_path = train_final_xgboost(X_train_processed, y_train_arr)

    print("7) Evaluate on test set using default threshold 0.5")
    y_prob_test = best_model.predict_proba(X_test_processed)[:,1]
    metrics = compute_classification_metrics(y_test_arr, y_prob_test, threshold=0.5)
    print("Metrics @ 0.5:", metrics)

    print("8) Run threshold sweep side quest (cost function default 500/100)")
    out_json, json_path, fig_path, checks = sweep_thresholds_cost(y_test_arr, y_prob_test, cost_fn=500, cost_fp=100)
    print("Side quest output:", out_json)
    print("Side quest checks:", checks)
    print("Saved threshold json:", json_path)
    print("Saved plot:", fig_path)

    if 'run' in locals() and run is not None:
        finish_wandb()

if __name__ == "__main__":
    main()
