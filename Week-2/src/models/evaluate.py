# Metrics calculation, PR/ROC curves, Brier, calibration, threshold analysis, fairness slices.
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

ARTIFACT_DIR = os.path.join(os.getcwd(), "reports")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob))
    }
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics

def sweep_thresholds_cost(y_true, y_prob, cost_fn=500, cost_fp=100, n_thresholds=101):
    # Side quest expects costs: cost_fn=5, cost_fp=1 by default, but your notebook used 500/100 earlier.
    thresholds = np.linspace(0,1,n_thresholds)
    total_costs = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        total_costs.append(int(fn * cost_fn + fp * cost_fp))
    min_idx = int(np.argmin(total_costs))
    best_t = float(thresholds[min_idx])
    min_cost = int(total_costs[min_idx])

    # Save JSON and plot
    out_json = {"threshold": best_t, "expected_cost": min_cost}
    json_path = os.path.join(ARTIFACT_DIR, "threshold.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f)

    # Plot
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(thresholds, total_costs, marker='o', markersize=3)
    ax.axvline(best_t, color='red', linestyle='--', label=f"best={best_t:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total cost")
    ax.set_title("Threshold vs Expected Cost")
    ax.legend()
    fig_path = os.path.join(ARTIFACT_DIR, "plot.png")
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)

    # Baseline checks
    # cost at 0.5
    y_pred_05 = (y_prob >= 0.5).astype(int)
    cost_05 = int(((y_pred_05 == 0) & (y_true == 1)).sum() * cost_fn + ((y_pred_05 == 1) & (y_true == 0)).sum() * cost_fp)
    # predict all negative baseline
    y_pred_all_neg = np.zeros_like(y_true)
    cost_all_neg = int(((y_pred_all_neg == 0) & (y_true == 1)).sum() * cost_fn)

    checks = {
        "cost_at_0.5": cost_05,
        "cost_all_negative": cost_all_neg,
        "chosen_cost": min_cost,
        "passes_cost_checks": (min_cost <= cost_05) and (min_cost <= cost_all_neg)
    }

    return out_json, json_path, fig_path, checks
