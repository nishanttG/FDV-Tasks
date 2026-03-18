# All model training code (LogReg, RF, XGB, SMOTE, nested CV, calibration) goes here.

# Functions should accept X_train, y_train, preprocessor as arguments.
import os
import pickle
from copy import deepcopy
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def build_default_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    }
    param_grids = {
        "LogisticRegression": {"C":[0.01,0.1,1,10], "solver":["liblinear"]},
        "RandomForest": {"n_estimators":[200,400], "max_depth":[None,10]},
        "XGBoost": {"max_depth":[3,5], "learning_rate":[0.1,0.01]}
    }
    return models, param_grids

def nested_cv_select(X_train, y_train):
    models, param_grids = build_default_models()
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = {}
    for name, estimator in models.items():
        grid = GridSearchCV(estimator, param_grids[name], scoring='roc_auc', cv=inner_cv, n_jobs=-1)
        outer_scores = cross_val_score(grid, X_train, y_train, scoring='roc_auc', cv=outer_cv, n_jobs=-1)
        grid.fit(X_train, y_train)
        results[name] = {
            "mean_roc_auc": float(np.mean(outer_scores)),
            "std_roc_auc": float(np.std(outer_scores)),
            "best_params": grid.best_params_,
            "best_estimator": grid.best_estimator_
        }
    return results

def train_final_xgboost(X_train, y_train, prefit_model=None):
    if prefit_model is not None:
        model = deepcopy(prefit_model)
    else:
        model = XGBClassifier(eval_metric="logloss", n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    # save
    path = os.path.join(MODEL_DIR, "xgboost_model_day5.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return model, path
