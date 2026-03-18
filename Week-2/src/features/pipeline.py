# Put all preprocessing pipelines (numerical/categorical, scaling, one-hot encoding) here.
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

NUMERICAL = ["tenure", "MonthlyCharges", "TotalCharges"]

def build_preprocessor(X_train: pd.DataFrame):
    categorical = [c for c in X_train.columns if c not in NUMERICAL]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, NUMERICAL),
        ("cat", cat_pipeline, categorical)
    ])

    preprocessor.fit(X_train)
    return preprocessor

def transform_to_df(preprocessor, X):
    arr = preprocessor.transform(X)
    # recover column names similar to notebook
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
        [c for c in X.columns if c not in NUMERICAL]
    )
    all_cols = np.concatenate([NUMERICAL, cat_features])
    return pd.DataFrame(arr, columns=all_cols, index=X.index)

def process_and_save(X_train, X_test, y_train, y_test, save_dir="data/processed"):
    """
    Build the preprocessing pipeline, transform train/test,
    convert to DataFrames, and save to data/processed.
    """

    # ensure dir exists
    os.makedirs(save_dir, exist_ok=True)

    # 1. Build fitted preprocessor
    preprocessor = build_preprocessor(X_train)

    # 2. Transform to DataFrame (with correct feature names)
    X_train_df = transform_to_df(preprocessor, X_train)
    X_test_df = transform_to_df(preprocessor, X_test)

    # 3. Save processed features
    X_train_df.to_csv(os.path.join(save_dir, "X_train.csv"), index=False)
    X_test_df.to_csv(os.path.join(save_dir, "X_test.csv"), index=False)

    # 4. Save labels
    # NOTE: y_train/y_test come from preprocessing.split() where Churn is mapped to 0/1
    pd.DataFrame({"Churn": y_train}).to_csv(os.path.join(save_dir, "y_train.csv"), index=False)
    pd.DataFrame({"Churn": y_test}).to_csv(os.path.join(save_dir, "y_test.csv"), index=False)

    # 5. Save preprocessor object for inference
    import pickle
    with open(os.path.join("models", "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    return X_train_df, X_test_df, preprocessor