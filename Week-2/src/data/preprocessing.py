# All code related to loading dataset, validating schema (Pandera), train/test split, feature conversions.
# src/data/preprocessing.py
import os
import hashlib
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def md5_checksum(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def load_and_validate(csv_path: str):
    df = pd.read_csv(csv_path)
    # Minimal schema based on your notebook
    schema = pa.DataFrameSchema({
        "customerID": Column(pa.String, nullable=False),
        "gender": Column(pa.String, checks=Check.isin(["Male","Female"]), nullable=False),
        "SeniorCitizen": Column(pa.Int, checks=Check.isin([0,1]), nullable=False),
        "Partner": Column(pa.String, checks=Check.isin(["Yes","No"]), nullable=False),
        "Dependents": Column(pa.String, checks=Check.isin(["Yes","No"]), nullable=False),
        "tenure": Column(pa.Int, checks=Check.ge(0), nullable=False),
        "PhoneService": Column(pa.String, checks=Check.isin(["Yes","No"]), nullable=False),
        "MultipleLines": Column(pa.String, checks=Check.isin(["Yes","No","No phone service"]), nullable=False),
        "InternetService": Column(pa.String, checks=Check.isin(["DSL","Fiber optic","No"]), nullable=False),
        "OnlineSecurity": Column(pa.String, checks=Check.isin(["Yes","No","No internet service"]), nullable=False),
        "OnlineBackup": Column(pa.String, checks=Check.isin(["Yes","No","No internet service"]), nullable=False),
        "DeviceProtection": Column(pa.String, checks=Check.isin(["Yes","No","No internet service"]), nullable=False),
        "TechSupport": Column(pa.String, checks=Check.isin(["Yes","No","No internet service"]), nullable=False),
        "StreamingTV": Column(pa.String, checks=Check.isin(["Yes","No","No internet service"]), nullable=False),
        "StreamingMovies": Column(pa.String, checks=Check.isin(["Yes","No","No internet service"]), nullable=False),
        "Contract": Column(pa.String, checks=Check.isin(["Month-to-month","One year","Two year"]), nullable=False),
        "PaperlessBilling": Column(pa.String, checks=Check.isin(["Yes","No"]), nullable=False),
        "PaymentMethod": Column(pa.String, checks=Check.isin([
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]), nullable=False),
        "MonthlyCharges": Column(pa.Float, checks=Check.ge(0), nullable=False),
        "TotalCharges": Column(pa.String, nullable=False),
        "Churn": Column(pa.String, checks=Check.isin(["Yes","No"]), nullable=False)
    })

    validated = schema.validate(df, lazy=True)
    # convert TotalCharges to numeric like notebook
    validated['TotalCharges'] = pd.to_numeric(validated['TotalCharges'], errors='coerce')
    return validated

def compute_checksums(data_dir: str):
    cs = {}
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            path = os.path.join(data_dir, fname)
            cs[fname] = md5_checksum(path)
    return cs

def stratified_split(df, target="Churn", test_size=0.2, random_state=RANDOM_SEED):
    X = df.drop(columns=[target])
    y = df[target].map({"Yes":1, "No":0})
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
