from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# Feature Engineer Wrapper Class
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = feature_engineering(X)
        self.feature_names_out_ = X_new.columns
        return X_new

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


# This function obtains all new features for the pipeline
def feature_engineering(df):
    
    df = df.copy()

    # 1. Add Missing Flags    
    for col in df.columns[df.isna().any()]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
        
    # 2. Ratios & Affordability
    # loan-to-balance
    ratio = df["amount"] / (
    df[["savings_balance", "checking_balance"]].fillna(0).sum(axis=1) + 1)

    df["loan_to_balance"] = np.clip(ratio, 0, 100)  # cap at 100

    # Refine the residence history and employment length
    df["residence_history"] = df["residence_history"].apply(convert_to_year)
    df["employment_length"] = df["employment_length"].apply(convert_to_year)

    # 4. Credit History Severity Score
    severity_map = {
        "critical": 0,
        "delayed": 1,
        "repaid": 5,
        "fully repaid": 8,
        "fully repaid this bank": 15}
    df["credit_history"] = df["credit_history"].map(severity_map)

    # 5. Interaction Feature
    #df["employment_x_amount"] = df["employment_length"].fillna(0) * df["amount"]

    return df

# a helper function to convert residence_history and employment_length to number of months
def convert_to_year(text):
    if pd.isna(text):
        return None
    text = text.lower()
    multiplier = 12 if "year" in text else 1
    return int(text.split()[0]) * multiplier//12