from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder,FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from utility.feature_engine import FeatureEngineer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
import numpy as np

def pipe_line(X_train, model,model_name):
    numeric_columns = X_train.select_dtypes(include=['float64','int64']).columns
    categorical_columns = X_train.select_dtypes(include=['object','category']).columns

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='first'))
    ])
    log_cols = [
                "amount",
                "savings_balance",
                "months_loan_duration"
            ]
    
    log_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                    ("scaler", RobustScaler())
                ])
    
    num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("scaler", RobustScaler())
])

    preprocessor = ColumnTransformer([
        ('log_num', log_pipeline, log_cols),
        ('num', num_pipeline, [c for c in numeric_columns.union(['loan_to_balance']) if c not in log_cols]),
        ('cat', categorical_transformer, categorical_columns)
    ])

    preprocessor.set_output(transform="pandas")
    # --- Feature selection depending on model ---
    if model_name in ["LogisticR","RandomForest","XGBoost"]:
        selector = SelectFromModel(model, threshold=0.005)
    elif model_name=="KNN":
        selector = SelectKBest(mutual_info_classif, k=40)
    else:
        raise ValueError(f"Feature selection not defined for {type(model)}")

    pipeline = ImbPipeline([
        ("feature_engineering", FeatureEngineer()),
        ('preprocessing', preprocessor),
        ("smote", SMOTE(random_state=24)),
        ("feature_selection", selector),
        ('classifier', model)
    ])
    return pipeline