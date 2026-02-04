from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from utility.utility_functions import save_model
from utility.model_pipeline import pipe_line
from sklearn import set_config
set_config(transform_output="pandas")
import pandas as pd

from scipy.stats import randint, uniform, loguniform


# Train all the models and cross-validate, then save the best model
def model_val(X_train, y_train, cv, scoring):

    y_train = y_train.values.ravel().astype(int)

    models = {
        "Logistic Regression": pipe_line(X_train, LogisticRegression(max_iter=1000), "LogisticR"),
        "Random Forest": pipe_line(X_train, RandomForestClassifier(random_state=24), "RandomForest"),
        "XGBoost": pipe_line(X_train, XGBClassifier(random_state=24,eval_metric="logloss",tree_method="hist"), "XGBoost"),
        "KNN": pipe_line(X_train, KNeighborsClassifier(), "KNN")
    }

    # Optimized parameter grids for RandomSearchCV
    param_grids = {
        "Logistic Regression": {
            'classifier__C': loguniform(0.001, 10),
            'classifier__class_weight': [None, 'balanced']
        },
        "Random Forest": {
            'classifier__n_estimators': randint(80, 250),
            'classifier__max_depth': randint(5, 8),
            'classifier__min_samples_leaf': randint(1, 10),
            'classifier__class_weight': [None, 'balanced'],
            'classifier__max_features': ['sqrt', 'log2'],
        },
        "XGBoost": {
            'classifier__n_estimators': randint(80, 250),
            'classifier__max_depth': randint(5, 8),
            'classifier__learning_rate': loguniform(0.01, 0.3),
            'classifier__subsample': uniform(0.7, 0.3),
            'classifier__colsample_bytree': uniform(0.7, 0.3),
            'classifier__min_child_weight': randint(1, 5),
            'classifier__gamma': uniform(0, 0.3)
        },
        "KNN": {
            'classifier__n_neighbors': randint(3, 15),
            'classifier__weights': ['uniform', 'distance'],
            'classifier__p': [1, 2]
        }
    }

    results_log = []

    best_score = -1
    best_model = None
    best_name = None

    for name, pipeline in models.items():

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grids[name],
            n_iter=50,
            cv=cv,
            scoring=scoring,
            refit="f1",
            random_state=24,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        search.fit(X_train, y_train)

        cv_res = search.cv_results_

        row = {
            "model": name,
            "roc_auc": search.best_score_,
            "recall": cv_res["mean_test_recall"][search.best_index_],
            "f1": cv_res["mean_test_f1"][search.best_index_],
            "balanced_acc": cv_res["mean_test_balanced_acc"][search.best_index_]
        }

        results_log.append(row)

        print(f"{name}: {search.best_score_}")

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search.best_estimator_
            best_name = name

    results_df = pd.DataFrame(results_log).sort_values("f1", ascending=False)

    results_df.to_csv("./report/model_comparison.csv", index=False)

    save_model(best_model, f"cv_{best_name}")
    print(f"\nBest Model: {best_name} | Score: {best_score}")

