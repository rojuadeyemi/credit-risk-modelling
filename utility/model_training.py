from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utility.utility_functions import save_model,plot_confusion_matrix,plot_roc_curve
from utility.model_pipeline import pipe_line
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,classification_report
from sklearn import set_config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
set_config(transform_output="pandas")

# Train all the models using the dataset
def train_and_evaluate_model(X_train, X_test, y_train, y_test,cv):

    
    y_train = y_train.values.ravel().astype('int')
    y_test = y_test.values.ravel().astype('int')
    test_ = []
    train_ = []
    valid_ = []

    models =[
        LogisticRegression(max_iter=1000),
        RandomForestClassifier(n_estimators=100,max_depth=4,random_state=42),
             XGBClassifier(max_depth=4,n_estimators=100,random_state=42)]

    model_names = ["LogisticR","RandomForest","XGBoost"]
    for model,model_name in zip(models,model_names):

        # Step 1: Full pipeline with Feature engineering, SMOTE and classifier
        pipeline = pipe_line(X_train,model)

        # Cross-validation (baseline comparison)
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["roc_auc", "recall", "balanced_accuracy","f1"],
            return_train_score=True,
            n_jobs=-1
        )

        print(f"Model: {model_name}")
        print("Validation Dataset Performance")
        print(f"ROC AUC: {cv_results['test_roc_auc'].mean():.2f}")
        print(f"Recall: {cv_results['test_recall'].mean():.2f}")
        print(f"f1: {cv_results['test_f1'].mean():.2f}")
        print(f"Balanced Accuracy: {cv_results['test_balanced_accuracy'].mean():.2f}")
        print("\n")
        
        print("Training Dataset Performance")
        print(f"ROC AUC: {cv_results['train_roc_auc'].mean():.2f}")
        print(f"Recall: {cv_results['train_recall'].mean():.2f}")
        print(f"f1: {cv_results['train_f1'].mean():.2f}")
        print(f"Balanced Accuracy: {cv_results['train_balanced_accuracy'].mean():.2f}")
        print("\n")

        # Step 2: Fit the model
        pipeline.fit(X_train, y_train)

        # Obtain predicted values
        y_pred_test = pipeline.predict(X_test)
        y_pred_prob_test = pipeline.predict_proba(X_test)[:, 1]
    
        # Step 4: Obtain performance metrics of the model 
        accuracy_test = accuracy_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test)
        recall_test = recall_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)
        roc_auc_test = roc_auc_score(y_test, y_pred_prob_test)
    
        print("Test Dataset Performance")
        print(f"Accuracy: {accuracy_test:.2f}")
        print(f"Precision: {precision_test:.2f}")
        print(f"Recall: {recall_test:.2f}")
        print(f"F1 Score: {f1_test:.2f}")
        print(f"ROC AUC: {roc_auc_test:.2f}")
        print("\n")
    
        #Save the pipeline after trainning
        save_model(pipeline,model_name)
    
        # Plot ROC Curve
        plot_roc_curve(pipeline, X_test, y_test, model_name)
    
        # Plot Confusion Matrix
        plot_confusion_matrix(y_test, y_pred_test, model_name)
    
        # Classification Report
        print(f"Classification Report for {model_name}:\n")
        print(classification_report(y_test, y_pred_test))

        # Store the metrics
        valid_metric = cv_results['test_roc_auc'].mean()
        valid_.append(valid_metric)

        train_metric = cv_results['train_roc_auc'].mean()
        train_.append(train_metric)

        test_.append(roc_auc_test)

    # Put all metrics into a DataFrame ---
    data = {
        "Model": model_names,
        "Train": train_,
        "Validation": valid_,
        "Test": test_
    }

    df = pd.DataFrame(data)

    # Plot Recall
    plot_metrics(df, ["Train", "Validation", "Test"], "ROC AUC Comparison")

# Plot function
def plot_metrics(df, metrics, title):
    x = np.arange(len(df["Model"]))  # model indices
    width = 0.25  # width of bars

    fig, ax = plt.subplots(figsize=(12,6))

    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, df[metric], width, label=metric)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df["Model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_title(title)
    ax.legend()
    plt.show()
