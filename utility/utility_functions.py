import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from fairlearn.metrics import (MetricFrame,selection_rate,demographic_parity_difference,demographic_parity_ratio,
    equalized_odds_difference,true_positive_rate)
from sklearn.metrics import (roc_auc_score,roc_curve,confusion_matrix,classification_report,
                             accuracy_score, precision_score, recall_score, f1_score)

def model_report(X_test, y_test,model,model_name):

    # load the model and predict
    y_pred = model.predict(X_test)

    # ROC Curve AUC
    roc_auc = plot_roc_curve(model, X_test, y_test, "CV "+model_name)

    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, "CV "+model_name)

    # Classification Report
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Model Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1],
        'roc_auc': [roc_auc]
    }
    path ="./report"
    print(f"{model_name} Model Performance Report")
    print(metrics)
    print()

    os.makedirs(path,exist_ok=True)
    pd.DataFrame(metrics).to_csv(os.path.join(path, f"{model_name}_performance_report.csv"),index=False)
    pd.DataFrame(class_report).to_csv(os.path.join(path, f"{model_name}_classification_report.csv"))
    
    print(f"Kindly check '{path}' for the reports")



def reweight(y_train, gender_train):

    y_train = np.asarray(y_train).ravel()
    gender_train = np.asarray(gender_train).ravel()

    df = pd.DataFrame({
        "y": y_train,
        "gender": gender_train
    })

    group_counts = df.groupby(["gender", "y"]).size()
    
    total = len(df)

    weights_dict = {}

    for (g, y_val), count in group_counts.items():
        weights_dict[(g, y_val)] = total / count
    
    weights = [
        weights_dict[(g, y_val)]
        for g, y_val in zip(gender_train, y_train)
    ]

    return weights

# custom function for loading dataset
def load_data(dataset_path): 
    return pd.read_csv(dataset_path,usecols=lambda col: col != "Unnamed: 0")

# Load model and test dataset
def load_model_and_data(model_path, X_test_path, y_test_path):

    X_test = load_data(X_test_path)
    y_test = load_data(y_test_path).values.ravel().astype('int')
    
    return load_model(model_path), X_test, y_test

def cormat(df):

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.difference(['telephone'])
    cmat = df[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(cmat, 
                 cbar=True, 
                 annot=True, 
                 square=True, 
                 fmt='.2f', 
                 annot_kws={'size': 10}, 
                 yticklabels=df[numeric_columns].columns, 
                 xticklabels=df[numeric_columns].columns, 
                 cmap="Spectral_r")
    plt.title(f'Correlation Plot')
    os.makedirs('plots/eda', exist_ok=True)
    plt.savefig('Correlation_plot.png')
    plt.show()
    plt.close()

def preprocess_data(df, target_column, output_dir,test_size):

    # protected features - to be removed
    protected = ['gender']
    
    y = df[target_column]
    X=df.drop([target_column,'dependents','telephone','foreign_worker'],axis=1)
    
    #Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = test_size)

    # Extract the protected features
    X_train_protected = X_train[protected]
    X_test_protected = X_test[protected]

    # Extract the model important features
    X_train = X_train.drop(protected,axis=1)
    X_test = X_test.drop(protected,axis=1)
    
    # Save all the datasets
    save_datasets(X_train, X_test, y_train, y_test, X_train_protected,X_test_protected, output_dir)

# Save the model
def save_model(model, model_name):
    os.makedirs('./model', exist_ok=True)
    joblib.dump(model, f"./model/{model_name}.pkl")

# Load the model
def load_model(model_path):
    return joblib.load(model_path)

# Function to plot ROC
def plot_roc_curve(model, X_test, y_test, model_name):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2%})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {model_name}')
    plt.legend(loc="best")
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{model_name}_roc_curve.png')
    plt.show()
    plt.close()

    return roc_auc
    
# Confusion Matrix function
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred,normalize = 'true')
    decision_label = ['Good Credit', 'Bad Credit']
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='.0%', xticklabels=decision_label,
                yticklabels=decision_label)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')
    plt.show()
    plt.close()

# This shows fairness metrics for a protected variable (e.g gender)
def fairness_check(y_test,y_pred,sensitive):
    # Wrap metrics into MetricFrame
    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "tpr": true_positive_rate  # recall for positives
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive
    )
    
    print("Performance by group:\n", mf.by_group)
    
    # Are selection rates equal across the categories?
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive)
    dp_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=sensitive)
    
    # Do these categories have equal chance of being correctly classified?
    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive)
    
    print("Demographic parity difference:", dp_diff)
    print("Demographic parity ratio:", dp_ratio)
    print("Equalized odds difference:", eo_diff)

# Save the individual dataset
def save_datasets(X_train, X_test, y_train, y_test, train_protected,test_protected,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    train_protected.to_csv(os.path.join(output_dir, 'protected_train.csv'), index=False)
    test_protected.to_csv(os.path.join(output_dir, 'protected_test.csv'), index=False)
    print(f"Datasets saved to {output_dir}")
    
def barplot(target, df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.difference(['telephone','default'])
    ncols = len(numeric_columns)

    r = 6
    c = (ncols + r - 1) // r  

    f, axes = plt.subplots(r, c, figsize=(20, 18))  # removed sharex
    axes = axes.flatten()

    for i, col in enumerate(numeric_columns):
        sns.barplot(x=target, y=col, data=df, ax=axes[i])
        axes[i].set_title(f'{col} by {target}')
        axes[i].set_xlabel(target)
        axes[i].tick_params(axis="x", rotation=45)

    for i in range(len(numeric_columns), len(axes)):
        axes[i].axis('off')

    os.makedirs('plots/eda', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'plots/eda/{target}_Barplot.png')
    plt.show()
    plt.close()