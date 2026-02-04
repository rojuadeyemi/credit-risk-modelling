from sklearn.metrics import classification_report,accuracy_score, precision_score, recall_score, f1_score
import os
from utility.utility_functions import plot_roc_curve, plot_confusion_matrix
from utility.model_evaluation import load_model
import pandas as pd

def best_model_report(X_test, y_test):

    # find best model name and path
    model_name,model_path = find_best()

    # load the model and predict
    model = load_model(model_path)
    y_pred = model.predict(X_test)

    # ROC Curve AUC
    roc_auc = plot_roc_curve(model, X_test, y_test, "Best "+model_name)

    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, "Best "+model_name)

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
    print("Best Model Performance Report")
    print(metrics)

    os.makedirs(path,exist_ok=True)
    pd.DataFrame(metrics).to_csv(os.path.join(path, "model_performance_report.csv"),index=False)
    pd.DataFrame(class_report).to_csv(os.path.join(path, "model_classification_report.csv"))
    
    print(f"Kindly check '{path}' for the reports")

# This function finds best model from the model folder
def find_best():

    # List all files in the model path
    files = os.listdir("./model")
    
    # Filter file that start with the specified "cv_"
    model_file = [f for f in files if f.startswith("cv_")]
    
    # return model path, and its name
    
    return model_file[0].split(".")[0], os.path.join("./model", model_file[0])
