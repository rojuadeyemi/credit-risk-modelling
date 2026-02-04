from sklearn.metrics import  classification_report
from utility.utility_functions import plot_confusion_matrix,plot_roc_curve,load_model

# Model performance Evaluation
def evaluate_model(X_test, y_test):
    model_paths = ["./model/LogisticR.pkl", "./model/RandomForest.pkl", "./model/XGBoost.pkl","./model/KNN.pkl"]
    model_names = ["Logistic Regression", "Random Forest", "XGBoost","KNN"]
    for model_path, model_name in zip(model_paths, model_names):
        model = load_model(model_path)
        y_pred = model.predict(X_test)
    
        # ROC Curve
        plot_roc_curve(model, X_test, y_test, model_name)
    
        # Confusion Matrix
        plot_confusion_matrix(y_test, y_pred, model_name)
    
        # Classification Report
        print(f"Classification Report for {model_name}:\n")
        print(classification_report(y_test, y_pred))



