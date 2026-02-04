from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from utility.utility_functions import save_model
from utility.model_pipeline import pipe_line
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,roc_curve
from sklearn import set_config
set_config(transform_output="pandas")

# Train all the models using the dataset
def train_and_evaluate_model(X_train, X_test, y_train, y_test):

    
    y_train = y_train.values.ravel().astype('int')
    y_test = y_test.values.ravel().astype('int')
    
    models =[
        LogisticRegression(max_iter=1000),
        RandomForestClassifier(n_estimators=100,max_depth=4,random_state=24),
             XGBClassifier(max_depth=3,min_child_weight=7,gamma=0.2,subsample=0.6,colsample_bytree=0.6,learning_rate=0.05,n_estimators=100,reg_alpha=0.5,reg_lambda=2,scale_pos_weight=1,random_state=24),
             KNeighborsClassifier(n_jobs=-1)]

    model_names = ["LogisticR","RandomForest","XGBoost","KNN"]
    for model,model_name in zip(models,model_names):
        # Step 1: Full pipeline with Feature engineering, SMOTE and classifier
        pipeline = pipe_line(X_train,model,model_name)
        
        # Step 2: Fit the model
        pipeline.fit(X_train, y_train)

        # Obtain predicted values
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        y_pred_prob_test = pipeline.predict_proba(X_test)[:, 1]
        y_pred_prob_train = pipeline.predict_proba(X_train)[:, 1]
    
        # Step 4: Obtain performance metrics of the model 
        accuracy_test = accuracy_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test)
        recall_test = recall_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)
        roc_auc_test = roc_auc_score(y_test, y_pred_prob_test)
    
        print(f"Model: {model_name}")
        print("Test dataset performance")
        print(f"Accuracy: {accuracy_test:.2f}")
        print(f"Precision: {precision_test:.2f}")
        print(f"Recall: {recall_test:.2f}")
        print(f"F1 Score: {f1_test:.2f}")
        print(f"ROC AUC: {roc_auc_test:.2f}")
        print("\n")
        print("Training dataset performance")
        accuracy_train = accuracy_score(y_train, y_pred_train)
        precision_train = precision_score(y_train, y_pred_train)
        recall_train = recall_score(y_train, y_pred_train)
        f1_train = f1_score(y_train, y_pred_train)
        roc_auc_train = roc_auc_score(y_train, y_pred_prob_train)
    
        print(f"Accuracy: {accuracy_train:.2f}")
        print(f"Precision: {precision_train:.2f}")
        print(f"Recall: {recall_train:.2f}")
        print(f"F1 Score: {f1_train:.2f}")
        print(f"ROC AUC: {roc_auc_train:.2f}")
        print("\n")
    
        #Save the pipeline after trainning
        save_model(pipeline,model_name)