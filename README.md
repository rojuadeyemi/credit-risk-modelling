# Credit Risk Modelling

This project provides a **FastAPI-based credit risk scoring API** that receives applicant data, generates predictions, and stores both inputs and outputs in an **SQLite database** for further analysis and auditing purposes.

---
## Key Components of the Project:

#### 1. Data Collection and Preprocessing
* Data Source: The data was sourced from [UCI](https://archive.ics.uci.edu/), which included various features relevant to credit risk prediction.

#### 2. Model Training and Evaluation:
* Data Cleaning: The data underwent extensive cleaning to handle missing values, outliers, and data inconsistencies.
* Feature Engineering: New features such as `Loan to Balance ratio, Age Bucket, Credit History Severity, Missing flags` were created from the existing ones to improve model performance.
* Automated feature selection during training and inferential, to remove redundant features.
* Hyperparameter Tuning: K-Fold Cross-validation and Random Search were used to fine-tune the models for optimal performance.
* Algorithms Tested: Logistic Regression, Random Forests, XGBoost.
* Performance Metrics: Accuracy, precision, recall, F1-score, and AUC-ROC.
* Fairness Check: Here we carried out fairness check for gender to ascertain model fairness irrespective of gender category.

#### 3. Model Deployment:
* Best Model Selection: `Random Forest` demonstrated the best performance in predicting credit risk, and in fairness irrespective of the gender group, and hence was selected for deployment.
* Real-Time Prediction: The selected model was integrated into a FlaskAPI, enabling real-time prediction of defaulting status based on user inputs.

#### 4. Database Integration:
* Data Storage: User specific inputs and prediction results are stored in a SQLite database for future reference and analysis.

### Technologies Used:
* Tools: Python
* Frameworks and Libraries: FastAPI, Scikit-learn, Pandas, NumPy, SQLAlchemy
* Database: SQLite
* Version Control: Git

**Model Development**: Check the [model development](model%20development.ipynb) for detailed data analysis and modelling steps.

## Model Deployment

### Prerequisites

- Python 3.10-3.12
- Pip (Python package installer)


## 🚀 Project Structure

- 📂 **utility/** : package folder containing helper functions (e.g., preprocessing, feature engineering,model_training, database schema etc).
- 📂 **models/** : folder containing the pre-trained ML models.
- 📄 **requirements.txt** : list of all required dependencies.
- 📄 **app.py** : main FastAPI application file.

---

## 🔧 Setup Instructions

1. **Clone the repository:**

    ```sh
    git clone https://github.com/rojuadeyemi/diabetes-test-app.git
    cd diabetes-test-app
    ```

2. **Create a Virtual Environment**

For **Linux/Mac**:

```sh
python -m venv .venv
source .venv/bin/activate
```

For **Windows**:

```sh
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies

Once the environment is active, install dependencies:

```sh
python.exe -m pip install -U pip
pip install -r requirements.txt
```

---

## ▶️ Launch the Uvicorn Server

Launch the server with:

```sh
uvicorn app:app
```

- By default, the API will be available at: **http://localhost:8000**

---

## 📌 Available Endpoints

### 🔹 1. Prediction Endpoint

**POST** `http://localhost:8000/predict`

This endpoint takes borrower/application data as input and returns:

- **status** → model’s classification (e.g., *Bad* or *Good*).
- **default_probability** → likelihood of default.

**Sample Payload**:

```json
{
    "checking_balance": null,
    "months_loan_duration": 1,
    "credit_history": "critical",
    "purpose": "car (new)",
    "amount": 250,
    "savings_balance": 6002,
    "employment_length": "2 years",
    "installment_rate": 2,
    "personal_status": null,
    "other_debtors": "none",
    "residence_history": "4 months",
    "property": "real estate",
    "age": 41,
    "installment_plan": "bank",
    "housing": "own",
    "existing_credits": 2,
    "job": "unskilled resident"
}
```

**Example Response**:

```json
{
    "status": "Bad",
    "default_probability": 0.72
}
```

---

### 🔹 2. Logs Endpoint

**GET** → `http://localhost:8000/logs`

This endpoint retrieves stored records from the SQLite database, allowing you to review past predictions along with their inputs.

**Example Response (truncated)**:

```json
[
  {
    "application_id": 1,
    "checking_balance": null,
    "months_loan_duration": 6,
    "status": "default",
    "probability": 0.72,
    "timestamp": "2025-09-08T14:32:11"
  }
]
```

---

## 🖼️ System Architecture

Below is the flow of data through the system:

```
Client (Postman, SwaggerUI, etc.)
        │
        ▼
   FastAPI App (app.py)
        │
        ├──> ML Model (models/)
        │
        └──> SQLite Database (logs requests & predictions)
```

This ensures both **real-time predictions** and **persistent storage** for monitoring and auditing.

---

With these steps, you can successfully **deploy, test, and query** the credit risk scoring API.


### Alternatively - Predicting using Native Python

If you prefer to use Python, use the code below to pass the inputs and load the model

```
import joblib
import pandas as pd

# Enter the input data here
input_data = {
    "checking_balance": None,
    "months_loan_duration": 6,
    "credit_history": "critical",
    "purpose": "car (new)",
    "amount": 250,
    "savings_balance": 6002,
    "employment_length": "2 years",
    "installment_rate": 2,
    "personal_status": None,
    "other_debtors": "none",
    "residence_history": "4 months",
    "property": "real estate",
    "age": 41,
    "installment_plan": "bank",
    "housing": "own",
    "existing_credits": 2,
    "job": "unskilled resident"
}

# Convert to DataFrame
df = pd.DataFrame([input_data])

# Load your trained model from the model path
model_path = r"models\cv_Random Forest.pkl"
joblib.load(model_path)

# Make prediction
prediction = model.predict(df)               # class label
probability = model.predict_proba(df)[0][1]  # probability of default

print("Prediction:", prediction[0])
print("Default probability:", probability)
```