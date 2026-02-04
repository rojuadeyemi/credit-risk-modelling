from fastapi import FastAPI, HTTPException, Body, Depends, Request
from pydantic import BaseModel
from typing import Optional
from utility.performance_report import find_best
from utility.utility_functions import load_model
from sqlalchemy.orm import Session
from utility.db_models import Log
from utility.database import SessionLocal, engine, Base
from typing import List
from datetime import datetime
import pandas as pd

app = FastAPI()

Base.metadata.create_all(bind=engine)  # Auto create table

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Define the input model
class CustomerData(BaseModel):
    checking_balance : Optional[float] = None
    months_loan_duration:float
    credit_history : str
    purpose : str
    amount : float
    savings_balance : Optional[float] = None
    employment_length : str
    installment_rate : float
    personal_status: Optional[str] = None
    other_debtors : str
    residence_history : str
    property : str
    age : int
    installment_plan : str
    housing : str
    existing_credits : int
    job : str

# Define the output model, for logging
class LogResponse(BaseModel):
    id: int
    checking_balance : Optional[float] = None
    months_loan_duration:float
    credit_history : str
    purpose : str
    amount : float
    savings_balance : Optional[float] = None
    employment_length : str
    installment_rate : float
    personal_status: Optional[str] = None
    other_debtors : str
    residence_history : str
    property : str
    age : int
    installment_plan : str
    housing : str
    existing_credits : int
    job : str
    default_probability : float
    status : str 
    timestamp: datetime

    class Config:
        from_attributes = True

@app.get("/")
def root():
    return {"message": "Welcome to the Credit Evaluation API"}

@app.post("/predict")
def evaluate_customer(
    data: CustomerData = Body(...),
    db: Session = Depends(get_db)):

    
    # Initialize and load the best model
    _, model_path = find_best()
    model = load_model(model_path)

    # define input_data
    input_df = pd.DataFrame([data.model_dump()])

    input_df = input_df.apply(pd.to_numeric, errors="ignore")
    # Run the model
    default_probability = float(model.predict_proba(input_df)[0][1])

    status = "Bad" if default_probability > 0.45 else "Good"

    # Save to DB
    try:
        log = Log(
        **data.model_dump(),

        # Output fields
        status=status,
        default_probability=default_probability
)

        db.add(log)
        db.commit()
    except Exception as e:
        print(f"Failed to log request: {e}") 

    return {"status": status, "default_probability": default_probability}

# endpoint for getting the records from the DB (maximum of 50 records)
@app.get("/logs", response_model=List[LogResponse])
def get_logs(db: Session = Depends(get_db)):
    return db.query(Log).order_by(Log.timestamp.desc()).limit(50).all()

# Exception hanling endpoint
@app.exception_handler(HTTPException)
def handle_error(request, exc):
    return {"status": "error", "detail": exc.detail}
