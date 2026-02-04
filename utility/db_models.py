from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime, timezone
from utility.database import Base

class Log(Base):
    # Table name
    __tablename__ = "credit_tbl"

    # Auto generated fields
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Model Input fields
    checking_balance = Column(Float)
    months_loan_duration = Column(Float)
    credit_history = Column(String)
    purpose = Column(String)
    amount = Column(Float)
    savings_balance = Column(Float)
    employment_length = Column(String)
    installment_rate = Column(Float)
    personal_status = Column(String)
    other_debtors = Column(String)
    residence_history = Column(String)
    property = Column(String)
    age = Column(Integer)
    installment_plan = Column(String)
    housing = Column(String)
    existing_credits = Column(Integer)
    job = Column(String)

    # Model Output fields
    default_probability = Column(Float)
    status = Column(String)
