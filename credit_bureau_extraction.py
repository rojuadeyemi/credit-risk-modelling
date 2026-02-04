from typing import List, Dict
from datetime import datetime
import pandas as pd
import numpy as np
from collections import Counter
import json


def extract_credit_features(reports: List[Dict]) -> pd.DataFrame:
    features = []
    now = pd.Timestamp.now()

    for report in reports:
        customer_data = report.get("data", {})
        record = next(iter(customer_data.values()), {}) 
        feat = {"application_id": report.get("application_id")}

        # --- Product mix ---
        rating_data = record.get('accountrating', {})
        feat["num_good_accounts"] = sum(int(v) for k,v in rating_data.items() if k.endswith("good"))
        feat["num_bad_accounts"] = sum(int(v) for k,v in rating_data.items() if k.endswith("bad"))

        # --- Account performance ---
        credit_data = record.get('creditaccountsummary', {})
        feat["bureau_rating"] = credit_data.get("rating")
        feat["total_accounts"] = sum(int(v) for k,v in credit_data.items() if k.startswith("totalaccounts"))
        feat["total_arrears"] = sum(to_float_safe(v) for k,v in credit_data.items() if k.startswith("amountarrear"))
        feat["account_in_arrears"] = sum(int(v) for k,v in credit_data.items() if k.startswith("totalaccountarrear"))
        feat["total_outstanding_debt"] = sum(to_float_safe(v) for k,v in credit_data.items() if k.startswith("totaloutstandingdebt"))
        feat["total_monthly_instalment"] = sum(to_float_safe(v) for k,v in credit_data.items() if k.startswith("totalmonthlyinstalment"))
        feat["dishonoured_count"] = sum(int(v) for k,v in credit_data.items() if k.startswith("totalnumberofdishonoured"))

        feat["total_month_past_due"] = int(record.get('deliquencyinformation', {}).get('monthsinarrears', 0))
        feat["provided_guarantor"] = "yes" if int(record.get('guarantorcount', {}).get('accounts', 0)) > 0 else "no"
        
        # --- Credit agreements ---
        agreements_data = record.get('creditagreementsummary', [])
        feat["total_active_loans"] = sum(1 for a in agreements_data if a.get("accountstatus") and a.get("accountstatus").lower() == "open")
        feat["total_written_off_loans"] = sum(1 for a in agreements_data if a.get("accountstatus") and a.get("accountstatus").lower() == "writtenoff")
        feat["total_closed_loans"] = sum(1 for a in agreements_data if a.get("accountstatus") and a.get("accountstatus").lower() == "closed")
        feat["total_NPL"] = sum(1 for a in agreements_data if a.get("performancestatus").lower() == "lost")
        feat['no_of_instituitions'] = len(set(x.get("subscribername") for x in agreements_data))
        feat["total_amount_overdue"] = sum(to_float_safe(a.get("amountoverdue", 0)) for a in agreements_data)
        feat["total_loan"] = sum(to_float_safe(a.get("openingbalanceamt", 0)) for a in agreements_data)
        feat["avg_account_balance"] =sum([to_float_safe(a.get("currentbalanceamt", 0)) for a in agreements_data])
        feat['avg_account_age'] = np.mean([(datetime.strptime(e_data.get('closeddate'), "%d/%m/%Y") - datetime.strptime(e_data.get('dateaccountopened'), "%d/%m/%Y")).days for e_data in agreements_data if e_data.get('closeddate')]) // 30  # average age in months
        
        durations = [int(a.get("loanduration", 0)) for a in agreements_data if a.get("loanduration")]
        
        feat["avg_loan_duration"] = np.mean(durations) if durations else 0
        
        # --- Enquiry history ---
        enquiries = record.get('enquiryhistorytop', [])
        last_6_months = now - pd.DateOffset(months=6)
        feat['num_enquiries_last_6m'] = sum(
            1 for e in enquiries if datetime.strptime(e.get('daterequested', '01/01/1900 00:00:00'), "%d/%m/%Y %H:%M:%S") >= last_6_months
        )
        # # --- Payment Record ---

        all_payment_ratios = []
        total_missed = 0

        for acc in record['accountmonthlypaymenthistory']:
            months = [v for k,v in acc.items() if k.startswith('m')]
            paid = sum(1 for m in months if m == '#')
            total = len(months)
            all_payment_ratios.append(paid/total if total > 0 else 0)
            total_missed += sum(1 for m in months if m == '0')

        feat['avg_payment_ratio'] = np.mean(all_payment_ratios)
        feat['total_missed_payment'] = total_missed
        feat['worst_payment_ratio'] = min(all_payment_ratios)

        # --- Employment & personal details ---
        personal_data = record.get("personaldetailssummary", {})
        feat["gender"] = personal_data.get("gender")
        feat["dependants"] = int(personal_data.get("dependants", 0))
        feat["property_type"] = personal_data.get("propertyownedtype", "")

        occupations = [emp.get("occupation", "").strip().upper() for emp in record.get('employmenthistory', []) if emp.get("occupation")]
        feat['occupation'] = Counter(occupations).most_common(1)[0][0] if occupations else None

        birthday = personal_data.get("birthdate")
        feat['age'] = (now - pd.to_datetime(birthday, dayfirst=True)).days // 365 if birthday else np.nan

        # Derived ratio
        feat["instalment_to_debt_ratio"] = feat["total_monthly_instalment"] / feat["total_outstanding_debt"] if feat["total_outstanding_debt"]>0 else 0
        feat["overdue_proportion"] = feat["total_amount_overdue"] / feat["total_loan"] if feat["total_loan"]>0 else 0

        features.append(feat)

    return pd.DataFrame(features)

# A helper function for cleaning data point values
def to_float_safe(x):
    if x is None:
        return 0
    if isinstance(x, (int, float)):
        return x
    return float(x.replace(',', ''))



if __name__=="__main__":

    # Enter the source path to the credit bureau report
    file_path = r"Others\Credit_bureau_sample_data.json"

    # This opens the report
    with open(file_path,"r") as file:
        report = json.load(file)

    # Extract the key data points, and return them in a dataframe
    df = extract_credit_features(report)
    
    # Print top 2 rows of the dataframe
    print(df.head(2))
