import pandas as pd
import numpy as np
import pickle
from typing import List

_model = None   # private variable (hanya bisa diakses di predict.py)
EXPECTED_COLUMNS = ['claim_number','age_of_driver','gender',
                    'marital_status','safty_rating','annual_income',
                    'high_education_ind','address_change_ind','living_status',
                    'zip_code','claim_date','claim_day_of_week','accident_site',
                    'past_num_of_claims','witness_present_ind','liab_prct',
                    'channel','policy_report_filed_ind','claim_est_payout','age_of_vehicle',
                    'vehicle_category','vehicle_price','vehicle_color','vehicle_weight']

def validate_schema(df:pd.DataFrame):
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Invalid schema. Expected {EXPECTED_COLUMNS}")

# function lazy model    
def get_model():
    global _model
    if _model is None:
        _model = pickle.load(open('model/artifacts/model_v1.sav', 'rb'))
        return _model
    return _model

def get_prediction(data:List[dict], proba=False) -> dict:    # tanda -> = 'output'
    df = pd.DataFrame(data)
    validate_schema(df)
    model = get_model()
    return model.predict_proba(df)[:,1] if proba else model.predict(df)

