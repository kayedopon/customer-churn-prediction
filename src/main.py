from fastapi import FastAPI

from src.schemas import CustomerData, PredictionResponse

import pandas as pd
import joblib


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn probability.",
    version="1.0.0"
)

model = joblib.load("models/xgboost_threshold_0.4.joblib")
categorical = ["Gender", "Contract", "PaymentMethod"]
training_columns = [
    "SeniorCitizen",
    "Tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Gender_Male",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Credit card",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check"
]

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: CustomerData):
    input_df = pd.DataFrame([request.model_dump()])
    input_df = input_df.rename(columns={
        "gender": "Gender",
        "senior_citizen": "SeniorCitizen",
        "tenure": "Tenure",
        "monthly_charges": "MonthlyCharges",
        "total_charges": "TotalCharges",
        "contract": "Contract",
        "payment_method": "PaymentMethod"
    })
    
    input_df = pd.get_dummies(input_df, columns=categorical)
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    proba = model.predict_proba(input_df)[:, 1][0]
    prediction = int(proba >= 0.4)

    return {
        "churn_probability": round(float(proba), 4),
        "prediction": prediction
    }