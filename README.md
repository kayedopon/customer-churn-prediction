# Customer Churn Prediction with MLflow, FastAPI and Docker

## Project Overview

This project implements an end-to-end machine learning workflow for
customer churn prediction. The goal was to practice the complete ML
lifecycle including experiment tracking, model management, API
deployment, and containerization.

Features: - Exploratory Data Analysis (EDA) - Hyperparameter
optimization with RandomizedSearchCV - Logistic Regression, Random
Forest, XGBoost and CatBoost comparison - Threshold tuning - MLflow
experiment tracking and model registry - FastAPI inference API - Docker
deployment

## MLflow

Tracked: - Hyperparameters - F1, Precision, Recall, ROC-AUC, PR-AUC -
Confusion Matrix - ROC and PR curves - Classification reports - Trained
models

## Key Findings

Best F1-score achieved: ~0.42

Models produced very similar performance, suggesting that the primary limitation was feature quality and dataset signal rather than algorithm choice.

SHAP analysis indicated that MonthlyCharges, Tenure, and TotalCharges 

were the most influential features. However, their overall impact remained relatively small, suggesting that the dataset contains limited predictive signal. This likely explains why different models achieved nearly identical performance and why improvements through model complexity were minimal.

## Run API

``` bash
uvicorn src.main:app --reload
```

Swagger:

http://127.0.0.1:8000/docs

Example request:

``` json
{
  "gender":"Male",
  "senior_citizen":0,
  "tenure":24,
  "monthly_charges":70.5,
  "total_charges":1500,
  "contract":"One year",
  "payment_method":"Credit card"
}
```

## Docker

``` bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```
