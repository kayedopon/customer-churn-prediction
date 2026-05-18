import pandas as pd
import mlflow

from pathlib import Path

from sklearn.model_selection import train_test_split

from src.utils import get_logres, get_forest, get_xgbc, get_cat, document


def main():
    data = Path("../data/processed/data.csv") 
    df = pd.read_csv(data)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    thresholds = [0.2, 0.3, 0.4]

    mlflow.set_experiment("customer-churn-prediction_lower_threshold")

    for t in thresholds:
        models = [
            (get_logres(), "logistic_regression", t), 
            (get_forest(), "random_forest", t), 
            (get_xgbc(), "xgboost", t), 
            (get_cat(), "catboost", t)
        ]

        for search_model, run_name, threshold in models:
            document(search_model, f"{run_name}_threshold_{threshold}", threshold, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()