import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter


def get_logres():
    logres = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", random_state=42))
    ])

    params = {
        "model__max_iter": [100, 200, 500, 1000],
        "model__C": [0.01, 0.1, 1, 10]
    }

    log_res_search = RandomizedSearchCV(
        estimator=logres,
        param_distributions=params,
        cv=5,
        scoring="f1",
        n_iter=10,
        random_state=42
    )
    return log_res_search

def get_forest():
    forest = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))
    ])

    params = {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__max_depth": [None, 10, 20],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2],
        'model__bootstrap': [True, False]
    }

    forest_search = RandomizedSearchCV(
        estimator=forest,
        param_distributions=params,
        cv=5,
        scoring="f1",
        n_iter=20,
        random_state=42
    )

    return forest_search

def get_xgbc(y):
    counter = Counter(y)
    estimate = counter[0] / counter[1]

    xgbc = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(scale_pos_weight=estimate,objective="binary:logistic", 
                                eval_metric="logloss",random_state=42))
    ])

    params = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__gamma": [0, 0.1, 0.3],
        "model__reg_lambda": [1, 5, 10]
    }

    xgbc_search = RandomizedSearchCV(
        estimator=xgbc,
        param_distributions=params,
        cv=5,
        scoring="f1",
        n_iter=30,
        random_state=42
    )

    return xgbc_search

def get_cat():
    catb = Pipeline([
        ("model", CatBoostClassifier(
            class_weights=(1, 3),
            random_state=42,
            verbose=0
        ))
    ])

    params = {
        "model__iterations": [200, 500],
        "model__depth": [4, 6, 8],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__l2_leaf_reg": [1, 3, 5]
    }

    catb_search = RandomizedSearchCV(
        estimator=catb,
        param_distributions=params,
        n_iter=20,
        cv=5,
        scoring="f1",
        random_state=42,
    )

    return catb_search

def document(model, exp_name, t, X_train, y_train, X_test, y_test):  
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    with mlflow.start_run(run_name=exp_name):
        model.fit(X_train, y_train)
        best_model = model.best_estimator_

        y_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= t).astype(int)
        
        # params
        mlflow.log_param("model_name", exp_name)
        mlflow.log_params(model.best_params_)
        mlflow.log_param("threshold", t)

        # metrics
        mlflow.log_metric("best_cv_f1", model.best_score_)
        mlflow.log_metric("test_balanced_acc", balanced_accuracy_score(y_test, y_pred))
        mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
        mlflow.log_metric("test_recall", recall_score(y_test, y_pred))
        mlflow.log_metric("test_f1", f1_score(y_test, y_pred))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_proba))
        mlflow.log_metric("test_pr_auc", average_precision_score(y_test, y_proba))

        # artifact: confusion matrix
        cm_path = artifacts_dir / f"{exp_name}_confusion_matrix.png"
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.title(f"{exp_name} - Confusion Matrix")
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # artifact: ROC curve
        roc_path = artifacts_dir / f"{exp_name}_roc_curve.png"
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"{exp_name} - ROC Curve")
        plt.savefig(roc_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(roc_path), artifact_path="plots")

        # artifact: Precision-Recall curve
        pr_path = artifacts_dir / f"{exp_name}_pr_curve.png"
        PrecisionRecallDisplay.from_predictions(y_test, y_proba)
        plt.title(f"{exp_name} - Precision-Recall Curve")
        plt.savefig(pr_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(pr_path), artifact_path="plots")

        # artifact: classification report
        report_path = artifacts_dir / f"{exp_name}_classification_report.txt"
        report = classification_report(y_test, y_pred, zero_division=0)
        with open(report_path, "w") as f:
            f.write(report)

        mlflow.log_artifact(str(report_path), artifact_path="reports")

        # model
        mlflow.sklearn.log_model(best_model, name="model")