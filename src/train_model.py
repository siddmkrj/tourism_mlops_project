import os
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from huggingface_hub import HfApi, hf_hub_download

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

TARGET_COL = "ProdTaken"


def download_splits_from_hf(dataset_repo_id: str):
    train_local_path = hf_hub_download(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        filename="data/train.csv",
    )
    test_local_path = hf_hub_download(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        filename="data/test.csv",
    )

    train_df = pd.read_csv(train_local_path)
    test_df = pd.read_csv(test_local_path)

    print(f"Downloaded train to {train_local_path}, shape={train_df.shape}")
    print(f"Downloaded test to {test_local_path}, shape={test_df.shape}")
    
    return train_df, test_df

def build_and_train_model(train_df: pd.DataFrame):
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ]
    )

    model = XGBClassifier(
        subsample=0.8,
        max_depth=7,
        learning_rate=0.05,
        colsample_bytree=1.0,
        n_estimators=300,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("Training XGBoost model...")
    clf.fit(X_train, y_train)
    print("✅ Training completed.")
    return clf


def evaluate_model(model, test_df: pd.DataFrame):
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return metrics


def extract_model_params(model):
    xgb_model = model.named_steps["model"]
    params = {
        "subsample": xgb_model.subsample,
        "max_depth": xgb_model.max_depth,
        "learning_rate": xgb_model.learning_rate,
        "colsample_bytree": xgb_model.colsample_bytree,
        "n_estimators": xgb_model.n_estimators,
        "eval_metric": xgb_model.eval_metric,
        "random_state": xgb_model.random_state,
        "tree_method": xgb_model.tree_method,
    }
    return params


def log_to_mlflow(model, params, metrics, run_name="XGBoost_Best"):
    mlflow.set_experiment("tourism_wellness_modeling")
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(model, artifact_path="xgb_model")
        print(f"✅ Logged to MLflow with run name: {run_name}")


def save_and_push_model(model, model_repo_id: str, token: str):
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "best_model.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    api = HfApi(token=token)

    api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_model.pkl",
        repo_id=model_repo_id,
        repo_type="model",
    )

    print(f"✅ Model uploaded to HF model hub: {model_repo_id}")


def main():
    dataset_repo_id = os.getenv(
        "HF_DATASET_REPO_ID", "mukherjee78/tourism-wellness-package"
    )
    model_repo_id = os.getenv(
        "HF_MODEL_REPO_ID", "mukherjee78/tourism-wellness-best-model"
    )
    token = os.getenv("HF_TOKEN")

    if token is None:
        raise ValueError(
            "HF_TOKEN environment variable is not set. "
            "Please set it in your environment or GitHub Actions secrets."
        )

    print(f"Downloading train/test from dataset repo: {dataset_repo_id}")
    train_df, test_df = download_splits_from_hf(dataset_repo_id)

    model = build_and_train_model(train_df)
    metrics = evaluate_model(model, test_df)

    params = extract_model_params(model)
    log_to_mlflow(model, params, metrics)

    print("Saving and pushing model to HF Model Hub...")
    save_and_push_model(model, model_repo_id, token)


if __name__ == "__main__":
    main()