import os
import joblib
import pandas as pd

from huggingface_hub import HfApi, hf_hub_download

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

TARGET_COL = "ProdTaken"


def download_splits_from_hf(dataset_repo_id: str):
    """
    Download train and test splits from the Hugging Face dataset hub.
    """
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

    # Drop index column if present
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop(columns=['Unnamed: 0'])

    return train_df, test_df


def build_and_train_model(train_df: pd.DataFrame):
    """
    Build preprocessing + XGBoost model pipeline and train on the train split.
    """
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
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
    """
    Evaluate the trained model on the test split and print metrics.
    """
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


def save_and_push_model(model, model_repo_id: str, token: str):
    """
    Save the trained model locally and push both the model file and a simple
    model card (README) to the Hugging Face model hub.
    """
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "best_model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    api = HfApi(token=token)


    api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model.joblib",
        repo_id=model_repo_id,
        repo_type="model",
    )

    model_card_text = (
        "# Tourism Wellness Package Classifier (XGBoost)\n\n"
        "This model predicts whether a customer is likely to purchase the Wellness Tourism Package.\n\n"
        "## Inputs\n"
        "- Customer demographic and interaction features (e.g., Age, CityTier, NumberOfTrips, etc.).\n\n"
        "## Target\n"
        "- `ProdTaken` (0 = No purchase, 1 = Purchase).\n\n"
        "## Training\n"
        "- Trained using an XGBoost (XGBClassifier) model inside a scikit-learn pipeline.\n"
        "- Preprocessing: median imputation for numeric features, most-frequent imputation and one-hot encoding for categorical features.\n"
    )

    model_card_path = os.path.join("models", "README.md")
    with open(model_card_path, "w", encoding="utf-8") as f:
        f.write(model_card_text)

    api.upload_file(
        path_or_fileobj=model_card_path,
        path_in_repo="README.md",
        repo_id=model_repo_id,
        repo_type="model",
    )

    print(f"✅ Model and model card uploaded to HF model hub: {model_repo_id}")


def main():
    dataset_repo_id = os.getenv(
        "HF_DATASET_REPO_ID", "mukherjee78/tourism-wellness-package"
    )
    model_repo_id = os.getenv(
        "HF_MODEL_REPO_ID", "mukherjee78/tourism-wellness-model"
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
    evaluate_model(model, test_df)

    print("Saving and pushing model to HF Model Hub...")
    save_and_push_model(model, model_repo_id, token)


if __name__ == "__main__":
    main()