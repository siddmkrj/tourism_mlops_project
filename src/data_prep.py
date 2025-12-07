import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

TARGET_COL = "ProdTaken"

def load_raw_from_hf(dataset_repo_id: str) -> pd.DataFrame:
    dataset = load_dataset(dataset_repo_id, data_files={"full": "data/tourism.csv"})
    df = dataset["full"].to_pandas()
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    cols_to_drop = ["CustomerID", "Unnamed: 0"]

    df_clean = df_clean.drop(columns=cols_to_drop)
    print(f"Dropped columns: {cols_to_drop}")

    # Drop duplicates
    before = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    after = df_clean.shape[0]
    print(f"Dropped {before - after} duplicate rows")

    # Impute missing values
    feature_cols = [c for c in df_clean.columns if c != TARGET_COL]
    numeric_cols = df_clean[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    df_imputed = df_clean.copy()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        df_imputed[numeric_cols] = num_imputer.fit_transform(df_imputed[numeric_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df_imputed[categorical_cols] = cat_imputer.fit_transform(df_imputed[categorical_cols])

    print("Remaining missing values after imputation:", df_imputed.isna().sum().sum())
    return df_imputed

def split_and_save(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test

    os.makedirs("data", exist_ok=True)
    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train to {train_path}, shape={train_df.shape}")
    print(f"Saved test to {test_path}, shape={test_df.shape}")

    return train_path, test_path

def upload_splits_to_hf(dataset_repo_id: str, token: str, train_path: str, test_path: str):
    api = HfApi(token=token)

    api.upload_file(
        path_or_fileobj=train_path,
        path_in_repo="data/train.csv",
        repo_id=dataset_repo_id,
        repo_type="dataset"
    )

    api.upload_file(
        path_or_fileobj=test_path,
        path_in_repo="data/test.csv",
        repo_id=dataset_repo_id,
        repo_type="dataset"
    )

    print("✅ Uploaded train.csv and test.csv to Hugging Face dataset.")

def main():
    dataset_repo_id = os.getenv("HF_DATASET_REPO_ID", "mukherjee78/tourism-wellness-package")
    token = os.getenv("HF_TOKEN")

    print(f"Loading raw data from HF dataset repo: {dataset_repo_id}")
    df = load_raw_from_hf(dataset_repo_id)

    print("Cleaning data...")
    df_clean = clean_data(df)

    print("Splitting and saving...")
    train_path, test_path = split_and_save(df_clean)

    print("Uploading splits back to HF dataset...")
    upload_splits_to_hf(dataset_repo_id, token, train_path, test_path)

if __name__ == "__main__":
    main()