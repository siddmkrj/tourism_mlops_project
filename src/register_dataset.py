import os
from huggingface_hub import HfApi

def main():
    dataset_repo_id = os.getenv("HF_DATASET_REPO_ID", "mukherjee78/tourism-wellness-package")
    token = os.getenv("HF_TOKEN")

    local_data_path = os.path.join("data", "tourism.csv")

    if not os.path.exists(local_data_path):
        raise FileNotFoundError(f"Raw data file not found at {local_data_path}")

    api = HfApi(token=token)

    print(f"Uploading {local_data_path} to dataset repo: {dataset_repo_id}")
    api.upload_file(
        path_or_fileobj=local_data_path,
        path_in_repo="data/tourism.csv",
        repo_id=dataset_repo_id,
        repo_type="dataset",
    )
    print("✅ Dataset uploaded successfully.")

if __name__ == "__main__":
    main()