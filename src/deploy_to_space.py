import os
from huggingface_hub import HfApi

def main():
    space_repo_id = os.getenv("HF_SPACE_REPO_ID", "mukherjee78/tourism-wellness-space")
    token = os.getenv("HF_TOKEN")

    ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
    print(ABSOLUTE_PATH)

    api = HfApi(token=token)

    api.create_repo(
        repo_id=space_repo_id,
        repo_type="space",
        exist_ok=True,
        space_sdk="docker"
    )

    files_to_upload = [
        (f"{ABSOLUTE_PATH}/Dockerfile", "Dockerfile"),
        (f"{ABSOLUTE_PATH}/app.py", "app.py"),
        (f"{ABSOLUTE_PATH}/requirements.txt", "requirements.txt"),
    ]

    for local_path, remote_path in files_to_upload:
        print(f"Uploading {local_path} to {space_repo_id}:{remote_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=space_repo_id,
            repo_type="space"
        )

    print(f"✅ Deployment files pushed to Hugging Face Space: {space_repo_id}")

if __name__ == "__main__":
    main()