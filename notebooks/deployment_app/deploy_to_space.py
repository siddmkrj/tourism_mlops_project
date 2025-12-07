
from huggingface_hub import HfApi
import os

HF_USERNAME = "mukherjee78"
SPACE_REPO_ID = f"{HF_USERNAME}/tourism-wellness-app"

api = HfApi()

# Create the Space (streamlit)
api.create_repo(
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
)

# Upload all files in deployment_app/
api.upload_folder(
    folder_path="deployment_app",
    repo_id=SPACE_REPO_ID,
    repo_type="space"
)

print("Deployment uploaded to:", f"https://huggingface.co/spaces/{SPACE_REPO_ID}")
