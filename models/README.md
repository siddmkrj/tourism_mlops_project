
# Tourism Wellness Package Prediction Model

This machine learning model predicts whether a customer is likely to purchase the **Wellness Tourism Package** offered by *Visit With Us*.

## Model Information
- **Selected Model:** XGBoost
- **Algorithms Tested:** RandomForest, XGBoost  
- **Final Model Chosen:** XGBoost (based on higher F1-score on test data)
- **Task:** Binary Classification  
- **Target Variable:** `ProdTaken` (1 = Will purchase, 0 = Will not purchase)

## Performance (Test Set)
- Accuracy: 0.9153  
- Precision: 0.8870  
- Recall: 0.6415  
- F1-score: 0.7445  
- ROC-AUC: 0.9443  

## Training Data
The model was trained using customer demographic and interaction data from the dataset:
**`mukherjee78/tourism-wellness-package`**

## Usage
Load the model in Python:

```python
from huggingface_hub import hf_hub_download
import joblib

model_path = hf_hub_download(
    repo_id="mukherjee78/tourism-wellness-best-model",
    filename="best_model.pkl"
)

model = joblib.load(model_path)

License

This model is provided for educational purposes as part of the Advanced ML & MLOps project.
