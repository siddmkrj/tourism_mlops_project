import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Tourism Package Predictor")

HF_USERNAME = "mukherjee78"
HF_MODEL_REPO = f"{HF_USERNAME}/tourism-wellness-best-model"
MODEL_FILENAME = "best_model.pkl"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=MODEL_FILENAME,
        repo_type="model"
    )
    model = joblib.load(model_path)
    return model

model = load_model()

st.title("🧘 Wellness Tourism Package - Purchase Prediction")
st.write("Enter customer details to predict whether they will purchase the package.")

# -------------------------
# USER INPUT FORM
# -------------------------
with st.form("input_form"):
    Age = st.number_input("Age", min_value=18, max_value=100, value=35)
    TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    CityTier = st.selectbox("City Tier", [1,2,3])
    Occupation = st.selectbox("Occupation", ["Salaried","Self Employed","Freelancer","Other"])
    Gender = st.selectbox("Gender", ["Male","Female"])
    NumberOfPersonVisiting = st.number_input("Number Of Persons Visiting", 1, 20, 2)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [1,2,3,4,5])
    MaritalStatus = st.selectbox("Marital Status", ["Single","Married","Divorced"])
    NumberOfTrips = st.number_input("Number Of Trips", 0, 50, 2)
    Passport = st.selectbox("Passport", [0,1])
    OwnCar = st.selectbox("Own Car", [0,1])
    NumberOfChildrenVisiting = st.number_input("Children Visiting", 0,10,0)
    Designation = st.selectbox("Designation", ["Executive","Manager","Senior Manager","Other"])
    MonthlyIncome = st.number_input("Monthly Income", 0, 1000000, 50000)
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    ProductPitched = st.selectbox("Product Pitched", ["Basic","Standard","Deluxe","Other"])
    NumberOfFollowups = st.number_input("Number Of Follow-ups", 0, 20, 1)
    DurationOfPitch = st.number_input("Duration Of Pitch (minutes)", 1, 200, 30)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "ProductPitched": ProductPitched,
        "NumberOfFollowups": NumberOfFollowups,
        "DurationOfPitch": DurationOfPitch
    }])

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✔ Customer is LIKELY to purchase (Probability: {prob:.2%})")
    else:
        st.warning(f"✘ Customer is UNLIKELY to purchase (Probability: {prob:.2%})")