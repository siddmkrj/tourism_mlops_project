import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/best_model.joblib")

st.title("Tourism Wellness Package Purchase Prediction")

with st.form("input_form"):
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited", "Employee Referral"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.number_input("Duration of Pitch", min_value=0, max_value=100, value=30)
    Occupation = st.selectbox("Occupation", ["Salaried", "Business", "Retired", "Student", "Other"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("Number of Person Visiting", min_value=1, max_value=10, value=1)
    NumberOfFollowups = st.number_input("Number of Followups", min_value=0, max_value=20, value=1)
    ProductPitched = st.selectbox("Product Pitched", ["Product1", "Product2", "Product3"])
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    NumberOfTrips = st.number_input("Number of Trips", min_value=0, max_value=50, value=1)
    Passport = st.selectbox("Passport", ["Yes", "No"])
    PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    OwnCar = st.selectbox("Own Car", ["Yes", "No"])
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
    Designation = st.selectbox("Designation", ["Manager", "Executive", "Staff", "Other"])
    MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=50000)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome,
    }
    input_df = pd.DataFrame([input_dict])

    if "Unnamed: 0" in getattr(getattr(model, "feature_names_in_", []), "tolist", lambda: [])():
        input_df.insert(0, "Unnamed: 0", 0)
    elif "Unnamed: 0" in getattr(model, "feature_names_in_", []):
        input_df.insert(0, "Unnamed: 0", 0)
    else:
        if "Unnamed: 0" not in input_df.columns:
            input_df.insert(0, "Unnamed: 0", 0)

    expected_columns = [
        "Unnamed: 0",
        "Age",
        "TypeofContact",
        "CityTier",
        "DurationOfPitch",
        "Occupation",
        "Gender",
        "NumberOfPersonVisiting",
        "NumberOfFollowups",
        "ProductPitched",
        "PreferredPropertyStar",
        "MaritalStatus",
        "NumberOfTrips",
        "Passport",
        "PitchSatisfactionScore",
        "OwnCar",
        "NumberOfChildrenVisiting",
        "Designation",
        "MonthlyIncome",
    ]

    input_df = input_df[expected_columns]

    prediction = model.predict(input_df)[0]
    result = "Will Purchase" if prediction == 1 else "Will Not Purchase"
    st.write(f"Prediction: **{result}**")