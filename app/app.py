import streamlit as st
import joblib
import pandas as pd
model = joblib.load("model/heart_disease_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction")

st.title("Heart Disease Predictor")

st.write("Predict the likelihood of heart disease.")
st.header("Patient Information")

age = st.number_input("Age", 18, 100, 50)
trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
thalch = st.number_input("Maximum Heart Rate", 60, 250, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression during excercise (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
sex = st.selectbox("Sex", ["Female", "Male"])

cp = st.selectbox(
    "Chest Pain Type",
    ["asymptomatic", "atypical angina", "non-anginal", "typical angina"]
)

restecg = st.selectbox(
    "Resting ECG",
    ["lv hypertrophy", "normal", "st-t abnormality"]
)

slope = st.selectbox(
    "ST Slope",
    ["downsloping", "flat", "upsloping"]
)

thal = st.selectbox(
    "Thallium stress test result ",
    ["fixed defect", "normal", "reversable defect"]
)
input_data = {
    "age": age,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "thalch": thalch,
    "exang": exang,
    "oldpeak":oldpeak,
    "ca": ca,
    "ca_missing": 0,

    "sex_Male": 1 if sex == "Male" else 0,
    "cp_atypical angina": 1 if cp == "atypical angina" else 0,
    "cp_non-anginal": 1 if cp == "non-anginal" else 0,
    "cp_typical angina": 1 if cp == "typical angina" else 0,

    "restecg_normal": 1 if restecg == "normal" else 0,
    "restecg_st-t abnormality": 1 if restecg == "st-t abnormality" else 0,

    "slope_flat": 1 if slope == "flat" else 0,
    "slope_upsloping": 1 if slope == "upsloping" else 0,

    "thal_normal": 1 if thal == "normal" else 0,
    "thal_reversable defect": 1 if thal == "reversable defect" else 0,
}

input_df = pd.DataFrame([input_data])
if st.button("Predict"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High likelihood of Heart Disease\nProbability: {probability:.1%}")
    else:
        st.success(f"✅ Low likelihood of Heart Disease\nProbability: {(1-probability):.1%}")
