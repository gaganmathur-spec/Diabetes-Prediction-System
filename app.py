import streamlit as st
import joblib
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# Load Model
model = joblib.load("diabetes_model.pkl")

# Title
st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter the patient's medical information below to predict whether the patient is likely to have diabetes."
)

st.divider()

# Input Fields
pregnancies = st.number_input("Pregnancies", 0, 20, 1)

glucose = st.number_input("Glucose", 0, 300, 120)

blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)

skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

insulin = st.number_input("Insulin", 0, 900, 80)

bmi = st.number_input("BMI", 0.0, 70.0, 25.0)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    0.000,
    3.000,
    0.500,
    format="%.3f"
)

age = st.number_input("Age", 1, 120, 30)

# Prediction Button
if st.button("Predict"):

    data = pd.DataFrame({
        "Pregnancies":[pregnancies],
        "Glucose":[glucose],
        "BloodPressure":[blood_pressure],
        "SkinThickness":[skin_thickness],
        "Insulin":[insulin],
        "BMI":[bmi],
        "DiabetesPedigreeFunction":[dpf],
        "Age":[age]
    })

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]

    st.divider()

    if prediction == 1:
        st.error("⚠️ The patient is likely to have Diabetes.")
        confidence = probability[1] * 100
    else:
        st.success("✅ The patient is unlikely to have Diabetes.")
        confidence = probability[0] * 100

    st.metric("Prediction Confidence", f"{confidence:.2f}%")

    st.subheader("Prediction Probability")

    st.write(f"🟢 Non-Diabetic : **{probability[0]*100:.2f}%**")
    st.write(f"🔴 Diabetic : **{probability[1]*100:.2f}%**")

st.divider()

st.info(
    "This prediction is generated using a Machine Learning model "
    "and should not be considered medical advice. "
    "Please consult a healthcare professional for diagnosis."
)