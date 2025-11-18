import streamlit as st
import joblib

# load models (files must be in same folder)
heart_model = joblib.load("heart_model.pkl")
diabetes_model = joblib.load("diabetes_model.pkl")

st.title("Simple Health Predictor")

option = st.radio("Choose model:", ["Heart disease", "Diabetes"])

# ---------------- HEART DISEASE ----------------
if option == "Heart disease":
    st.subheader("Heart disease input")

    # These must be in the SAME ORDER as training
    age = st.number_input("age", value=50)
    sex = st.number_input("sex (1=male, 0=female)", min_value=0, max_value=1, value=1)
    cp = st.number_input("cp (0-3)", min_value=0, max_value=3, value=0)
    trestbps = st.number_input("trestbps (resting BP)", value=130)
    chol = st.number_input("chol (cholesterol)", value=230)
    fbs = st.number_input("fbs (1 if >120 mg/dl else 0)", min_value=0, max_value=1, value=0)
    restecg = st.number_input("restecg (0-2)", min_value=0, max_value=2, value=0)
    thalach = st.number_input("thalach (max heart rate)", value=150)
    exang = st.number_input("exang (1=yes, 0=no)", min_value=0, max_value=1, value=0)
    oldpeak = st.number_input("oldpeak", value=1.0)
    slope = st.number_input("slope (0-2)", min_value=0, max_value=2, value=1)
    ca = st.number_input("ca (0-4)", min_value=0, max_value=4, value=0)
    thal = st.number_input("thal (0-3)", min_value=0, max_value=3, value=1)

    if st.button("Predict (heart)"):
        X = [[
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]]
        pred = heart_model.predict(X)[0]
        if pred == 1:
            st.write("🧠 Prediction: **High risk of heart disease**")
        else:
            st.write("🧠 Prediction: **Low risk of heart disease**")

# ---------------- DIABETES ----------------
elif option == "Diabetes":
    st.subheader("Diabetes input")

    pregnancies = st.number_input("Pregnancies", min_value=0, value=1)
    glucose = st.number_input("Glucose", value=120.0)
    blood_pressure = st.number_input("BloodPressure", value=70.0)
    skin_thickness = st.number_input("SkinThickness", value=20.0)
    insulin = st.number_input("Insulin", value=80.0)
    bmi = st.number_input("BMI", value=30.0)
    dpf = st.number_input("DiabetesPedigreeFunction", value=0.5)
    age = st.number_input("Age", min_value=1, value=30)

    if st.button("Predict (diabetes)"):
        X = [[
            pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi,
            dpf, age
        ]]
        pred = diabetes_model.predict(X)[0]
        if pred == 1:
            st.write("🧠 Prediction: **High risk of diabetes**")
        else:
            st.write("🧠 Prediction: **Low risk of diabetes**")
streamlit run app.py