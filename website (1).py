import streamlit as st
import joblib

# ---------- PAGE SETTINGS ----------
st.set_page_config(page_title="Health Risk Predictor", page_icon="❤️")

st.title("Health Risk Predictor")
st.write(
    "Mini project using **Logistic Regression** models to estimate risk for:\n"
    "- ❤️ Heart disease\n"
    "- 💉 Diabetes\n\n"
    "- ♀️ PCOS\n\n\n"
    "This is only for learning, **not** real medical advice."
)

# ---------- LOAD MODELS ----------
heart_model = joblib.load("heart_model.pkl")
diabetes_model = joblib.load("diabetes_model.pkl")
pcos_model = joblib.load("pcos_final.pkl")


# ---------- TABS ----------
tab_heart, tab_diabetes, tab_pcos = st.tabs(["❤️ Heart Disease", "💉 Diabetes", "♀️PCOS"])

# ===================== HEART TAB (SIMPLE) =====================
with tab_heart:
    st.subheader("Heart Disease Risk (Simple Inputs)")

    age = st.number_input("Age", min_value=1, max_value=120, value=50)

    sex = st.radio("Gender", ["Male", "Female"])
    sex_val = 1 if sex == "Male" else 0

    cp = st.selectbox(
        "Chest Pain Type (cp)",
        [
            "0 - Typical angina",
            "1 - Atypical angina",
            "2 - Non-anginal pain",
            "3 - Asymptomatic"
        ]
    )
    cp_val = int(cp[0])  # take the number at start

    chol = st.number_input(
        "Cholesterol (mg/dl)",
        min_value=100, max_value=600, value=230
    )

    thalach = st.number_input(
        "Max Heart Rate Achieved (thalach)",
        min_value=60, max_value=250, value=150
    )

    if st.button("Predict Heart Risk"):
        # DEFAULT VALUES for remaining features (same order as training)
        trestbps = 130   # resting blood pressure
        fbs = 0          # fasting blood sugar > 120 mg/dl (0 = no)
        restecg = 0      # resting ECG
        exang = 0        # exercise induced angina
        oldpeak = 1.0    # ST depression
        slope = 1        # slope of peak exercise ST segment
        ca = 0           # number of major vessels
        thal = 1         # thal value

        X = [[
            age, sex_val, cp_val, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]]

        pred = heart_model.predict(X)[0]

        if pred == 1:
            st.error("🔴 Model prediction: **High risk of heart disease**")
        else:
            st.success("🟢 Model prediction: **Low risk of heart disease**")

# ===================== DIABETES TAB =====================
with tab_diabetes:
    st.subheader("Diabetes Risk")

    pregnancies = st.number_input("Pregnancies", min_value=0, value=1)
    glucose = st.number_input("Glucose", value=120.0)
    blood_pressure = st.number_input("Blood Pressure", value=70.0)
    skin_thickness = st.number_input("Skin Thickness", value=20.0)
    insulin = st.number_input("Insulin", value=80.0)
    bmi = st.number_input("BMI", value=30.0)
    dpf = st.number_input("Diabetes Pedigree Function", value=0.5)
    age_d = st.number_input("Age", min_value=1, value=30)

    if st.button("Predict Diabetes Risk"):
        X = [[
            pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, dpf, age_d
        ]]
        pred = diabetes_model.predict(X)[0]

        if pred == 1:
            st.error("🔴 Model prediction: **High risk of diabetes**")
        else:
            st.success("🟢 Model prediction: **Low risk of diabetes**")

#================== PCOS TAB =======================
with tab_pcos:
    st.header("PCOS Risk Prediction")

    age = st.number_input("Age", 10, 60)
    bmi = st.number_input("BMI", 10.0, 60.0)

    menstrual = st.selectbox("Menstrual Regularity", ["Regular", "Irregular"])
    acne = st.selectbox("Acne Severity", ["Mild", "Moderate", "Severe"])
    stress = st.selectbox("Stress Levels", ["Low", "Medium", "High"])

    # mappings
    menstrual_map = {"Regular": 0, "Irregular": 1}
    acne_map = {"Mild": 0, "Moderate": 1, "Severe": 2}
    stress_map = {"Low": 0, "Medium": 1, "High": 2}

    menstrual_val = menstrual_map[menstrual]
    acne_val = acne_map[acne]
    stress_val = stress_map[stress]

    if st.button("Predict PCOS Risk"):
        X = [[age, bmi, menstrual_val, acne_val, stress_val]]
        pred = pcos_model.predict(X)[0]

        if pred == 1:
            st.error("🔴 Model prediction: High PCOS Risk")
        else:
            st.success("🟢 Model prediction: Low PCOS Risk")

st.markdown("---")
st.caption("Created as a first-year engineering mini project.")
