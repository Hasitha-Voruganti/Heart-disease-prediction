import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# ---------------------------
# Load Model, Scaler, and Feature Columns
# ---------------------------
model = joblib.load('heart_disease_svm_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="💓", layout="centered")
st.title("💓 Heart Disease Prediction App")
st.markdown("""
Predict the likelihood of **heart disease** using patient health data.  
Model trained on **Kaggle Heart Failure Prediction Dataset** using **SVM**.
""")

# ---------------------------
# User Input Section
# ---------------------------
st.header("🧍‍♂️ Enter Patient Health Parameters")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=55)
    
    sex_input = st.selectbox("Sex", ["Male", "Female"])
    sex_map = {"Male": "M", "Female": "F"}
    sex_col = f"Sex_{sex_map[sex_input]}"
    
    chest_pain_input = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina (TA)", "Atypical Angina (ATA)", 
         "Non-Anginal Pain (NAP)", "Asymptomatic (ASY)"]
    )
    chest_pain_map = {"Typical Angina (TA)":"TA",
                      "Atypical Angina (ATA)":"ATA",
                      "Non-Anginal Pain (NAP)":"NAP",
                      "Asymptomatic (ASY)":"ASY"}
    chest_pain_col = f"ChestPainType_{chest_pain_map[chest_pain_input]}"
    
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=140)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=250)
    
    fasting_bs_input = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
    fasting_bs = 1 if fasting_bs_input == "Yes" else 0

with col2:
    resting_ecg_input = st.selectbox("Resting ECG", 
                                     ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    resting_ecg_map = {"Normal":"Normal",
                       "ST-T Wave Abnormality":"ST",
                       "Left Ventricular Hypertrophy":"LVH"}
    resting_ecg_col = f"RestingECG_{resting_ecg_map[resting_ecg_input]}"
    
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=120)
    
    exercise_angina_input = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    exercise_angina_map = {"No":"N", "Yes":"Y"}
    exercise_angina_col = f"ExerciseAngina_{exercise_angina_map[exercise_angina_input]}"
    
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=2.3)
    
    st_slope_input = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    st_slope_col = f"ST_Slope_{st_slope_input}"

# ---------------------------
# Prepare Input DataFrame
# ---------------------------
input_df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

# Fill numeric values
input_df.loc[0, 'Age'] = age
input_df.loc[0, 'RestingBP'] = resting_bp
input_df.loc[0, 'Cholesterol'] = cholesterol
input_df.loc[0, 'FastingBS'] = fasting_bs
input_df.loc[0, 'MaxHR'] = max_hr
input_df.loc[0, 'Oldpeak'] = oldpeak

# Fill one-hot encoded categorical values safely
for col in [sex_col, chest_pain_col, resting_ecg_col, exercise_angina_col, st_slope_col]:
    if col in input_df.columns:
        input_df.loc[0, col] = 1

# ---------------------------
# Scale and Predict
# ---------------------------
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

try:
    probability = model.predict_proba(input_scaled)[0][1]
except:
    probability = 0.6 if prediction == 1 else 0.2

# Risk levels
if probability < 0.33:
    risk = "🟢 Low Risk"
    color = "green"
elif probability < 0.66:
    risk = "🟠 Medium Risk"
    color = "orange"
else:
    risk = "🔴 High Risk"
    color = "red"

# ---------------------------
# Display Results
# ---------------------------
st.subheader("📊 Prediction Result")
st.markdown(f"### **Prediction:** {'⚠ Heart Disease Detected' if prediction==1 else '💚 No Heart Disease Detected'}")
st.markdown(f"### **Risk Level:** {risk}")

# Gauge chart for probability
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability*100,
    title={'text': "Heart Disease Risk (%)"},
    gauge={
        'axis': {'range': [0,100]},
        'bar': {'color': color},
        'steps': [
            {'range': [0,33], 'color': "#90EE90"},
            {'range': [33,66], 'color': "#FFD580"},
            {'range': [66,100], 'color': "#FF6F61"},
        ],
    }
))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
---
💡 *Developed by Hasitha Voruganti*  
🩺 Powered by Machine Learning (SVM)
""")
