# crop_app.py

import streamlit as st
import joblib
import numpy as np


model = joblib.load('crop_recommendation_model.pkl')

st.set_page_config(page_title="🌱 Crop Recommendation", layout="centered")

st.title("🌾 Smart Krishi Assistant – Crop Recommendation")

st.header("🧪 Enter Soil and Environmental Parameters")


col1, col2 = st.columns(2)
with col1:
    nitrogen = st.number_input("Nitrogen (N)", 0, 200, step=1)
    phosphorus = st.number_input("Phosphorus (P)", 0, 200, step=1)
    potassium = st.number_input("Potassium (K)", 0, 200, step=1)
    moisture = st.slider("Moisture (%)", 0.0, 100.0, 30.0)
with col2:
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
    soil_type = st.selectbox("Soil Type", ['sandy', 'loamy', 'black', 'red', 'clay'])  # Adjust as per dataset

if st.button("🔍 Recommend Crop"):

    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, moisture, soil_type]])
    
    predicted_crop = model.predict(input_data)[0]

    st.success(f"✅ Recommended Crop: **{predicted_crop}**")
