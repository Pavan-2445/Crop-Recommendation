import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('crop_recommendation_model.pkl')

# Soil type mapping used during training
soil_mapping = {
    'sandy': 0,
    'loamy': 1,
    'black': 2,
    'red': 3,
    'clay': 4
}

st.set_page_config(page_title="🌱 Crop Recommendation", layout="centered")
st.title("🌾 Smart Krishi Assistant – Crop Recommendation")
st.header("🧪 Enter Soil and Environmental Parameters")

col1, col2 = st.columns(2)

with col1:
    nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=300, step=1)
    phosphorus = st.number_input("Phosphorus (P)", min_value=0, max_value=300, step=1)
    potassium = st.number_input("Potassium (K)", min_value=0, max_value=300, step=1)
    moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.2f")

with col2:
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, step=0.1, format="%.2f")
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.2f")
    soil_type = st.selectbox("Soil Type", list(soil_mapping.keys()))

# Predict
if st.button("🔍 Recommend Crop"):
    encoded_soil = soil_mapping[soil_type]
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, moisture, encoded_soil]])

    predicted_crop = model.predict(input_data)[0]

    st.success(f"✅ Recommended Crop: **{predicted_crop}**")
