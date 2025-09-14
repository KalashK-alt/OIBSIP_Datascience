# app.py
import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("Car Price Prediction")

PIPE_PATH = os.path.join("artifacts", "car_price_pipeline.pkl")
CURRENT_YEAR = 2025

@st.cache_data(show_spinner=False)
def load_pipeline():
    return joblib.load(PIPE_PATH)

pipeline = load_pipeline()

st.markdown("Enter car details below and click **Predict**")

# Inputs
col1, col2 = st.columns(2)
with col1:
    car_name = st.text_input("Car name / model (e.g., 'ciaz', 'city', 'fortuner')", value="ciaz")
    year = st.number_input("Year (manufacture)", min_value=1950, max_value=CURRENT_YEAR, value=2016)
    present_price = st.number_input("Present (company) price", min_value=0.0, value=8.0, format="%.2f")
    driven_kms = st.number_input("Driven kms", min_value=0, value=20000, step=1000)
with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    selling_type = st.selectbox("Selling Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("No. of previous owners", [0, 1, 3])

if st.button("Predict"):
    age = CURRENT_YEAR - year
    brand = str(car_name).split()[0].lower()
    row = {
        "Present_Price": present_price,
        "Driven_kms": driven_kms,
        "Owner": owner,
        "Age": age,
        "Fuel_Type": fuel_type,
        "Selling_type": selling_type,
        "Transmission": transmission,
        "Brand": brand
    }
    X = pd.DataFrame([row])
    pred = pipeline.predict(X)[0]
    st.success(f"Predicted selling price: **{pred:.2f}** (same units as `Selling_Price` column in dataset)")
    st.info("Tip: this is a baseline model. See README for improvement ideas.")
