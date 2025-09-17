import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# =========================
# Load Pickle Files
# =========================
base_path = os.path.dirname(__file__)

with open(os.path.join(base_path, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)

with open(os.path.join(base_path, "rfe.pkl"), "rb") as f:
    rfe = pickle.load(f)

with open(os.path.join(base_path, "final_model.pkl"), "rb") as f:
    model = pickle.load(f)

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Prediction App", layout="wide")

st.title("🚀 ML Prediction App")
st.write("This app makes predictions using the trained ML model (`final_model.pkl`).")

# User Input
st.subheader("Enter Feature Values")

input_data = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0, step=0.1)
    input_data.append(value)

# Convert input to numpy array
input_array = np.array(input_data).reshape(1, -1)

# Apply RFE Transformation
input_transformed = rfe.transform(input_array)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_transformed)
    st.success(f"✅ Predicted Value: {prediction[0]}")
