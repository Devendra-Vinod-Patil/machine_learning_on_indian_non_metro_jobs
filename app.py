import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and RFE transformer
@st.cache_resource
def load_artifacts():
    model = joblib.load("final_model.pkl")
    rfe = joblib.load("rfe.pkl")
    feature_names = joblib.load("feature_names.pkl")  # original X_train.columns
    return model, rfe, feature_names

# App layout
def main():
    st.set_page_config(page_title="RFE + Random Forest Prediction", layout="wide")
    st.title("📊 ML Prediction using RFE + Random Forest")

    st.markdown("""
        Upload a dataset matching the **original feature columns** before RFE was applied.
        The model will apply RFE transformation and then make predictions.
    """)

    model, rfe, feature_names = load_artifacts()

    uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            missing_cols = [col for col in feature_names if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns in uploaded data: {missing_cols}")
                return

            X_input = df[feature_names]
            X_rfe = rfe.transform(X_input)

            preds = model.predict(X_rfe)

            df["Prediction"] = preds
            st.success("✅ Predictions generated successfully!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
