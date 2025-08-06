import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="RFE + Random Forest", layout="wide")

def load_data():
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    return X, y, X.columns.tolist()

def main():
    st.title("🏡 California Housing Price Prediction with RFE + Random Forest")
    st.markdown("""
    This app demonstrates **Recursive Feature Elimination (RFE)** with a **Random Forest Regressor**  
    using the built-in California Housing dataset.
    """)

    # Load dataset
    X, y, all_columns = load_data()
    st.write(f"Dataset shape: {X.shape}")

    with st.expander("📊 Preview Dataset"):
        st.dataframe(X.head())

    # Sidebar for number of features
    st.sidebar.header("🔧 RFE Settings")
    n_features = st.sidebar.slider("Select number of features to keep", min_value=1, max_value=len(all_columns), value=5)

    # 1. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. RFE setup
    base_model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=base_model, n_features_to_select=n_features, step=1)

    st.info("Running RFE...")
    start = time.time()
    rfe.fit(X_train, y_train)
    end = time.time()
    st.success(f"✅ RFE completed in {end - start:.2f} seconds")

    # 3. Transform the data
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)

    # 4. Final model
    final_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    final_model.fit(X_train_rfe, y_train)
    y_pred = final_model.predict(X_test_rfe)

    # 5. Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Model Evaluation")
    st.write(f"**Mean Squared Error (MSE):** `{mse:.4f}`")
    st.write(f"**R² Score:** `{r2:.4f}`")

    # 6. Display selected features
    selected_features = X.columns[rfe.support_]
    st.subheader("🔍 Selected Features by RFE")
    st.write(selected_features.tolist())

if __name__ == "__main__":
    main()
