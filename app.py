import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

@st.cache_data
def load_data():
    # Simulated data load — replace with your real dataset if embedding in app
    df = pd.read_csv("indian_non_metro_jobs_dataset.csv")
    
    # Basic cleaning: drop rows with nulls
    df = df.dropna()
    
    # Use only numerical features for this model
    df_num = df.select_dtypes(include='number')
    
    # Define target and features
    target_col = 'Salary'  # Adjust based on your dataset
    X = df_num.drop(columns=[target_col])
    y = df_num[target_col]
    
    return X, y, list(X.columns)

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Lightweight model for RFE
    base_model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    rfe = RFE(estimator=base_model, n_features_to_select=5, step=5)
    rfe.fit(X_train, y_train)
    
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)
    
    # Stronger model after feature selection
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_rfe, y_train)

    score = model.score(X_test_rfe, y_test)
    return model, rfe, score

def main():
    st.title("🎯 Job Salary Prediction (Non-Metro Cities)")

    X, y, feature_names = load_data()
    model, rfe, accuracy = train_model(X, y)

    st.success(f"Model trained with R² score: {accuracy:.2f}")
    
    # Get selected features
    selected_features = [f for f, selected in zip(feature_names, rfe.support_) if selected]

    st.subheader("📥 Enter values to predict Salary")
    user_input = {}
    for feature in selected_features:
        value = st.number_input(f"{feature}", min_value=0.0, step=1.0)
        user_input[feature] = value

    if st.button("Predict Salary"):
        input_array = np.array([list(user_input.values())]).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        st.success(f"Predicted Salary: ₹{prediction:,.2f}")

if __name__ == "__main__":
    main()
