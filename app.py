import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_indian_non_metro_jobs_data.csv")
    df_num = df.select_dtypes(include=[np.number])
    target_col = 'MaxSalary'

    # Drop rows with missing target
    df_num = df_num.dropna(subset=[target_col])
    X = df_num.drop(columns=[target_col])
    y = df_num[target_col]

    return X, y, X.columns.tolist()

# Train model using RFE
@st.cache_data
def train_model(X, y, n_features=30):
    base_model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    rfe = RFE(estimator=base_model, n_features_to_select=n_features, step=10)
    rfe.fit(X, y)

    selected_features = X.columns[rfe.support_]
    X_rfe = rfe.transform(X)

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_rfe, y)

    return model, rfe, selected_features

def main():
    st.title("📊 Non-Metro Job Salary Predictor (Using RFE + Random Forest)")
    X, y, feature_names = load_data()

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model using RFE
    model, rfe, selected_features = train_model(X_train, y_train, n_features=30)

    # Transform test set and evaluate
    X_test_rfe = rfe.transform(X_test)
    y_pred = model.predict(X_test_rfe)

    st.subheader("📈 Model Evaluation")
    st.write(f"**R² Score**: {r2_score(y_test, y_pred):.2f}")
    st.write(f"**MSE**: {mean_squared_error(y_test, y_pred):,.2f}")

    # User input
    st.sidebar.header("📥 Enter Feature Values")
    user_input = []
    for feature in selected_features:
        value = st.sidebar.number_input(f"{feature}", value=float(X[feature].mean()))
        user_input.append(value)

    if st.sidebar.button("Predict Salary"):
        input_array = np.array(user_input).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        st.success(f"💰 Predicted Max Salary: ₹{prediction:,.2f}")

if __name__ == "__main__":
    main()
