import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    st.title("Random Forest with RFE Feature Selection")

    # Generate synthetic regression data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y

    st.subheader("Synthetic Dataset Preview")
    st.write(df.head())

    # Select number of features to keep
    n_select = st.slider("Select number of features to keep", min_value=1, max_value=20, value=10)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # RFE
    rfe = RFE(estimator=model, n_features_to_select=n_select)
    rfe.fit(X_train, y_train)

    selected_features = np.array(feature_names)[rfe.support_]
    st.subheader("Selected Features")
    st.write(selected_features)

    # Predict with selected features
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe, y_train)
    y_pred = model.predict(X_test_rfe)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.subheader("Model Performance")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

if __name__ == "__main__":
    main()
