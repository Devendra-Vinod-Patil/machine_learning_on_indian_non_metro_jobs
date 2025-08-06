import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Simulated dataset creation
def create_data():
    np.random.seed(42)
    n = 500
    data = {
        'Experience': np.random.randint(0, 20, n),
        'Education': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], n),
        'Job Role': np.random.choice(['Developer', 'Data Scientist', 'Manager', 'Analyst'], n),
        'City': np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Hyderabad'], n),
        'Company Rating': np.round(np.random.uniform(2.5, 5.0, n), 1),
    }

    df = pd.DataFrame(data)

    # Generate Salary with some logic
    base_salary = 25000
    df['Salary'] = base_salary + \
                   df['Experience'] * 2000 + \
                   df['Company Rating'] * 3000 + \
                   df['Education'].map({
                       'High School': 0,
                       'Bachelors': 5000,
                       'Masters': 10000,
                       'PhD': 15000
                   }) + \
                   df['Job Role'].map({
                       'Developer': 5000,
                       'Data Scientist': 10000,
                       'Manager': 8000,
                       'Analyst': 6000
                   }) + \
                   np.random.normal(0, 2000, n)
    return df

# Encode categorical columns
def preprocess(df):
    le = LabelEncoder()
    df['Education'] = le.fit_transform(df['Education'])
    df['Job Role'] = le.fit_transform(df['Job Role'])
    df['City'] = le.fit_transform(df['City'])
    return df

# Train RFE + RF Model
@st.cache_resource
def train_model():
    df = create_data()
    df = preprocess(df)

    X = df.drop('Salary', axis=1)
    y = df['Salary']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    rfe = RFE(estimator=model, n_features_to_select=4)
    rfe.fit(X, y)

    selected_features = X.columns[rfe.support_]
    model.fit(X[selected_features], y)

    return model, selected_features, df

# Main Streamlit app
def main():
    st.title("💼 Salary Prediction App (Random Forest + RFE)")

    model, selected_features, df = train_model()

    st.sidebar.header("📋 Enter Candidate Details")

    user_input = {}

    if 'Experience' in selected_features:
        user_input['Experience'] = st.sidebar.slider("Years of Experience", 0, 30, 2)

    if 'Education' in selected_features:
        education = st.sidebar.selectbox("Education Level", ['High School', 'Bachelors', 'Masters', 'PhD'])
        user_input['Education'] = {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3}[education]

    if 'Job Role' in selected_features:
        job_role = st.sidebar.selectbox("Job Role", ['Developer', 'Data Scientist', 'Manager', 'Analyst'])
        user_input['Job Role'] = {'Developer': 0, 'Data Scientist': 1, 'Manager': 2, 'Analyst': 3}[job_role]

    if 'City' in selected_features:
        city = st.sidebar.selectbox("City", ['Delhi', 'Mumbai', 'Bangalore', 'Hyderabad'])
        user_input['City'] = {'Delhi': 0, 'Mumbai': 1, 'Bangalore': 2, 'Hyderabad': 3}[city]

    if 'Company Rating' in selected_features:
        user_input['Company Rating'] = st.sidebar.slider("Company Rating", 2.5, 5.0, 4.0, step=0.1)

    # Convert input to dataframe
    input_df = pd.DataFrame([user_input])

    st.subheader("🧪 Selected Features for Prediction")
    st.write(selected_features.tolist())

    if st.button("Predict Salary 💰"):
        prediction = model.predict(input_df[selected_features])[0]
        st.success(f"💸 Estimated Salary: ₹{int(prediction):,}")

    st.subheader("📊 Sample of Training Data")
    st.write(df.head())

if __name__ == "__main__":
    main()
