import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Generate sample dataset based on notebook structure
def create_sample_dataset():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "Experience_Years": np.random.randint(0, 20, n),
        "Company_Size": np.random.randint(10, 5000, n),
        "City": np.random.choice(['Delhi', 'Bangalore', 'Pune', 'Jaipur'], n),
        "Industry": np.random.choice(['IT', 'Finance', 'Healthcare', 'Education'], n),
        "Job_Role": np.random.choice(['Data Analyst', 'ML Engineer', 'Web Developer', 'Manager'], n),
        "Skill_Set": np.random.choice(['Python', 'SQL', 'Excel', 'Java'], n),
        "Experience_Level": np.random.choice(['Fresher', 'Junior', 'Mid', 'Senior'], n)
    })

    # Generate salary based on logic
    df['Salary'] = (
        20000 +
        df['Experience_Years'] * 1500 +
        df['Company_Size'] * 2 +
        df['Experience_Level'].map({'Fresher': 0, 'Junior': 5000, 'Mid': 10000, 'Senior': 15000}) +
        df['Job_Role'].map({'Data Analyst': 5000, 'ML Engineer': 8000, 'Web Developer': 6000, 'Manager': 10000}) +
        np.random.normal(0, 3000, n)
    )

    return df

# Preprocess: encode categorical columns
def preprocess(df):
    le_dict = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict

# Train RandomForest + RFE
@st.cache_resource
def train_model():
    df = create_sample_dataset()
    df, le_dict = preprocess(df)

    X = df.drop("Salary", axis=1)
    y = df["Salary"]

    model = RandomForestRegressor(random_state=42)
    rfe = RFE(model, n_features_to_select=5)
    rfe.fit(X, y)

    selected_features = X.columns[rfe.support_]
    model.fit(X[selected_features], y)

    return model, selected_features, le_dict

# Main app logic
def main():
    st.title("💼 Salary Prediction App (Random Forest + RFE)")

    model, selected_features, le_dict = train_model()

    st.sidebar.header("📝 Enter Candidate Details")

    user_input = {}

    if "Experience_Years" in selected_features:
        user_input["Experience_Years"] = st.sidebar.slider("Years of Experience", 0, 30, 2)

    if "Company_Size" in selected_features:
        user_input["Company_Size"] = st.sidebar.slider("Company Size (Employees)", 10, 10000, 500)

    if "City" in selected_features:
        city = st.sidebar.selectbox("City", le_dict["City"].classes_)
        user_input["City"] = le_dict["City"].transform([city])[0]

    if "Industry" in selected_features:
        industry = st.sidebar.selectbox("Industry", le_dict["Industry"].classes_)
        user_input["Industry"] = le_dict["Industry"].transform([industry])[0]

    if "Job_Role" in selected_features:
        role = st.sidebar.selectbox("Job Role", le_dict["Job_Role"].classes_)
        user_input["Job_Role"] = le_dict["Job_Role"].transform([role])[0]

    if "Skill_Set" in selected_features:
        skill = st.sidebar.selectbox("Skill Set", le_dict["Skill_Set"].classes_)
        user_input["Skill_Set"] = le_dict["Skill_Set"].transform([skill])[0]

    if "Experience_Level" in selected_features:
        level = st.sidebar.selectbox("Experience Level", le_dict["Experience_Level"].classes_)
        user_input["Experience_Level"] = le_dict["Experience_Level"].transform([level])[0]

    input_df = pd.DataFrame([user_input])

    st.subheader("✅ Selected Features Used by Model")
    st.write(selected_features.tolist())

    if st.button("💰 Predict Salary"):
        prediction = model.predict(input_df[selected_features])[0]
        st.success(f"Estimated Salary: ₹ {int(prediction):,}")

if __name__ == "__main__":
    main()
