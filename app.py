import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

# Skill options
SKILL_OPTIONS = ['Python', 'SQL', 'Excel', 'Java']

# Generate synthetic dataset
def create_sample_dataset():
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        "Experience_Years": np.random.randint(0, 20, n),
        "Company_Size": np.random.randint(10, 5000, n),
        "City": np.random.choice(['Delhi', 'Bangalore', 'Pune', 'Jaipur'], n),
        "Industry": np.random.choice(['IT', 'Finance', 'Healthcare', 'Education'], n),
        "Job_Role": np.random.choice(['Data Analyst', 'ML Engineer', 'Web Developer', 'Manager'], n),
        "Experience_Level": np.random.choice(['Fresher', 'Junior', 'Mid', 'Senior'], n),
    })

    for skill in SKILL_OPTIONS:
        df[f"Skill_{skill}"] = np.random.choice([0, 1], n)

    df['Salary'] = (
        20000 +
        df['Experience_Years'] * 1800 +
        df['Company_Size'] * 2.5 +
        df['Experience_Level'].map({'Fresher': 0, 'Junior': 5000, 'Mid': 10000, 'Senior': 15000}) +
        df['Job_Role'].map({'Data Analyst': 5000, 'ML Engineer': 9000, 'Web Developer': 6000, 'Manager': 12000}) +
        df[[f"Skill_{s}" for s in SKILL_OPTIONS]].sum(axis=1) * 2500 +
        np.random.normal(0, 2000, n)
    )
    return df

# Encode categoricals
def preprocess(df):
    le_dict = {}
    for col in ['City', 'Industry', 'Job_Role', 'Experience_Level']:
        df[col], uniques = pd.factorize(df[col])
        le_dict[col] = uniques
    return df, le_dict

@st.cache_resource
def train_model():
    df = create_sample_dataset()
    df, le_dict = preprocess(df)

    X = df.drop("Salary", axis=1)
    y = df["Salary"]

    model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=4, random_state=42)
    rfe = RFE(estimator=model, n_features_to_select=6)
    rfe.fit(X, y)

    selected_features = X.columns[rfe.support_]
    model.fit(X[selected_features], y)

    return model, selected_features, le_dict

# UI
def main():
    st.set_page_config(page_title="Salary Predictor", layout="wide")
    st.title("💼 Salary Prediction App")
    st.markdown("---")

    model, selected_features, le_dict = train_model()

    st.sidebar.header("📝 Candidate Information")
    user_input = {}

    # Input sliders and selectors
    if "Experience_Years" in selected_features:
        user_input["Experience_Years"] = st.sidebar.slider("Years of Experience", 0, 30, 2)

    if "Company_Size" in selected_features:
        user_input["Company_Size"] = st.sidebar.number_input("Company Size", min_value=10, max_value=10000, value=100)

    if "City" in selected_features:
        city = st.sidebar.selectbox("City", le_dict["City"])
        user_input["City"] = list(le_dict["City"]).index(city)

    if "Industry" in selected_features:
        industry = st.sidebar.selectbox("Industry", le_dict["Industry"])
        user_input["Industry"] = list(le_dict["Industry"]).index(industry)

    if "Job_Role" in selected_features:
        role = st.sidebar.selectbox("Job Role", le_dict["Job_Role"])
        user_input["Job_Role"] = list(le_dict["Job_Role"]).index(role)

    if "Experience_Level" in selected_features:
        level = st.sidebar.selectbox("Experience Level", le_dict["Experience_Level"])
        user_input["Experience_Level"] = list(le_dict["Experience_Level"]).index(level)

    selected_skills = st.sidebar.multiselect("Skills", SKILL_OPTIONS)
    for skill in SKILL_OPTIONS:
        user_input[f"Skill_{skill}"] = 1 if skill in selected_skills else 0

    input_df = pd.DataFrame([user_input])

    st.markdown("## 🔍 Features Used in Prediction")
    st.info(", ".join(selected_features))

    # Display summary
    with st.expander("📋 Candidate Summary", expanded=True):
        show_df = input_df.copy()
        for col in le_dict:
            if col in show_df.columns:
                idx = show_df.at[0, col]
                show_df.at[0, col] = le_dict[col][idx]
        st.table(show_df)

    if st.button("💰 Predict Salary", use_container_width=True):
        prediction = model.predict(input_df[selected_features])[0]
        st.markdown(f"""<div style='text-align:center; font-size:28px; padding:20px; background-color:#f0f8ff; border-radius:10px;'>
            <strong>Estimated Salary: ₹ {int(prediction):,}</strong>
        </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
