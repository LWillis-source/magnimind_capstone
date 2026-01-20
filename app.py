import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Titanic Survival Predictor",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction")
st.write("This app uses a **pre-trained Logistic Regression pipeline**.")

# ---------------------------
# Load saved model
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("titanic_kaggle_best_model.pkl")

model = load_model()

# ---------------------------
# User input
# ---------------------------
st.header("Passenger Information")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
    sibsp = st.number_input("Siblings / Spouses", 0, 10, 0)

with col2:
    parch = st.number_input("Parents / Children", 0, 10, 0)
    fare = st.number_input("Fare", min_value=0.0, value=32.0)
    embarked = st.selectbox("Embarked", ["S", "C", "Q"])
    cabin = st.text_input("Cabin (optional)", value="")

name = st.text_input("Full Name", "Doe, Mr. John")
ticket = st.text_input("Ticket", "A/5 21171")

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Survival"):
    input_df = pd.DataFrame([{
        "PassengerId": 0,
        "Pclass": pclass,
        "Name": name,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": ticket,
        "Fare": fare,
        "Cabin": cabin if cabin != "" else np.nan,
        "Embarked": embarked,
        "Survived": 0   # dummy, dropped by pipeline
    }])

    prob = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.metric("Survival Probability", f"{prob:.2%}")

    if pred == 1:
        st.success("✅ Likely to Survive")
    else:
        st.error("❌ Unlikely to Survive")
