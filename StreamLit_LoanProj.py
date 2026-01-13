import pandas as pd
import streamlit as st
import os
import joblib
from sklearn.preprocessing import StandardScaler

######################## TO RUN:
# 1 INSTALL STREAMLIT
# 2 streamlit run demo_main.py


st.set_page_config(
    page_title="Loan Approval Checker",
    page_icon="🏦",
    layout="centered"
)

# Title
st.title("🏦 Loan Approval Checker")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Loan_Logi_Model.joblib")
prediction = 0

# If model missing -> show waiting message and stop
if not os.path.exists(MODEL_PATH):
     st.warning("The system is initializing, please wait")
     st.code(MODEL_PATH)
     st.stop()


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

model = load_model(MODEL_PATH)


# ---- Input form (screenshot-friendly UI) ----
with st.form("loan_form"):
    st.subheader("Loan Application Details")

    applicant_income = st.number_input(
        "Applicant Income (monthly)",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income (monthly)",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    loan_amount = st.number_input(
        "Requested Loan Amount",
        min_value=0.0,
        value=150.0,
        step=10.0
    )

    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=1.0,
        value=360.0,
        step=12.0
    )

    credit_history = st.selectbox(
        "Credit History",
        options=[1, 0],
        index=0,
        format_func=lambda x: "Exists (1)" if x == 1 else "Does not exist (0)"
    )
    dict_opt = {"Yes": "Married", "No": "Single"}
    married = st.selectbox(
        "Marital Status",
        options=["Yes", "No"],
        index=0,
        format_func=lambda x: dict_opt[x]
    )

    submitted = st.form_submit_button("Check Loan Eligibility")


# Runs only after clicking the button
if submitted:
    # Convert Married to dummy (drop_first=True → keep only Married_Yes)
    married_yes = 1 if married == "Yes" else 0

    # IMPORTANT:
    # Column names must exactly match those used during model training
    X_to_pred = pd.DataFrame([{
        "ApplicantIncome": float(applicant_income),
        "CoapplicantIncome": float(coapplicant_income),
        "LoanAmount": float(loan_amount),
        "Loan_Amount_Term": float(loan_term),
        "Credit_History": int(credit_history),
        "Married": int(married_yes),
    }])

    st.dataframe(X_to_pred)



    prediction = model.predict(X_to_pred)
    st.info("You approved for taking a loan Good luck" if prediction == 1 else "You are not allowed to take a loan we are sorry")




