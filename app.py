import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model_path = "rfmodel.pkl"
model = joblib.load(model_path)

st.title("Loan Approval Prediction App")

name = st.text_input("Enter your name")

if 'show_form' not in st.session_state:
    st.session_state.show_form = False

if st.button("Next"):
    if name.strip():
        st.session_state.show_form = True
    else:
        st.warning("Please enter your name first.")


if st.session_state.show_form:
    st.write(f"👋 Hi, {name}! Please fill in your loan details below:")

    gender = st.selectbox("Gender", ['Male', 'Female'])
    married = st.selectbox("Marital Status", ['Married', 'UnMarried'])
    education = st.selectbox("Education", ['Graduate', 'Not Graduated'])
    self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0, step=1)
    property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])
    applicant_income = st.number_input("Applicant Income (₹)", min_value=0.0, value=5000.0, step=100.0)
    coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0.0, value=0.0, step=50.0)
    loan_amount = st.number_input("Loan Amount (₹ in thousands)", min_value=1.0, value=100.0, step=1.0)
    loan_term = st.number_input("Loan Term (in months)", min_value=1.0, value=360.0)
    credit_history = st.selectbox("Credit History", ['Yes', 'No'])


    if st.button("Predict Loan Approval"):
        total_income = applicant_income + coapplicant_income
        emi_ratio = loan_amount / total_income if total_income != 0 else 0

        gender = 1 if gender == "Male" else 0
        married = 1 if married == "Married" else 0
        education = 0 if education == "Graduate" else 1
        self_employed = 1 if self_employed == "Yes" else 0
        credit_history = 1.0 if credit_history == "Yes" else 0.0
        property_area_map = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
        property_area = property_area_map[property_area]
        dependents = min(int(dependents), 3)

 
        input_data = pd.DataFrame([[
            education, married, dependents, property_area,
            loan_term, credit_history, emi_ratio
        ]], columns=[
            'Education', 'Married', 'Dependents', 'Property_Area',
            'Loan_Amount_Term', 'Credit_History', 'EMI_Ratio'
        ])


        input_data = input_data.reindex(columns=model.feature_names_in_)

        prediction = model.predict(input_data)[0]
        result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"

        st.subheader("Prediction Result")
        st.success(result)
