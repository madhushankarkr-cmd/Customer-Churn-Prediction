import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ==========================================
# LOAD MODEL, SCALER AND TRAINING COLUMNS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "models" / "churn_model.pkl"
)

scaler = joblib.load(
    BASE_DIR / "models" / "scaler.pkl"
)

training_columns = joblib.load(
    BASE_DIR / "models" / "training_columns.pkl"
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)

st.title("Customer Churn Prediction App")

st.markdown(
    "Enter customer details to predict whether the customer will churn or not."
)


# ==========================================
# INPUT FIELDS
# ==========================================

col1, col2 = st.columns(2)


# ------------------------------------------
# LEFT COLUMN
# ------------------------------------------

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    Partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    Dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (months)",
        1,
        72,
        12
    )

    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    InternetService = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )


# ------------------------------------------
# RIGHT COLUMN
# ------------------------------------------

with col2:

    OnlineBackup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    DeviceProtection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    StreamingTV = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    Contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    MonthlyCharges = st.number_input(
        "Monthly Charges",
        min_value=18.0,
        max_value=120.0,
        value=70.0
    )

    TotalCharges = st.number_input(
        "Total Charges",
        min_value=18.0,
        max_value=9000.0,
        value=1000.0
    )


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("Predict Churn"):

    # --------------------------------------
    # CREATE INPUT DATA
    # --------------------------------------

    input_dict = {

        "gender": gender,

        "SeniorCitizen": SeniorCitizen,

        "Partner": Partner,

        "Dependents": Dependents,

        "tenure": tenure,

        "PhoneService": PhoneService,

        "MultipleLines": MultipleLines,

        "InternetService": InternetService,

        "OnlineSecurity": OnlineSecurity,

        "OnlineBackup": OnlineBackup,

        "DeviceProtection": DeviceProtection,

        "TechSupport": TechSupport,

        "StreamingTV": StreamingTV,

        "StreamingMovies": StreamingMovies,

        "Contract": Contract,

        "PaperlessBilling": PaperlessBilling,

        "PaymentMethod": PaymentMethod,

        "MonthlyCharges": MonthlyCharges,

        "TotalCharges": TotalCharges
    }


    # --------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------

    input_df = pd.DataFrame([input_dict])


    # --------------------------------------
    # ONE-HOT ENCODING
    # --------------------------------------

    input_df = pd.get_dummies(input_df)


    # --------------------------------------
    # ADD MISSING TRAINING COLUMNS
    # --------------------------------------

    for col in training_columns:

        if col not in input_df.columns:

            input_df[col] = 0


    # --------------------------------------
    # REORDER COLUMNS
    # --------------------------------------

    input_df = input_df[training_columns]


    # --------------------------------------
    # SCALE INPUT
    # --------------------------------------

    input_scaled = scaler.transform(input_df)


    # --------------------------------------
    # MAKE PREDICTION
    # --------------------------------------

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]


    # --------------------------------------
    # DISPLAY RESULT
    # --------------------------------------

    if prediction == 1:

        st.error(
            "This customer is likely to **Churn**"
        )

        st.write(
            f"Churn Probability: **{probability * 100:.2f}%**"
        )

    else:

        st.success(
            "This customer is likely to **Stay**"
        )

        st.write(
            f"Churn Probability: **{probability * 100:.2f}%**"
        )