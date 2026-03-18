import streamlit as st
import pandas as pd
import requests
import json
import time

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction")

st.subheader("Input Customer Details")

# Input Widgets (raw inputs)
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    # SeniorCitizen must be 0 or 1, as defined in FastAPI's CustomerFeatures model
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], help="1 for Senior Citizen, 0 otherwise")
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    # Use float for tenure to match the type pandas often uses, though the model expects int
    tenure = st.number_input("Tenure (Months)", 0, 100, 1) 

with col2:
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col3:
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    # Ensuring these are float types
    MonthlyCharges = st.number_input("Monthly Charges", value=0.0, step=0.1)
    TotalCharges = st.number_input("Total Charges", value=0.0, step=0.1)


# PREDICTION LOGIC
if st.button("Predict Churn Probability", type="primary"):
    
    # 1. Constructed the raw input dictionary
    raw_input = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure), # Ensure tenure is integer for FastAPI model
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

    #  Prepare the payload for the API
    url = "http://127.0.0.1:8000/predict"
    payload = {"features": raw_input}

    #  Display the input being sent
    st.markdown("---")
    st.markdown("### Debugbing Info: Data Sent to API")
    st.code(json.dumps(payload, indent=4), language="json")


    # SEND TO FASTAPI 
    try:
        # Show loading state
        with st.spinner('Sending request to FastAPI...'):
            start_time = time.time()
            res = requests.post(url, json=payload, timeout=5) # Added timeout for safety
            elapsed_time = time.time() - start_time

        # Check API status code
        if res.status_code == 200:
            output = res.json()
            prob = output["probability"]
            pred_label = output["prediction_label"]

            # Display Results
            st.success(f" Prediction successful! (Took {elapsed_time:.2f} seconds)")
            st.markdown("### Prediction Results")
            st.markdown(f"## Churn Probability: **{prob:.3f}**")
            
            # Color the decision based on outcome
            if output["prediction"] == 1:
                st.markdown(f"### Decision: 🔴 **{pred_label}**")
            else:
                st.markdown(f"### Decision: 🟢 **{pred_label}**")

            st.markdown(f"*(Prediction based on threshold: {output['threshold_used']:.3f})*")
        # validation error 
        elif res.status_code == 422:
            st.error("API Validation Error (422): Check data types or values. The data sent does not match the required schema.")
            st.code(res.json(), language="json")

        else:
            st.error(f"API Error ({res.status_code}): FastAPI failed to process the request.")
            st.code(res.text, language="text")

    except requests.exceptions.ConnectionError:
        st.error(f" Connection Error! Could not connect to the FastAPI server at `{url}`.")
        st.markdown("Make sure your FastAPI application is running locally (e.g., using `uvicorn fastapi_app:app --reload`).")

    except Exception as e:
        st.error(f"An unexpected error occurred during API communication: {e}")