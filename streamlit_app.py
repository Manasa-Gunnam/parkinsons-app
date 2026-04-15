import streamlit as st
import numpy as np
import pickle
import os

# ================================
# LOAD MODEL
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "parkinsons_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Parkinson's Prediction",
    page_icon="🧠",
    layout="wide"
)

# ================================
# SIDEBAR
# ================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "About"])

# ================================
# HOME PAGE
# ================================
if page == "Home":
    st.title("🧠 Parkinson's Disease Prediction App")
    
    st.markdown("""
    ### 🔍 Overview
    This application predicts whether a person has Parkinson's Disease using Machine Learning.

    ### ⚙️ Features
    - Uses trained ML model
    - High accuracy (~95–98%)
    - Simple and interactive UI

    👉 Go to **Prediction tab** to test the model.
    """)

# ================================
# PREDICTION PAGE
# ================================
elif page == "Prediction":
    st.title("🔬 Enter Medical Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.number_input("Fo (Hz)")
        fhi = st.number_input("Fhi (Hz)")
        flo = st.number_input("Flo (Hz)")
        jitter_percent = st.number_input("Jitter (%)")
        jitter_abs = st.number_input("Jitter (Abs)")
        rap = st.number_input("RAP")
        ppq = st.number_input("PPQ")

    with col2:
        ddp = st.number_input("DDP")
        shimmer = st.number_input("Shimmer")
        shimmer_db = st.number_input("Shimmer (dB)")
        apq3 = st.number_input("APQ3")
        apq5 = st.number_input("APQ5")
        apq = st.number_input("APQ")
        dda = st.number_input("DDA")

    with col3:
        nhr = st.number_input("NHR")
        hnr = st.number_input("HNR")
        rpde = st.number_input("RPDE")
        dfa = st.number_input("DFA")
        spread1 = st.number_input("Spread1")
        spread2 = st.number_input("Spread2")
        d2 = st.number_input("D2")
        ppe = st.number_input("PPE")

    st.markdown("---")

    if st.button("🔍 Predict"):
        input_data = np.array([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                                shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                                rpde, dfa, spread1, spread2, d2, ppe]])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        st.subheader("📊 Result")

        if prediction[0] == 1:
            st.error(f"⚠️ Parkinson's Detected\n\nConfidence: {np.max(probability)*100:.2f}%")
        else:
            st.success(f"✅ Healthy\n\nConfidence: {np.max(probability)*100:.2f}%")

# ================================
# ABOUT PAGE
# ================================
elif page == "About":
    st.title("📘 About Project")

    st.markdown("""
    ### 👨‍💻 Project Details
    - **Project:** Parkinson's Disease Prediction
    - **Technology:** Machine Learning
    - **Models Used:** SVM, Random Forest, XGBoost
    - **Frontend:** Streamlit

    ### 🎯 Objective
    To predict Parkinson's Disease using voice measurements.

    ### 📊 Dataset
    Parkinson's dataset containing biomedical voice features.

    ### 🚀 Developed By
    (Add your name here)
    """)
