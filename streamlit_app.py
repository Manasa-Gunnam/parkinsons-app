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
st.set_page_config(page_title="Parkinson's Detection", layout="wide")

# ================================
# SIMPLE USER DATABASE
# ================================
USER_CREDENTIALS = {
    "ManasaGunnam": "Manasa@123",
    "user": "password"
}

# ================================
# SESSION STATE
# ================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ================================
# LOGIN FUNCTION
# ================================
def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.success("Login Successful!")
            st.rerun()
        else:
            st.error("Invalid Username or Password")

# ================================
# LOGOUT FUNCTION
# ================================
def logout():
    st.session_state.logged_in = False
    st.rerun()

# ================================
# MAIN APP
# ================================
def main_app():

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "About"])

    st.sidebar.button("Logout", on_click=logout)

    if page == "Prediction":
        st.title("🧠 Parkinson's Prediction")

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

        if st.button("Predict"):
            input_data = np.array([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                                    shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                                    rpde, dfa, spread1, spread2, d2, ppe]])

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)

            confidence = np.max(probability) * 100

            if prediction[0] == 1:
                st.error(f"⚠️ Parkinson's Detected (Confidence: {confidence:.2f}%)")
            else:
                st.success(f"✅ Healthy (Confidence: {confidence:.2f}%)")

    elif page == "About":
        st.title("About Project")
        st.write("Parkinson's Disease Prediction using Machine Learning.")

# ================================
# APP ROUTING
# ================================
if not st.session_state.logged_in:
    login()
else:
    main_app()
