import streamlit as st
import numpy as np
import pickle
import os
import json
import hashlib

# ================================
# LOAD MODEL
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "parkinsons_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

# ================================
# USER DATABASE FILE
# ================================
USER_DB = "users.json"

# Create file if not exists
if not os.path.exists(USER_DB):
    with open(USER_DB, "w") as f:
        json.dump({}, f)

# ================================
# PASSWORD HASH FUNCTION
# ================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ================================
# LOAD USERS
# ================================
def load_users():
    with open(USER_DB, "r") as f:
        return json.load(f)

# ================================
# SAVE USERS
# ================================
def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f)

# ================================
# SESSION STATE
# ================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ================================
# SIGNUP
# ================================
def signup():
    st.title("📝 Signup")

    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")

    if st.button("Create Account"):
        users = load_users()

        if new_user in users:
            st.error("User already exists")
        else:
            users[new_user] = hash_password(new_pass)
            save_users(users)
            st.success("Account created! Go to login.")

# ================================
# LOGIN
# ================================
def login():
    st.title("🔐 Login")

    user = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users = load_users()

        if user in users and users[user] == hash_password(password):
            st.session_state.logged_in = True
            st.session_state.username = user
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")

# ================================
# LOGOUT
# ================================
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# ================================
# MAIN APP
# ================================
def main_app():

    st.sidebar.title(f"👋 Welcome {st.session_state.username}")
    st.sidebar.button("Logout", on_click=logout)

    page = st.sidebar.radio("Navigation", ["Prediction", "About"])

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
        st.title("📘 About")
        st.write("Parkinson's Disease Prediction using ML with secure login system.")

# ================================
# AUTH FLOW
# ================================
if not st.session_state.logged_in:
    choice = st.sidebar.radio("Select", ["Login", "Signup"])
    if choice == "Login":
        login()
    else:
        signup()
else:
    main_app()
