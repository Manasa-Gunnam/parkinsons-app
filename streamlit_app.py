import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("parkinsons_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Parkinson's Disease Prediction")

st.write("Enter medical values:")

# Inputs (must match dataset features)
fo = st.number_input("MDVP:Fo(Hz)")
fhi = st.number_input("MDVP:Fhi(Hz)")
flo = st.number_input("MDVP:Flo(Hz)")
jitter_percent = st.number_input("MDVP:Jitter(%)")
jitter_abs = st.number_input("MDVP:Jitter(Abs)")
rap = st.number_input("MDVP:RAP")
ppq = st.number_input("MDVP:PPQ")
ddp = st.number_input("Jitter:DDP")
shimmer = st.number_input("MDVP:Shimmer")
shimmer_db = st.number_input("MDVP:Shimmer(dB)")
apq3 = st.number_input("Shimmer:APQ3")
apq5 = st.number_input("Shimmer:APQ5")
apq = st.number_input("MDVP:APQ")
dda = st.number_input("Shimmer:DDA")
nhr = st.number_input("NHR")
hnr = st.number_input("HNR")
rpde = st.number_input("RPDE")
dfa = st.number_input("DFA")
spread1 = st.number_input("spread1")
spread2 = st.number_input("spread2")
d2 = st.number_input("D2")
ppe = st.number_input("PPE")

if st.button("Predict"):
    input_data = np.array([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                            shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                            rpde, dfa, spread1, spread2, d2, ppe]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Parkinson's Disease Detected")
    else:
        st.success("Healthy")
