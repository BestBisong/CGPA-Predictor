import streamlit as st
import pandas as pd
import joblib
import os

# --- Config & Style ---
st.set_page_config(page_title="CGPA Predictor", page_icon="📈")

st.markdown("""
    <style>
    .big-font { font-size: 30px !important; font-weight: bold; color: #2E86C1; }
    .result-box { 
        background-color: #f0f2f6; padding: 20px; border-radius: 10px; 
        text-align: center; margin-top: 20px; border-left: 5px solid #2E86C1;
    }
    .stButton>button { 
        width: 100%; background-color: #2E86C1; color: white; height: 3em; border-radius: 8px; font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model ---
if not os.path.exists('model/model.pkl'):
    st.error("⚠️ Model not found. Please run `python train_new.py`.")
    st.stop()

artifacts = joblib.load('model/model.pkl')
model = artifacts['model']
features = artifacts['features']

# --- UI Layout ---
st.markdown('<p class="big-font">🎓 CGPA Predictor</p>', unsafe_allow_html=True)

with st.form("simple_form"):
    
    # Section 1: Hard Data
    st.subheader("1. The Metrics")
    c1, c2 = st.columns(2)
    with c1:
        G1 = st.number_input("Year 1 CGPA", 0.0, 5.0, 3.5)
        failures = st.number_input("Carryovers", 0, 10, 0)
    with c2:
        G2 = st.number_input("Year 2 CGPA", 0.0, 5.0, 3.8)
        absences = st.number_input("Class Absences", 0, 50, 2)

    st.markdown("---")

    # Section 2: Soft Data
    st.subheader("2. The Lifestyle")
    c3, c4 = st.columns(2)
    with c3:
        studytime = st.select_slider("Weekly Study Hours", options=[1, 2, 3, 4], 
                                   format_func=lambda x: {1:"<2h", 2:"2-5h", 3:"5-10h", 4:">10h"}[x])
        health = st.slider("Energy / Health Level", 1, 5, 4, help="1=Burned Out, 5=Full Energy")
    with c4:
        goout = st.slider("Social / Partying Frequency", 1, 5, 3, help="1=Hermit, 5=Every Night")
        col_check1, col_check2 = st.columns(2)
        with col_check1:
            higher = st.checkbox("Masters Plan?", value=True)
        with col_check2:
            activities = st.checkbox("Extra-curriculars?", value=True)

    # Submit
    submitted = st.form_submit_button("Calculate Final CGPA")

if submitted:
    # Map inputs to dataframe
    input_data = {
        'G1': G1, 'G2': G2, 'failures': failures, 'absences': absences,
        'studytime': studytime, 'health': health, 'goout': goout,
        'higher': 1 if higher else 0,
        'activities': 1 if activities else 0
    }
    
    # Predict
    input_df = pd.DataFrame([input_data])
    
    # Ensure column order matches training
    final_input = input_df[features]
    
    prediction = model.predict(final_input)[0]
    final_cgpa = min(max(prediction, 0.0), 5.0) # Clamp between 0 and 5

    # Verdict Colors
    if final_cgpa >= 4.5: color, msg = "#28a745", "First Class Honours 🏆"
    elif final_cgpa >= 3.5: color, msg = "#17a2b8", "Second Class Upper (2:1) 🥈"
    elif final_cgpa >= 2.4: color, msg = "#ffc107", "Second Class Lower (2:2) ⚠️"
    else: color, msg = "#dc3545", "Third Class / Fail 🛑"

    st.markdown(f"""
        <div class="result-box">
            <h4 style="color: #555; margin:0;">Predicted Final CGPA</h4>
            <h1 style="font-size: 4rem; color: {color}; margin: 10px 0;">{final_cgpa:.2f}</h1>
            <h3 style="color: {color};">{msg}</h3>
        </div>
    """, unsafe_allow_html=True)