import streamlit as st
import pandas as pd
import joblib
import os
import shap # 

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
    /* Style for SHAP force plot in Streamlit */
    .stAlert > div { margin-bottom: 0px !important; }
    </style>
""", unsafe_allow_html=True)

# --- Load Model and Data for SHAP ---
if not os.path.exists('model/model.pkl'):
    st.error(" Model not found. Please run `python train_new.py` or ensure 'model/model.pkl' exists.")
    st.stop()
    
if not os.path.exists('nigerian_students_dynamic.csv'):
    st.error(" Data file for SHAP background not found. Please run 'generate_data.py'.")
    st.stop()

artifacts = joblib.load('model/model.pkl')
model = artifacts['model']
features = artifacts['features']

# Load a sample of the training data for SHAP Explainer background
# This is crucial for tree-based models like RandomForestRegressor
@st.cache_data
def load_data_for_shap():
    df = pd.read_csv('nigerian_students_dynamic.csv')
    # Use a small, representative sample for efficiency
    return df.drop('G3', axis=1).sample(100, random_state=42)

shap_background_data = load_data_for_shap()
# --- SHAP Explainer Setup ---
# Use TreeExplainer for tree-based models like Random Forest
# It's faster and more accurate for these models.
explainer = shap.TreeExplainer(model, shap_background_data)

# --- UI Layout ---
st.markdown('<p class="big-font">🎓 CGPA Predictor</p>', unsafe_allow_html=True)

with st.form("simple_form"):
    
    # Section 1: Hard Data
    st.subheader("1. The Metrics")
    c1, c2 = st.columns(2)
    with c1:
        # Use st.session_state for persistence, setting initial value if not present
        if 'G1' not in st.session_state: st.session_state.G1 = 3.5
        G1 = st.number_input("Year 1 CGPA", 0.0, 5.0, value=st.session_state.G1, key='G1_input')
        
        if 'failures' not in st.session_state: st.session_state.failures = 0
        failures = st.number_input("Carryovers", 0, 10, value=st.session_state.failures, key='failures_input')
    with c2:
        if 'G2' not in st.session_state: st.session_state.G2 = 3.8
        G2 = st.number_input("Year 2 CGPA", 0.0, 5.0, value=st.session_state.G2, key='G2_input')
        
        if 'absences' not in st.session_state: st.session_state.absences = 2
        absences = st.number_input("Class Absences", 0, 50, value=st.session_state.absences, key='absences_input')

    st.markdown("---")

    # Section 2: Soft Data
    st.subheader("2. The Lifestyle")
    c3, c4 = st.columns(2)
    with c3:
        if 'studytime' not in st.session_state: st.session_state.studytime = 3
        studytime = st.select_slider("Weekly Study Hours", options=[1, 2, 3, 4],
                                format_func=lambda x: {1:"<2h", 2:"2-5h", 3:"5-10h", 4:">10h"}[x],
                                value=st.session_state.studytime, key='studytime_input')
        
        if 'health' not in st.session_state: st.session_state.health = 4
        health = st.slider("Energy / Health Level", 1, 5, value=st.session_state.health, key='health_input', help="1=Burned Out, 5=Full Energy")
    with c4:
        if 'goout' not in st.session_state: st.session_state.goout = 3
        goout = st.slider("Social / Partying Frequency", 1, 5, value=st.session_state.goout, key='goout_input', help="1=Hermit, 5=Every Night")
        
        col_check1, col_check2 = st.columns(2)
        with col_check1:
            if 'higher' not in st.session_state: st.session_state.higher = True
            higher = st.checkbox("Masters Plan?", value=st.session_state.higher, key='higher_input')
        with col_check2:
            if 'activities' not in st.session_state: st.session_state.activities = True
            activities = st.checkbox("Extra-curriculars?", value=st.session_state.activities, key='activities_input')

    # Submit
    submitted = st.form_submit_button("Calculate Final CGPA")

# --- Prediction and SHAP Logic ---
if submitted:
    
    # 1. Map inputs to dataframe
    input_data = {
        'G1': G1, 'G2': G2, 'failures': failures, 'absences': absences,
        'studytime': studytime, 'health': health, 'goout': goout,
        'higher': 1 if higher else 0,
        'activities': 1 if activities else 0
    }
    
    input_df = pd.DataFrame([input_data])
    final_input = input_df[features]

    # 2. Predict
    with st.spinner('Calculating prediction and explanations...'):
        prediction = model.predict(final_input)[0]
        final_cgpa = min(max(prediction, 0.0), 5.0) # Clamp between 0 and 5

        # 3. Calculate SHAP Values
        # Note: We pass the single input row (final_input) to the explainer
        shap_values = explainer.shap_values(final_input)

    # 4. Verdict Colors and Result Box
    if final_cgpa >= 4.5: color, msg = "#28a745", "First Class Honours "
    elif final_cgpa >= 3.5: color, msg = "#17a2b8", "Second Class Upper (2:1) "
    elif final_cgpa >= 2.4: color, msg = "#ffc107", "Second Class Lower (2:2) "
    else: color, msg = "#dc3545", "Third Class / Fail / Advice to drop out"

    st.markdown(f"""
        <div class="result-box">
            <h4 style="color: #555; margin:0;">Predicted Final CGPA</h4>
            <h1 style="font-size: 4rem; color: {color}; margin: 10px 0;">{final_cgpa:.2f}</h1>
            <h3 style="color: {color};">{msg}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # 5. SHAP Explanation Section
    st.subheader("💡 Why this Prediction?")
    st.info("The chart below shows how each input factor pushed the final CGPA prediction higher (red) or lower (blue) from the average predicted CGPA.")
    
    # Use st.pyplot to display the SHAP plot
    # We wrap the plot generation in a figure to display correctly in Streamlit
    fig = shap.force_plot(
        explainer.expected_value, 
        shap_values[0,:], 
        final_input.iloc[0,:], 
        matplotlib=True, 
        show=False,
        link='logit' # Use logit for better color separation if needed, or 'identity'
    )
    
    st.pyplot(fig, bbox_inches='tight')

    st.markdown("---")
    
    st.subheader("Top Influencing Factors")
    
    # Create a summary bar plot for a slightly different view
    # This requires running the explanation for the entire background data
    # st.pyplot(shap.summary_plot(shap_values, final_input, plot_type="bar", show=False), bbox_inches='tight')
    
    # Simple list of top 3 features for non-technical users
    # Combine feature names and SHAP values
    shap_df = pd.DataFrame({
        'Feature': final_input.columns,
        'SHAP_Value': shap_values[0]
    })
    
    # Sort by absolute SHAP value to find most influential
    shap_df['Abs_SHAP'] = shap_df['SHAP_Value'].abs()
    top_3 = shap_df.sort_values(by='Abs_SHAP', ascending=False).head(3)
    
    st.markdown("The **top 3** most influential factors for your specific input were:")
    
    for index, row in top_3.iterrows():
        feature_name = row['Feature']
        influence = "increased" if row['SHAP_Value'] > 0 else "decreased"
        influence_color = "red" if row['SHAP_Value'] > 0 else "blue"
        
        # Nicer display of feature names
        display_name = {
            'G1': 'Year 1 CGPA', 'G2': 'Year 2 CGPA', 'failures': 'Carryovers', 
            'absences': 'Class Absences', 'studytime': 'Weekly Study Hours', 
            'health': 'Energy / Health Level', 'goout': 'Social / Partying Frequency',
            'higher': 'Masters Plan', 'activities': 'Extra-curriculars'
        }.get(feature_name, feature_name)
        
        st.markdown(f"* **{display_name}**: which generally <span style='color:{influence_color}; font-weight:bold;'>{influence}</span> the predicted CGPA.", unsafe_allow_html=True)