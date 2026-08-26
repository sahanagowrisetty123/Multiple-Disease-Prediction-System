import streamlit as st
import prediction_functions

# Set page configuration for the web app
st.set_page_config(page_title="Multiple Disease Prediction System", page_icon="⚕️", layout="wide")

# Sidebar for navigation
st.sidebar.title("Disease Prediction System")
st.sidebar.write("Select a disease to predict:")
app_mode = st.sidebar.radio("Navigation", 
                            ["Diabetes", "Heart Disease", "Parkinson's Disease"])

# --- Diabetes Prediction Section (Random Forest) ---
if app_mode == "Diabetes":
    st.title("🩸 Diabetes Prediction using Random Forest")
    st.write("Please enter the clinical data to check for Diabetes.")
    
    # Diabetes dataset has 8 features
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input("Pregnancies", value=0.0)
        SkinThickness = st.number_input("SkinThickness", value=0.0)
        DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", value=0.0)
    with col2:
        Glucose = st.number_input("Glucose", value=0.0)
        Insulin = st.number_input("Insulin", value=0.0)
        Age = st.number_input("Age", value=0.0)
    with col3:
        BloodPressure = st.number_input("BloodPressure", value=0.0)
        BMI = st.number_input("BMI", value=0.0)
        
    # Prediction button
    if st.button("Predict Diabetes"):
        result = prediction_functions.predict_diabetes(
            Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        )
        
        st.divider()
        if result == "Disease Detected":
            st.error("⚠️ The model predicts that the patient **HAS** Diabetes.")
            st.warning("💡 **Recommendation:** Maintain a balanced diet, limit sugar intake, exercise regularly, and consult a doctor for a professional diagnosis.")
        elif result == "No Disease Detected":
            st.success("✅ The model predicts that the patient does **NOT** have Diabetes.")
            st.info("💡 **Recommendation:** Continue maintaining a healthy lifestyle, stay active, and eat a balanced diet to keep your blood sugar in check.")
        else:
            st.warning(result)

# --- Heart Disease Prediction Section (Logistic Regression) ---
elif app_mode == "Heart Disease":
    st.title("🫀 Heart Disease Prediction using Logistic Regression")
    st.write("Please enter the clinical data to check for Heart Disease.")
    
    # Heart disease dataset has 13 features
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", value=0.0)
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", value=0.0)
        restecg = st.number_input("Resting ECG results (restecg)", value=0.0)
        oldpeak = st.number_input("ST depression (oldpeak)", value=0.0)
        thal = st.number_input("Thal (thal)", value=0.0)
    with col2:
        sex = st.number_input("Sex (1=M, 0=F)", value=0.0)
        chol = st.number_input("Cholesterol (chol)", value=0.0)
        thalach = st.number_input("Max Heart Rate (thalach)", value=0.0)
        slope = st.number_input("Slope of ST segment (slope)", value=0.0)
    with col3:
        cp = st.number_input("Chest Pain Type (cp)", value=0.0)
        fbs = st.number_input("Fasting Blood Sugar > 120 (fbs)", value=0.0)
        exang = st.number_input("Exercise Induced Angina (exang)", value=0.0)
        ca = st.number_input("Number of major vessels (ca)", value=0.0)
        
    if st.button("Predict Heart Disease"):
        result = prediction_functions.predict_heart_disease(
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        )
        
        st.divider()
        if result == "Disease Detected":
            st.error("⚠️ The model predicts that the patient **HAS** Heart Disease.")
            st.warning("💡 **Recommendation:** Avoid smoking, reduce salt and saturated fats, manage stress, and seek immediate cardiology consultation.")
        elif result == "No Disease Detected":
            st.success("✅ The model predicts that the patient does **NOT** have Heart Disease.")
            st.info("💡 **Recommendation:** Keep your heart healthy by exercising daily, eating whole foods, and managing your cholesterol and blood pressure.")
        else:
            st.warning(result)

# --- Parkinson's Disease Prediction Section (SVM) ---
elif app_mode == "Parkinson's Disease":
    st.title("🧠 Parkinson's Disease Prediction using SVM")
    st.write("Please enter the clinical voice measurements data to check for Parkinson's Disease.")
    
    # Parkinson's dataset has 22 features (we dropped 'name' and 'status')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", value=0.0)
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", value=0.0)
        shimmer = st.number_input("MDVP:Shimmer", value=0.0)
        apq5 = st.number_input("Shimmer:APQ5", value=0.0)
        hnr = st.number_input("HNR", value=0.0)
        spread2 = st.number_input("spread2", value=0.0)
    with col2:
        fhi = st.number_input("MDVP:Fhi(Hz)", value=0.0)
        rap = st.number_input("MDVP:RAP", value=0.0)
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", value=0.0)
        apq = st.number_input("MDVP:APQ", value=0.0)
        rpde = st.number_input("RPDE", value=0.0)
        d2 = st.number_input("D2", value=0.0)
    with col3:
        flo = st.number_input("MDVP:Flo(Hz)", value=0.0)
        ppq = st.number_input("MDVP:PPQ", value=0.0)
        apq3 = st.number_input("Shimmer:APQ3", value=0.0)
        dda = st.number_input("Shimmer:DDA", value=0.0)
        dfa = st.number_input("DFA", value=0.0)
        ppe = st.number_input("PPE", value=0.0)
    with col4:
        jitter_pct = st.number_input("MDVP:Jitter(%)", value=0.0)
        jitter_ddp = st.number_input("Jitter:DDP", value=0.0)
        nhr = st.number_input("NHR", value=0.0)
        spread1 = st.number_input("spread1", value=0.0)
        
    if st.button("Predict Parkinson's"):
        result = prediction_functions.predict_parkinsons(
            fo, fhi, flo, jitter_pct, jitter_abs, rap, ppq, jitter_ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe
        )
        
        st.divider()
        if result == "Disease Detected":
            st.error("⚠️ The model predicts that the patient **HAS** Parkinson's Disease.")
            st.warning("💡 **Recommendation:** Engage in physical therapies, focus on speech and mobility exercises, and consult a neurologist for a comprehensive care plan.")
        elif result == "No Disease Detected":
            st.success("✅ The model predicts that the patient does **NOT** have Parkinson's Disease.")
            st.info("💡 **Recommendation:** Maintain neurological health by staying mentally active, eating antioxidant-rich foods, and exercising regularly.")
        else:
            st.warning(result)

# Footer section
st.sidebar.divider()
st.sidebar.info("Developed for Final Year College Project")
