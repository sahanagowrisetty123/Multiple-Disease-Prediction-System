import pickle
import os
import numpy as np

# Helper function to load models safely
def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    print(f"Warning: Model not found at {model_path}.")
    return None

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    """
    Predicts Diabetes using the trained Random Forest model.
    Returns:
        String: "Disease Detected" or "No Disease Detected"
    """
    model = load_model('models/diabetes_model.sav')
    if model is None:
        return "Error: Model not found. Please train models first."
        
    # Prepare the input data array ensuring correct shape (1 row, 8 columns)
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    
    # Run the model
    prediction = model.predict(input_data)
    
    # Return result
    if prediction[0] == 1:
        return "Disease Detected"
    else:
        return "No Disease Detected"

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    """
    Predicts Heart Disease using the trained Logistic Regression model.
    Returns:
        String: "Disease Detected" or "No Disease Detected"
    """
    model = load_model('models/heart_disease_model.sav')
    if model is None:
        return "Error: Model not found. Please train models first."
        
    # Prepare input data array (1 row, 13 columns)
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        return "Disease Detected"
    else:
        return "No Disease Detected"

def predict_parkinsons(fo, fhi, flo, jitter_pct, jitter_abs, rap, ppq, jitter_ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe):
    """
    Predicts Parkinson's Disease using the trained SVM model.
    Returns:
        String: "Disease Detected" or "No Disease Detected"
    """
    model = load_model('models/parkinsons_model.sav')
    if model is None:
        return "Error: Model not found. Please train models first."
        
    # Prepare input data array (1 row, 22 columns)
    input_data = np.array([[fo, fhi, flo, jitter_pct, jitter_abs, rap, ppq, jitter_ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        return "Disease Detected"
    else:
        return "No Disease Detected"

if __name__ == "__main__":
    # Simple test code if the script is run directly
    print("Testing Prediction Functions...\n")
    print("Diabetes Test:", predict_diabetes(1, 85, 66, 29, 0, 26.6, 0.351, 31))
    print("Heart Test:", predict_heart_disease(41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2))
    print("Parkinson's Test:", predict_parkinsons(119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554, 0.01109, 0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033, 0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654))
