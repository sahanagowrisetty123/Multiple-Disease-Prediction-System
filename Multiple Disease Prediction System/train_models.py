import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import os
import preprocess_datasets

# Create folders if they don't exist
os.makedirs('datasets', exist_ok=True)
os.makedirs('models', exist_ok=True)

def evaluate_and_save(model, X_test, y_test, model_path, disease_name):
    """Evaluates the model and saves it to the models directory."""
    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy for {disease_name}: {accuracy * 100:.2f}%")
    
    # Save the trained model
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved successfully: {model_path}\n")

if __name__ == "__main__":
    print("------------------------------------------")
    print("Starting Model Training Process...")
    print("------------------------------------------\n")
    
    # 1. Train model for Diabetes (Random Forest)
    d_X_train, d_X_test, d_y_train, d_y_test = preprocess_datasets.preprocess_diabetes_data()
    if d_X_train is not None:
        print("Training Random Forest model for Diabetes...")
        diabetes_model = RandomForestClassifier(random_state=42)
        diabetes_model.fit(d_X_train, d_y_train)
        evaluate_and_save(diabetes_model, d_X_test, d_y_test, 'models/diabetes_model.sav', 'Diabetes')
    
    # 2. Train model for Heart Disease (Logistic Regression)
    h_X_train, h_X_test, h_y_train, h_y_test = preprocess_datasets.preprocess_heart_data()
    if h_X_train is not None:
        print("Training Logistic Regression model for Heart Disease...")
        # max_iter increased to ensure convergence for medical datasets
        heart_model = LogisticRegression(random_state=42, max_iter=1000)
        heart_model.fit(h_X_train, h_y_train)
        evaluate_and_save(heart_model, h_X_test, h_y_test, 'models/heart_disease_model.sav', 'Heart Disease')
    
    # 3. Train model for Parkinson's Disease (Support Vector Machine)
    p_X_train, p_X_test, p_y_train, p_y_test = preprocess_datasets.preprocess_parkinsons_data()
    if p_X_train is not None:
        print("Training Support Vector Machine (SVM) model for Parkinson's...")
        # using a linear kernel which generally performs well for simple text/metric datasets
        parkinsons_model = SVC(kernel='linear', random_state=42)
        parkinsons_model.fit(p_X_train, p_y_train)
        evaluate_and_save(parkinsons_model, p_X_test, p_y_test, 'models/parkinsons_model.sav', "Parkinson's Disease")
    
    print("------------------------------------------")
    print("All models have been trained and saved successfully!")
    print("------------------------------------------")
