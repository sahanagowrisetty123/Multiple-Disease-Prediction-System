import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_diabetes_data():
    """
    Loads, cleans, and splits the Diabetes dataset.
    Target variable: 'Outcome' (1 for Diabetes, 0 for Normal)
    """
    print("--- Preprocessing Diabetes Dataset ---")
    
    # 1. Load dataset using pandas
    filepath = 'datasets/diabetes.csv'
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}. Please run download_datasets.py first.")
        return None, None, None, None
        
    df = pd.read_csv(filepath)
    print(f"Successfully loaded {filepath} with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # 2. Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Found {missing_values} missing values. Handling them by dropping rows...")
        df = df.dropna() # Simple approach for beginners: dropping missing rows
    else:
        print("No missing values found.")
        
    # 3. Remove unnecessary columns
    # The diabetes dataset features are all relevant medical metrics, so no columns to drop here.
    
    # 4. Split dataset into input features (X) and target variable (y)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # 5. Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split successful: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.\n")
    
    return X_train, X_test, y_train, y_test

def preprocess_heart_data():
    """
    Loads, cleans, and splits the Heart Disease dataset.
    Target variable: 'target' (1 for Heart Disease, 0 for Normal)
    """
    print("--- Preprocessing Heart Disease Dataset ---")
    
    filepath = 'datasets/heart.csv'
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}.")
        return None, None, None, None
        
    # 1. Load dataset
    df = pd.read_csv(filepath)
    print(f"Successfully loaded {filepath} with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # 2. Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        df = df.dropna()
        print(f"Dropped rows with missing values.")
    else:
        print("No missing values found.")
        
    # 3. Remove unnecessary columns 
    # Current heart.csv has only relevant columns, but this is where we'd drop IDs etc.
    
    # 4. Split into X and y
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 5. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split successful: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.\n")
    
    return X_train, X_test, y_train, y_test

def preprocess_parkinsons_data():
    """
    Loads, cleans, and splits the Parkinson's Disease dataset.
    Target variable: 'status' (1 for Parkinson's, 0 for Normal)
    """
    print("--- Preprocessing Parkinson's Disease Dataset ---")
    
    filepath = 'datasets/parkinsons.csv'
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}.")
        return None, None, None, None
        
    # 1. Load dataset
    df = pd.read_csv(filepath)
    print(f"Successfully loaded {filepath} with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # 2. Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        df = df.dropna()
        print(f"Dropped rows with missing values.")
    else:
        print("No missing values found.")
        
    # 3. Remove unnecessary columns
    # The 'name' column contains the patient's ID string, which is useless for machine learning
    print("Dropping unnecessary 'name' column (Text ID data).")
    df = df.drop(columns=['name'])
    
    # 4. Split into X and y
    X = df.drop(columns=['status'])
    y = df['status']
    
    # 5. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split successful: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.\n")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("=========================================")
    print("Running Dataset Preprocessing Subroutines")
    print("=========================================\n")
    
    # Test our functions
    diab_X_train, diab_X_test, diab_y_train, diab_y_test = preprocess_diabetes_data()
    heart_X_train, heart_X_test, heart_y_train, heart_y_test = preprocess_heart_data()
    park_X_train, park_X_test, park_y_train, park_y_test = preprocess_parkinsons_data()
    
    print("All datasets preprocessed successfully!")
