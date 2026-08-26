# Multiple Disease Prediction System ⚕️

**A Final Year College Project**

## 📖 Project Description
The Multiple Disease Prediction System is a machine learning-based web application designed to predict the likelihood of a patient having **Diabetes**, **Heart Disease**, or **Parkinson's Disease**. By analyzing clinical data and health parameters, the system provides an instant prediction along with helpful health recommendations. 

This project aims to assist in early disease detection using predictive analytics. 

## 🛠️ Technologies Used
- **Python**: Core programming language.
- **Pandas**: For reading and manipulating the dataset files.
- **Scikit-Learn**: For training and building the Machine Learning models.
  - *Random Forest* for Diabetes
  - *Logistic Regression* for Heart Disease
  - *Support Vector Machine (SVM)* for Parkinson's
- **Streamlit**: To create the user-friendly web interface.

## 🚀 How to Run the Project

Follow these simple steps to run the project on your local machine:

1. **Install Required Libraries:**
   Open your command prompt or terminal inside the project folder and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Datasets:** (Only needed once)
   We have included an automated script to fetch real-world data from public GitHub repositories into the `datasets` folder:
   ```bash
   python download_datasets.py
   ```

3. **Train the Machine Learning Models:**
   Train the models on the data. They will be automatically saved in the `models` folder:
   ```bash
   python train_models.py
   ```

4. **Start the Web Application:**
   Run the Streamlit app to interact with the system:
   ```bash
   streamlit run app.py
   ```
   *(This will automatically open the web interface in your default browser!)*
