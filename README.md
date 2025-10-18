# 💓 Heart Disease Prediction App

Predict the likelihood of **heart disease** using patient health data. This app is built using **Python**, **Scikit-learn (SVM)**, and **Streamlit** for interactive deployment.

---

## 🌐 Live Demo
Check out the deployed app here: [Heart Disease Prediction App](https://heart-disease-prediction-ubd9wjj8huvphh4atd5ut6.streamlit.app/)

---

## 🧰 Tech Stack
- **Python 3.13**
- **Streamlit** – for interactive web app
- **Scikit-learn** – SVM model for classification
- **Pandas & Numpy** – data handling
- **Plotly** – visualizing heart disease risk

---

## 📊 Dataset
- Based on the **Kaggle Heart Failure Prediction Dataset**.
- Includes features such as Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Max Heart Rate, ECG results, Exercise-induced Angina, Oldpeak, and ST Slope.
- Features were preprocessed and one-hot encoded for categorical values.

---

## 🔍 Features
- **Predict heart disease** risk with probability score.
- **Interactive inputs** for patient health parameters.
- **Visual risk gauge** to display low, medium, and high risk.
- **Full-form descriptions** for abbreviations (ATA, NAP, ASY, etc.) for user clarity.

---

## ⚙ How It Works
1. User enters health parameters in the Streamlit app.
2. Inputs are scaled using the same **StandardScaler** used during training.
3. **SVM model** predicts the likelihood of heart disease.
4. Probability is displayed along with a **risk level** (Low, Medium, High).
5. Visual gauge updates dynamically to reflect the probability.

---

## 🏗 Project Structure
heart_disease_prediction/
│
├── app.py # Streamlit app
├── heart_disease_svm_model.pkl # Trained SVM model
├── scaler.pkl # StandardScaler object
├── feature_columns.pkl # Columns used during training
├── code.ipynb # Jupyter notebook with EDA and model building
├── .gitignore # Git ignore file
└── README.md # Project documentation

---

## 🛠️ Prerequisites
Python 3.8+
Required Python libraries:
pip install streamlit pandas numpy scikit-learn plotly joblib

---

## 🚀 Running the App Locally
Clone the repository:
git clone https://github.com/Hasitha-Voruganti/Heart-disease-prediction.git
cd Heart-disease-prediction
Run the Streamlit app:
python -m streamlit run app.py
The app will open in your browser at:
👉 http://localhost:8501
