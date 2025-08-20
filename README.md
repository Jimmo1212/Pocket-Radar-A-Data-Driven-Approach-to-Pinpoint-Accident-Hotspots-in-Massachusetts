# Pocket-Radar-A-Data-Driven-Approach-to-Pinpoint-Accident-Hotspots-in-Massachusetts
Examined factors that influence traffic accidents severity. Proposed accident simulation system that provides real-time severity heatmaps for drivers, police officers, and government officials.
# Accident Severity Prediction - PCA + Logistic Regression

## 📌 Project Overview
This project aims to predict the **severity of traffic accidents** using **PCA (Principal Component Analysis) and Logistic Regression**. The dataset used for training and validation is **US_Accidents_MA.csv**, and the model is implemented as a Flask web application.

## 📂 Project Structure
```
├── Model_EDA.ipynb            # Exploratory Data Analysis (EDA)
├── Model_Intro.ipynb          # Introduction to the dataset and initial processing
├── Model_Processing.ipynb     # Feature Engineering & Preprocessing
├── Model_training.ipynb       # Training PCA + Logistic Regression Model
├── Accidents_simulation.py    # Flask API for accident severity prediction
├── pca_logreg_accident_model.pkl # Trained model file (PCA + Logistic Regression)
├── US_Accidents_MA.csv        # Dataset (accident records)
└── templates/                 # HTML templates for Flask web app
```

## 🚀 How to Run the Project

### 1️⃣ **Install Dependencies**
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2️⃣ **Run the Flask App**
```bash
python Accidents_simulation.py
```
The application will be accessible at:
```
http://127.0.0.1:8080/
```

### 3️⃣ **Make Predictions**
- Open the web interface
- Enter relevant details like time, weather conditions, and location
- Get predicted accident severity and location mapping

## 🏗️ Model Details
- **Feature Engineering**: PCA applied for dimensionality reduction
- **Algorithm**: Logistic Regression
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

## 📊 Notebooks Description
- `Model_EDA.ipynb` → Data exploration, missing values, distributions
- `Model_Processing.ipynb` → Feature selection, encoding, normalization
- `Model_training.ipynb` → PCA transformation, Logistic Regression training

## 📝 Future Improvements
- Integrate real-time traffic & weather APIs
- Improve classification with ensemble learning
- Deploy to cloud (AWS, GCP, or Heroku)

---
📧 For any queries, contact: **jimshi1122@outlook.com**
