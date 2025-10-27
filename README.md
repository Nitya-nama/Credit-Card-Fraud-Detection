# 💳 Credit Card Fraud Detection Dashboard  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)
![Scikit-learn](https://img.shields.io/badge/ML-ScikitLearn-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

A **Machine Learning–based web app** that detects fraudulent credit card transactions using a **Logistic Regression** model trained on the Kaggle Credit Card dataset.  
The app allows users to upload transaction data, visualize fraud statistics, and download prediction results — all within a neat and interactive Streamlit dashboard.

---

## 🧠 Project Overview  

This project is divided into two main parts:

### **1️⃣ Model Training (`train_model.py`)**
- Loads the Kaggle credit card fraud dataset  
- Scales features using `StandardScaler`  
- Trains a **Logistic Regression** classifier  
- Saves both the model and the scaler as `.pkl` files  

### **2️⃣ Streamlit App (`app.py`)**
- Loads the pre-trained model and scaler  
- Accepts a CSV upload  
- Predicts whether each transaction is **Fraudulent** or **Legit**  
- Displays fraud statistics with **pie, box, and violin plots**  
- Allows downloading the results as a CSV  

---

## 📁 Project Structure  

credit-card-fraud-detection/
│
├── creditcard.csv # Dataset (from Kaggle)
├── train_model.py # Model training script
├── app.py # Streamlit dashboard
├── fraud_model.pkl # Trained model
├── scaler.pkl # Saved scaler
├── requirements.txt # Dependencies list
└── README.md # Project documentation


---

## ⚙ Installation  

<details>
<summary>📦 Step-by-step setup</summary>

## 1️⃣ Clone the repository  

git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection

## 2️⃣ Create a virtual environment (recommended)

python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

## 3️⃣ Install dependencies
pip install -r requirements.txt


If you don’t have the file, install manually:

pip install streamlit pandas scikit-learn matplotlib seaborn joblib

</details>

## 📊 Dataset

Use the Kaggle dataset:
🔗 Credit Card Fraud Detection – Kaggle

After downloading, place it in your project folder as:

creditcard.csv

## 🧩 Model Training

Run the following command to train and save your model:

python train_model.py


You should see:

✅ Model trained successfully!
💾 Model and Scaler saved successfully!


This creates:

fraud_model.pkl

scaler.pkl

## 🖥 Running the Streamlit Dashboard

Once the model is ready, start the app:

streamlit run app.py


Then open the provided local URL (usually http://localhost:8501
).

## 🧭 How to Use

1️⃣ Upload a CSV file with transaction data (same format as creditcard.csv)
2️⃣ The app predicts Fraudulent or Legit for each transaction
3️⃣ View:

Prediction summary (fraud vs legit counts)

Pie chart showing fraud percentage

Box and violin plots comparing transaction amounts
4️⃣ Download the predictions as a CSV

## 📊 Example Output

Fraud Detection Summary:
✅ Legit: 284,315
🚨 Fraud: 492

Interactive Visuals:

📈 Pie chart for fraud ratio

📦 Box & 🎻 Violin plots for transaction amounts

## CSV Download:

Includes a “Prediction” column with results

## 🧰 Tech Stack

🐍 Python
📊 Pandas, NumPy, Scikit-learn
🎨 Matplotlib, Seaborn
🌐 Streamlit (for interactive dashboard)

## 🚀 Future Enhancements

Add Plotly interactive visualizations

Integrate real-time fraud detection API

Try RandomForest or XGBoost models

Deploy to Streamlit Cloud, Render, or AWS

## 🧾 Relation to Big Data Analytics

This project relates to Big Data Analytics as it processes and analyzes large volumes of transactional data to identify fraudulent patterns.
In real-world banking systems, millions of transactions are generated every day, exhibiting the volume, velocity, and variety of Big Data.
By applying machine learning–based predictive analytics, the system detects anomalies that indicate fraud.
The dashboard visualizes insights from massive data, demonstrating the power of Big Data–driven decision-making.

## 👩‍💻 Author

Nitya Nama
Machine Learning & Data Science Enthusiast
