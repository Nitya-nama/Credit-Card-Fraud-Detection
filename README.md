### 💳 Credit Card Fraud Detection Dashboard ###

A machine learning–based web app that detects fraudulent credit card transactions using a Logistic Regression model trained on the popular Kaggle credit card dataset.
The app lets you upload transaction data, visualize fraud statistics, and download the prediction results.


---

🧠 Project Overview

This project is divided into two main parts:

1. Model Training (train_model.py)

Loads the Kaggle credit card fraud dataset

Scales features using StandardScaler

Trains a Logistic Regression classifier

Saves both the model and the scaler as .pkl files



2. Streamlit App (app.py)

Loads the pre-trained model and scaler

Accepts a CSV file upload

Predicts whether each transaction is Fraudulent or Legit

Displays statistics, donut/pie charts, and box/violin plots

Allows users to download the results as a CSV





---

📁 Project Structure

credit-card-fraud-detection/
│
├── creditcard.csv               # Dataset (from Kaggle)
├── train_model.py               # Model training script
├── app.py                       # Streamlit dashboard
├── fraud_model.pkl              # Saved trained model
├── scaler.pkl                   # Saved scaler
└── README.md                    # Project documentation


---

⚙ Installation

1. Clone the repository

git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection

2. Create a virtual environment (recommended)

python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

3. Install dependencies

pip install -r requirements.txt

If you don’t have a requirements.txt, here are the needed packages:

pip install streamlit pandas scikit-learn matplotlib seaborn joblib


---

📊 Dataset

Use the Kaggle Credit Card Fraud Detection dataset:
🔗 https://www.kaggle.com/mlg-ulb/creditcardfraud

After downloading, place the file in your project folder as creditcard.csv.


---

🧩 Training the Model

Run this once to train and save your model:

python train_model.py

You should see:

✅ Model trained successfully!
💾 Model and scaler saved successfully!

This will create two files:

fraud_model.pkl

scaler.pkl



---

🖥 Running the Streamlit App

Once the model is ready, launch the web dashboard:

streamlit run app.py

Then open the local URL it shows (usually http://localhost:8501).


---

🧭 How to Use

1. Upload a transaction CSV file (same format as creditcard.csv)


2. Wait for predictions to load


3. Explore:

Prediction summary (total fraud vs legit)

Donut and Pie charts for fraud distribution

Box & Violin plots for transaction amount comparison



4. Download results as a CSV




---

🧾 Example Output

Fraud Detection Summary

✅ Legit: 284,315

🚨 Fraud: 492


Interactive charts showing the proportion of frauds

CSV download of predictions



---

🧰 Tech Stack

Python

Pandas, NumPy, scikit-learn

Matplotlib, Seaborn

Streamlit (for frontend dashboard)



---

📦 Future Improvements

Add Plotly interactive charts

Integrate real-time detection API

Try advanced models like RandomForest or XGBoost

Deploy on Streamlit Cloud / AWS / Render



---

👩‍💻 Author

Nitya
Machine Learning & Data Science Enthusiast


---

Would you like me to generate a short requirements.txt and train_model.py filename version too so you can directly zip and run the whole project setup?
