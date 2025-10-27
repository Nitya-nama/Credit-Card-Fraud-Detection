import streamlit as st
import pandas as pd
import joblib
import io
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# 🪪 Page Config
# -------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# -------------------------------
# 🎯 App Header
# -------------------------------
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("""
Detect **fraudulent transactions** from credit card data using a pre-trained ML model.  
Upload your CSV file, view predictions, and explore interactive visual insights.  
---
""")

# -------------------------------
# 🧠 Load Model and Scaler
# -------------------------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("fraud_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model or scaler: {e}")
        st.stop()

model, scaler = load_assets()

# -------------------------------
# 📂 Sidebar - File Upload
# -------------------------------
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# -------------------------------
# 🧩 Main Section
# -------------------------------
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(data.head(), use_container_width=True)

    # Drop existing 'Class' column if present
    if "Class" in data.columns:
        data = data.drop("Class", axis=1)

    # Numeric filtering
    numeric_data = data.select_dtypes(include=["number"])
    if numeric_data.shape[1] != data.shape[1]:
        st.warning("⚠️ Non-numeric columns detected — ignored during scaling.")

    # Scaling and prediction
    data_scaled = scaler.transform(numeric_data)
    predictions = model.predict(data_scaled)
    data["Prediction"] = predictions
    data["Prediction"] = data["Prediction"].map({0: "Legit", 1: "Fraud"})

    # -------------------------------
    # 📊 Metrics Section
    # -------------------------------
    fraud_count = (data["Prediction"] == "Fraud").sum()
    legit_count = (data["Prediction"] == "Legit").sum()
    fraud_rate = (fraud_count / len(data)) * 100

    st.markdown("## 📈 Prediction Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Legit Transactions", f"{legit_count:,}")
    c2.metric("🚨 Fraudulent Transactions", f"{fraud_count:,}")
    c3.metric("💥 Fraud Rate (%)", f"{fraud_rate:.3f}%")

    st.markdown("---")

    # -------------------------------
    # 🧮 Fraud vs Legit Chart
    # -------------------------------
    st.markdown("## 🧩 Fraud vs Legit Distribution")

    fraud_ratio = data["Prediction"].value_counts(normalize=True).reset_index()
    fraud_ratio.columns = ["Transaction Type", "Percentage"]
    fig_pie = px.pie(
        fraud_ratio,
        names="Transaction Type",
        values="Percentage",
        color="Transaction Type",
        color_discrete_map={"Fraud": "#E74C3C", "Legit": "#27AE60"},
        hole=0.4,
        title="Fraud vs Legit Transactions",
    )
    fig_pie.update_traces(textinfo="percent+label", pull=[0.1, 0])
    st.plotly_chart(fig_pie, use_container_width=True)

    # -------------------------------
    # 💵 Transaction Amount Insights
    # -------------------------------
    if "Amount" in data.columns:
        st.markdown("## 💵 Transaction Amount Analysis")

        fig_box = px.box(
            data,
            x="Prediction",
            y="Amount",
            color="Prediction",
            color_discrete_map={"Fraud": "#E74C3C", "Legit": "#27AE60"},
            title="Transaction Amounts by Prediction Type",
            points="all",
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("⚠️ No 'Amount' column found — skipping amount visualizations.")

    # -------------------------------
    # 🔎 Feature Importance (if available)
    # -------------------------------
    if hasattr(model, "feature_importances_"):
        st.markdown("## 🔎 Feature Importance")
        feat_imp = pd.DataFrame({
            "Feature": numeric_data.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig_imp = px.bar(
            feat_imp.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            title="Top 10 Most Important Features",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # -------------------------------
    # 📥 Download Section
    # -------------------------------
    st.markdown("---")
    st.markdown("### 📦 Download Predictions")

    csv_buffer = io.BytesIO()
    csv_buffer.write(data.to_csv(index=False).encode("utf-8"))
    csv_buffer.seek(0)

    st.download_button(
        label="📥 Download Fraud Predictions CSV",
        data=csv_buffer,
        file_name="fraud_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:
    st.info("👈 Upload a CSV file from the sidebar to start detecting frauds.")

# -------------------------------
# 🌙 Footer
# -------------------------------
st.markdown("""
---
<p style="text-align:center; color:grey;">
Made with ❤️ using Streamlit | Credit Card Fraud Detection
</p>
""", unsafe_allow_html=True)
