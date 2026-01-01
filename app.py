import streamlit as st

# =================================================
# PAGE CONFIG (MUST BE FIRST)
# =================================================
st.set_page_config(
    page_title="Enterprise GenAI BI – Churn Intelligence",
    layout="wide"
)

# =================================================
# BACKGROUND IMAGE – LIGHT / MEDIUM SHADE
# =================================================
import base64
import os

def set_bg(image_file):
    if not os.path.exists(image_file):
        st.warning(f"Background image not found: {image_file}")
        return

    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        /* MAIN BACKGROUND */
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* LIGHT OVERLAY (THIS SOFTENS THE IMAGE) */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.55);
            z-index: -1;
        }}

        /* CONTENT CONTAINER */
        .block-container {{
            background: rgba(255, 255, 255, 0.88);
            padding: 2rem;
            border-radius: 16px;
        }}

        /* HEADINGS */
        h1, h2, h3 {{
            color: #0f172a;
        }}

        /* SIDEBAR */
        section[data-testid="stSidebar"] {{
            background: rgba(255, 255, 255, 0.95);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ IMAGE NAME
set_bg("enterprise.jpeg")

# ================= IMPORTS =================
import pandas as pd
import joblib
import numpy as np
import warnings
from sqlalchemy import create_engine
import openai

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

warnings.filterwarnings("ignore")

# =================================================
# SIDEBAR – GENAI CONFIG
# =================================================
st.sidebar.title("🤖 GenAI Configuration")

openai_api_key = st.sidebar.text_input(
    "Enter OpenAI API Key",
    type="password"
)

genai_enabled = bool(openai_api_key)
if genai_enabled:
    openai.api_key = openai_api_key
    st.sidebar.success("GenAI Enabled")
else:
    st.sidebar.warning("GenAI Disabled")

# =================================================
# HEADER
# =================================================
st.title("🏦 Enterprise GenAI Business Intelligence Platform")
st.write(
    "Future churn risk prediction, explainability, "
    "SQL ingestion, customer search & AI-powered BI insights."
)

# =================================================
# LOAD MODEL
# =================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    columns = joblib.load("columns.pkl")
    return model, columns

model, columns = load_artifacts()

# =================================================
# DATA SOURCE
# =================================================
data_source = st.radio(
    "Select Data Source",
    ["Upload CSV", "Connect SQL Database"]
)

df_raw = None

# =================================================
# CSV UPLOAD
# =================================================
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("📂 Upload Customer Dataset", type=["csv"])
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)

# =================================================
# SQL INGESTION
# =================================================
elif data_source == "Connect SQL Database":
    st.subheader("🔌 Secure SQL Connection")

    db_type = st.selectbox("Database Type", ["MySQL", "PostgreSQL"])
    host = st.text_input("Host")
    port = st.text_input("Port", "3306" if db_type == "MySQL" else "5432")
    db = st.text_input("Database Name")
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")
    table = st.text_input("Table Name")

    if st.button("Connect & Load"):
        engine = create_engine(
            f"{'mysql+pymysql' if db_type=='MySQL' else 'postgresql'}://{user}:{password}@{host}:{port}/{db}"
        )
        df_raw = pd.read_sql(f"SELECT * FROM {table}", engine)
        st.success("Database connected successfully")

# =================================================
# PROCESS DATA
# =================================================
if df_raw is not None:
    try:
        st.subheader("📄 Dataset Overview")
        st.write(f"Total Records: **{len(df_raw)}**")
        st.dataframe(df_raw, use_container_width=True)

        df = df_raw.copy()

        for col in ["churn", "Churn", "Exited", "is_churn", "Churn_Risk"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="ignore")

        df = pd.get_dummies(df, drop_first=True)
        df = df.reindex(columns=columns, fill_value=0)

        churn_prob = model.predict_proba(df)[:, 1]

        results = df_raw.copy()
        results["Churn_Probability"] = churn_prob
        results["Churn_Risk"] = np.where(
            churn_prob >= 0.7, "High Risk",
            np.where(churn_prob >= 0.4, "Low Risk", "Safe")
        )

        # =================================================
        # DASHBOARD
        # =================================================
        st.subheader("📊 Executive Risk Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", len(results))
        c2.metric("Safe", (results["Churn_Risk"] == "Safe").sum())
        c3.metric("Low Risk", (results["Churn_Risk"] == "Low Risk").sum())
        c4.metric("High Risk", (results["Churn_Risk"] == "High Risk").sum())

        # =================================================
        # DOWNLOAD
        # =================================================
        st.subheader("📥 Downloads")
        st.download_button(
            "⬇ Full Results",
            results.to_csv(index=False),
            "full_predictions.csv"
        )

    except Exception as e:
        st.error("Enterprise processing error")
        st.code(str(e))

else:
    st.info("Upload CSV or connect SQL database to begin analysis.")
