import streamlit as st

# MUST be first Streamlit command
st.set_page_config(
    page_title="Enterprise GenAI BI – Churn Intelligence",
    layout="wide"
)

# ================= IMPORTS =================
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sqlalchemy import create_engine
import openai

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

warnings.filterwarnings("ignore")

# =================================================
# SIDEBAR – GENAI CONFIG (TOP RIGHT)
# =================================================
st.sidebar.title("🤖 GenAI Configuration")

openai_api_key = st.sidebar.text_input(
    "Enter OpenAI API Key",
    type="password",
    help="Required to enable GenAI features"
)

genai_enabled = bool(openai_api_key)
if genai_enabled:
    openai.api_key = openai_api_key
    st.sidebar.success("GenAI Enabled")
else:
    st.sidebar.warning("GenAI Disabled (API key required)")

# =================================================
# HEADER
# =================================================
st.title("🏦 Enterprise GenAI Business Intelligence Platform")
st.write(
    "Future churn risk prediction, explainability, "
    "SQL ingestion, customer search & AI-powered BI insights."
)

# =================================================
# LOAD MODEL & METADATA
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
    host = st.text_input("Host", "localhost")
    port = st.text_input("Port", "3306" if db_type == "MySQL" else "5432")
    db = st.text_input("Database Name")
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")
    table = st.text_input("Table Name")

    if st.button("Connect & Load"):
        if db_type == "MySQL":
            engine = create_engine(
                f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
            )
        else:
            engine = create_engine(
                f"postgresql://{user}:{password}@{host}:{port}/{db}"
            )

        df_raw = pd.read_sql(f"SELECT * FROM {table}", engine)
        st.success("Database connected successfully")

# =================================================
# PROCESS DATA
# =================================================
if df_raw is not None:
    try:
        df_raw.columns = df_raw.columns.str.strip()

        st.subheader("📄 Dataset Overview")
        st.write(f"Total Records: **{df_raw.shape[0]}**")
        st.dataframe(df_raw, use_container_width=True)

        df = df_raw.copy()

        # ---------------------------------------------
        # DROP TARGET COLUMN IF PRESENT
        # ---------------------------------------------
        for col in ["churn", "Churn", "Exited", "is_churn", "Churn_Risk"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # ---------------------------------------------
        # TYPE FIXING
        # ---------------------------------------------
        for col in df.columns:
            if df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notnull().sum() > 0:
                    df[col] = converted

        # ---------------------------------------------
        # ENCODING
        # ---------------------------------------------
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        df.fillna(0, inplace=True)
        df = df.reindex(columns=columns, fill_value=0)

        st.success("Enterprise-grade data validation completed")

        # =================================================
        # PREDICTION
        # =================================================
        churn_prob = model.predict_proba(df)[:, 1]

        results = df_raw.copy()
        results["Churn_Probability"] = churn_prob

        results["Churn_Risk"] = np.where(
            churn_prob >= 0.7, "High Risk",
            np.where(churn_prob >= 0.4, "Low Risk", "Safe")
        )

        # UI SAFETY
        results = results.astype(str)

        # =================================================
        # SUMMARY DASHBOARD
        # =================================================
        st.subheader("📊 Executive Risk Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", len(results))
        c2.metric("Safe", (results["Churn_Risk"] == "Safe").sum())
        c3.metric("Low Risk", (results["Churn_Risk"] == "Low Risk").sum())
        c4.metric("High Risk", (results["Churn_Risk"] == "High Risk").sum())

        # =================================================
        # CUSTOMER SEARCH
        # =================================================
        st.subheader("🔎 Customer Risk Lookup")

        id_col = st.selectbox("Select Customer Identifier", results.columns)
        search_val = st.text_input("Enter Customer ID").strip()

        match = None
        if search_val:
            match = results[results[id_col].str.strip() == search_val]

            if match.empty:
                st.warning("No customer found")
            else:
                st.success("Customer located")
                st.dataframe(match, use_container_width=True)

                # =================================================
                # GENAI CUSTOMER EXPLANATION
                # =================================================
                if genai_enabled:
                    st.markdown("### 🧠 AI Customer Explanation")

                    prompt = f"""
                    Explain the churn risk for this customer in business terms.
                    Customer data: {match.iloc[0].to_dict()}
                    Include:
                    - Why risky/safe
                    - Business interpretation
                    - Recommended action
                    """

                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3
                    )

                    st.info(response.choices[0].message.content)

        # =================================================
        # DOWNLOADS
        # =================================================
        st.subheader("📥 Actionable Lists")

        st.download_button(
            "⬇ Full Results",
            results.to_csv(index=False),
            "full_predictions.csv"
        )

        # =================================================
        # EXECUTIVE SUMMARY PDF
        # =================================================
        if genai_enabled and st.button("📄 Generate Executive Summary PDF"):
            pdf_path = "Executive_Summary.pdf"
            doc = SimpleDocTemplate(pdf_path)
            styles = getSampleStyleSheet()

            summary_prompt = f"""
            Create an executive summary.
            Total customers: {len(results)}
            High Risk: {(results['Churn_Risk']=='High Risk').sum()}
            Low Risk: {(results['Churn_Risk']=='Low Risk').sum()}
            Safe: {(results['Churn_Risk']=='Safe').sum()}
            """

            summary = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3
            ).choices[0].message.content

            doc.build([Paragraph(summary, styles["Normal"])])

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇ Download Executive Summary",
                    f,
                    file_name="Executive_Summary.pdf"
                )

        # =================================================
        # GENAI BI COPILOT
        # =================================================
        st.subheader("💬 GenAI BI Copilot")

        question = st.text_input(
            "Ask any business question about this dataset"
        )

        if genai_enabled and question:
            prompt = f"""
            You are a senior enterprise BI analyst.
            Dataset columns: {list(results.columns)}
            Question: {question}
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            st.success(response.choices[0].message.content)

        elif question and not genai_enabled:
            st.warning("Enter OpenAI API key to enable GenAI features")

    except Exception as e:
        st.error("Enterprise processing error")
        st.code(str(e))

else:
    st.info("Upload CSV or connect SQL database to begin analysis")
