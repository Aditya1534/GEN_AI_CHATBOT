import streamlit as st

# =================================================
# PAGE CONFIG (MUST BE FIRST)
# =================================================
st.set_page_config(
    page_title="Enterprise GenAI BI – Churn Intelligence",
    layout="wide"
)

# =================================================
# BACKGROUND IMAGE
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
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .block-container {{
            background-color: rgba(0, 0, 0, 0.55);
            padding: 2rem;
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("enterprise.jpeg")

# =================================================
# IMPORTS
# =================================================
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
        if all([host, db, user, password, table]):
            engine = create_engine(
                f"{'mysql+pymysql' if db_type=='MySQL' else 'postgresql'}://{user}:{password}@{host}:{port}/{db}"
            )
            df_raw = pd.read_sql(f"SELECT * FROM {table}", engine)
            st.success("Database connected successfully")
        else:
            st.error("Please fill all database fields")

# =================================================
# PROCESS DATA
# =================================================
if df_raw is not None:
    try:
        df_raw.columns = df_raw.columns.str.strip()

        st.subheader("📄 Dataset Overview")
        st.write(f"Total Records: **{len(df_raw)}**")
        st.dataframe(df_raw, use_container_width=True)

        df = df_raw.copy()

        # Remove target column if present
        for col in ["churn", "Churn", "Exited", "is_churn", "Churn_Risk"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Convert numeric strings
        for col in df.columns:
            if df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notnull().sum() > 0:
                    df[col] = converted

        # Encode categorical
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        df.fillna(0, inplace=True)
        df = df.reindex(columns=columns, fill_value=0)

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

        results = results.astype(str)

        # =================================================
        # DASHBOARD
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

        if search_val:
            match = results[results[id_col].str.strip() == search_val]

            if match.empty:
                st.warning("No customer found")
            else:
                st.success("Customer located")
                st.dataframe(match, use_container_width=True)

                if genai_enabled:
                    st.markdown("### 🧠 AI Customer Explanation")
                    prompt = f"""
                    Explain this customer's churn risk.
                    Customer data: {match.iloc[0].to_dict()}
                    Give business reasoning and action.
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
            "⬇ Download Full Results",
            results.to_csv(index=False),
            "full_predictions.csv"
        )

        safe_df = results[results["Churn_Risk"] == "Safe"]
        low_df = results[results["Churn_Risk"] == "Low Risk"]
        high_df = results[results["Churn_Risk"] == "High Risk"]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Safe Customers", len(safe_df))
            st.download_button("⬇ Safe", safe_df.to_csv(index=False), "safe.csv")

        with c2:
            st.metric("Low Risk Customers", len(low_df))
            st.download_button("⬇ Low Risk", low_df.to_csv(index=False), "low_risk.csv")

        with c3:
            st.metric("High Risk Customers", len(high_df))
            st.download_button("⬇ High Risk", high_df.to_csv(index=False), "high_risk.csv")

        # =================================================
        # EXECUTIVE PDF
        # =================================================
        if genai_enabled and st.button("📄 Generate Executive Summary PDF"):
            pdf_path = "Executive_Summary.pdf"
            doc = SimpleDocTemplate(pdf_path)
            styles = getSampleStyleSheet()

            summary_prompt = f"""
            Create an executive churn summary.
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
        question = st.text_input("Ask any business question")

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

    except Exception as e:
        st.error("Enterprise processing error")
        st.code(str(e))

else:
    st.info("Upload CSV or connect SQL database to begin analysis.")
