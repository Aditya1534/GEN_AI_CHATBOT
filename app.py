import streamlit as st

# MUST be first Streamlit command
st.set_page_config(
    page_title="Enterprise GenAI BI – Churn Intelligence",
    layout="wide"
)

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sqlalchemy import create_engine
import openai

warnings.filterwarnings("ignore")

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
            engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}")
        else:
            engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db}")

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
        possible_targets = ["churn", "Churn", "Exited", "is_churn", "Churn_Risk"]
        for col in possible_targets:
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

        # ---------------------------------------------
        # MISSING VALUES
        # ---------------------------------------------
        df.fillna(0, inplace=True)

        # ---------------------------------------------
        # SCHEMA ALIGNMENT
        # ---------------------------------------------
        df = df.reindex(columns=columns, fill_value=0)

        st.success("Enterprise-grade data validation completed")

        # =================================================
        # PREDICTION
        # =================================================
        churn_prob = model.predict_proba(df)[:, 1]

        results = df_raw.copy()
        results["Churn_Probability"] = churn_prob

        # =================================================
        # ENTERPRISE RISK SEGMENTATION
        # =================================================
        conditions = [
            churn_prob >= 0.7,
            (churn_prob >= 0.4) & (churn_prob < 0.7),
            churn_prob < 0.4
        ]
        choices = ["Safe", "Low Risk", "High Risk"]

        results["Churn_Risk"] = np.select(conditions, choices)

        # =================================================
        # SUMMARY DASHBOARD
        # =================================================
        st.subheader("📊 Executive Risk Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", results.shape[0])
        c2.metric("Safe", (results["Churn_Risk"] == "Safe").sum())
        c3.metric("Low Risk", (results["Churn_Risk"] == "Low Risk").sum())
        c4.metric("High Risk", (results["Churn_Risk"] == "High Risk").sum())

        # =================================================
        # CUSTOMER SEARCH
        # =================================================
        st.subheader("🔎 Customer Risk Lookup")

        id_col = st.selectbox("Select Customer Identifier", results.columns)
        search_val = st.text_input("Enter Customer ID")

        if search_val:
            match = results[results[id_col].astype(str) == search_val]
            if match.empty:
                st.warning("No customer found")
            else:
                st.success("Customer located")
                st.dataframe(match, use_container_width=True)

        # =================================================
        # DOWNLOAD ACTION LISTS
        # =================================================
        st.subheader("📥 Actionable Lists")

        high_df = results[results["Churn_Risk"] == "High Risk"]
        low_df = results[results["Churn_Risk"] == "Low Risk"]
        safe_df = results[results["Churn_Risk"] == "Safe"]

        st.download_button("⬇ High Risk Customers", high_df.to_csv(index=False), "high_risk.csv")
        st.download_button("⬇ Low Risk Customers", low_df.to_csv(index=False), "low_risk.csv")
        st.download_button("⬇ Safe Customers", safe_df.to_csv(index=False), "safe.csv")
        st.download_button("⬇ Full Results", results.to_csv(index=False), "full_predictions.csv")

        # =================================================
        # SHAP EXPLAINABILITY
        # =================================================
        st.subheader("🔍 Model Explainability (Regulatory Ready)")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.summary_plot(shap_values, df, show=False)
        st.pyplot(fig)

        # =================================================
        # GENAI BI COPILOT (OPTIONAL)
        # =================================================
        st.subheader("💬 GenAI BI Copilot")

        openai.api_key = st.text_input("OpenAI API Key", type="password")

        question = st.text_input(
            "Example: Which customer segment should we prioritize for retention?"
        )

        if openai.api_key and question:
            prompt = f"""
            You are a senior banking BI analyst.
            Dataset columns: {list(results.columns)}
            Question: {question}
            Provide SQL-style analytical guidance.
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )

            st.info(response.choices[0].message.content)

    except Exception as e:
        st.error("Enterprise processing error")
        st.code(str(e))

else:
    st.info("Upload CSV or connect SQL database to begin analysis")
