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
            background-attachment: fixed;
            background-position: center;
        }}
        .block-container {{
            background-color: rgba(0,0,0,0.60);
            padding: 2rem;
            border-radius: 14px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

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
# SIDEBAR – GENAI CONFIG + COPILOT
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
    st.sidebar.warning("GenAI Disabled (API Key Required)")

st.sidebar.markdown("---")
st.sidebar.subheader("💬 GenAI BI Copilot")

sidebar_question = st.sidebar.text_area(
    "Ask any business question about the dataset",
    height=130
)

# =================================================
# HEADER
# =================================================
st.title("🏦 Enterprise GenAI Business Intelligence Platform")
st.write(
    "Future churn risk prediction, executive explainability, "
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
    uploaded_file = st.file_uploader(
        "📂 Upload Customer Dataset",
        type=["csv"]
    )
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
        df_raw.columns = df_raw.columns.str.strip()

        st.subheader("📄 Dataset Overview")
        st.write(f"Total Records: **{len(df_raw)}**")
        st.dataframe(df_raw, use_container_width=True)

        df = df_raw.copy()

        # ---------------------------------------------
        # REMOVE TARGET IF EXISTS
        # ---------------------------------------------
        for col in ["churn", "Churn", "Exited", "is_churn", "Churn_Risk"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # ---------------------------------------------
        # TYPE FIXING
        # ---------------------------------------------
        for col in df.columns:
            if df[col].dtype == "object":
                num = pd.to_numeric(df[col], errors="coerce")
                if num.notnull().sum() > 0:
                    df[col] = num

        # ---------------------------------------------
        # ENCODING + ALIGNMENT
        # ---------------------------------------------
        df = pd.get_dummies(df, drop_first=True)
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
        # A/B TESTING – EXPERIMENT ASSIGNMENT
        # =================================================
        np.random.seed(42)

        results["AB_Test_Group"] = np.random.choice(
            ["Control", "Treatment_A", "Treatment_B"],
            size=len(results),
            p=[0.4, 0.3, 0.3]
        )

        results["Retention_Strategy"] = np.where(
            results["AB_Test_Group"] == "Control", "No Action",
            np.where(
                results["AB_Test_Group"] == "Treatment_A",
                "Discount / Offer",
                "Personalized Outreach"
            )
        )

        # =================================================
        # EXECUTIVE DASHBOARD
        # =================================================
        st.subheader("📊 Executive Risk Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", len(results))
        c2.metric("Safe", (results["Churn_Risk"] == "Safe").sum())
        c3.metric("Low Risk", (results["Churn_Risk"] == "Low Risk").sum())
        c4.metric("High Risk", (results["Churn_Risk"] == "High Risk").sum())

        st.subheader("🧪 A/B Testing Experiment Overview")

        ab1, ab2, ab3 = st.columns(3)
        ab1.metric("Control Group", (results["AB_Test_Group"] == "Control").sum())
        ab2.metric("Treatment A", (results["AB_Test_Group"] == "Treatment_A").sum())
        ab3.metric("Treatment B", (results["AB_Test_Group"] == "Treatment_B").sum())

        st.caption(
            "⚠️ Customers are randomly assigned to experimental groups. "
            "Actual churn reduction is measured over time."
        )

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
                    Explain the churn risk in business terms.
                    Customer data: {match.iloc[0].to_dict()}

                    Also explain:
                    - Why this customer is placed in the assigned A/B test group
                    - How the retention strategy helps measure business impact
                    - What success metrics should be tracked
                    """

                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3
                    )

                    st.info(response.choices[0].message.content)

        # =================================================
        # DOWNLOADS – ACTIONABLE LISTS
        # =================================================
        st.subheader("📥 Actionable Lists")

        safe_df = results[results["Churn_Risk"] == "Safe"]
        low_df = results[results["Churn_Risk"] == "Low Risk"]
        high_df = results[results["Churn_Risk"] == "High Risk"]

        st.download_button("⬇ Full Results", results.to_csv(index=False), "full_predictions.csv")
        st.download_button("⬇ Safe Customers", safe_df.to_csv(index=False), "safe_customers.csv")
        st.download_button("⬇ Low Risk Customers", low_df.to_csv(index=False), "low_risk_customers.csv")
        st.download_button("⬇ High Risk Customers", high_df.to_csv(index=False), "high_risk_customers.csv")

        # =================================================
        # EXECUTIVE SUMMARY PDF
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
                st.download_button("⬇ Download Executive Summary", f, file_name="Executive_Summary.pdf")

        # =================================================
        # SIDEBAR COPILOT RESPONSE
        # =================================================
        if genai_enabled and sidebar_question:
            prompt = f"""
            You are a senior enterprise BI analyst.
            Dataset columns: {list(results.columns)}
            Risk distribution:
            Safe={len(safe_df)}, Low={len(low_df)}, High={len(high_df)}
            Question: {sidebar_question}
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            st.sidebar.success(response.choices[0].message.content)

    except Exception as e:
        st.error("Enterprise processing error")
        st.code(str(e))

else:
    st.info("Upload CSV or connect SQL database to begin analysis.")
