"""
╔══════════════════════════════════════════════════════════════════╗
║          LOAN DEFAULT PREDICTION - STREAMLIT APPLICATION         ║
║          AI-Powered Loan Risk Assessment Dashboard               ║
╚══════════════════════════════════════════════════════════════════╝

Author  : ML Project
Model   : Balanced Random Forest (Pre-trained .pkl)
Task    : Binary Classification — Predict Loan Default

Run:
    streamlit run app.py
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION  (must be first Streamlit command)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS  — professional fintech / banking style
# ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main background ── */
    .main { background-color: #f7f9fc; }

    /* ── Header card ── */
    .header-card {
        background: linear-gradient(135deg, #1a3c5e 0%, #2563eb 100%);
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 28px;
        color: white;
        box-shadow: 0 8px 32px rgba(37, 99, 235, 0.20);
    }
    .header-card h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .header-card h3 {
        font-size: 1.15rem;
        font-weight: 400;
        margin: 0 0 12px 0;
        opacity: 0.90;
    }
    .header-card p {
        font-size: 0.95rem;
        opacity: 0.80;
        margin: 0;
        line-height: 1.6;
    }

    /* ── Section cards ── */
    .section-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 20px;
        border: 1px solid #e8edf4;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a3c5e;
        border-left: 4px solid #2563eb;
        padding-left: 12px;
        margin-bottom: 18px;
    }

    /* ── Result cards ── */
    .result-safe {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 28px 32px;
        text-align: center;
    }
    .result-safe h2 { color: #065f46; font-size: 1.8rem; margin: 0 0 8px 0; }
    .result-safe p  { color: #047857; font-size: 1rem; margin: 0; }

    .result-danger {
        background: linear-gradient(135deg, #fff1f2 0%, #fecdd3 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 28px 32px;
        text-align: center;
    }
    .result-danger h2 { color: #7f1d1d; font-size: 1.8rem; margin: 0 0 8px 0; }
    .result-danger p  { color: #991b1b; font-size: 1rem; margin: 0; }

    /* ── Metric tiles ── */
    .metric-tile {
        background: #ffffff;
        border: 1px solid #e8edf4;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .metric-tile .label  { font-size: 0.80rem; color: #64748b; text-transform: uppercase; letter-spacing: .8px; }
    .metric-tile .value  { font-size: 1.7rem; font-weight: 700; color: #1a3c5e; margin-top: 4px; }

    /* ── Predict button ── */
    .stButton > button {
        background: linear-gradient(90deg, #1a3c5e 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 40px;
        font-size: 1.05rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: opacity .2s;
        letter-spacing: .3px;
    }
    .stButton > button:hover { opacity: 0.90; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #1a2744;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #93c5fd !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: #ef4444;
        border-radius: 8px;
        font-size: 0.9rem;
        padding: 10px 20px;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.82rem;
        padding: 30px 0 10px 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 40px;
    }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid #e8edf4; margin: 20px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────
# MODEL LOADING  — cached so it loads only once per session
# ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "loan_default_model.pkl")

@st.cache_resource(show_spinner="Loading AI model…")
def load_model():
    """
    Load the pre-trained sklearn pipeline / model from disk.
    Uses joblib first; falls back to pickle.
    Returns the model object or None on failure.
    """
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found at: `{MODEL_PATH}`"

    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as e:
        try:
            import pickle
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            return model, None
        except Exception as e2:
            return None, f"Could not load model.\n\n**joblib error:** {e}\n\n**pickle error:** {e2}"


model, load_error = load_model()


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Loan Risk AI")
    st.markdown("---")

    # Model status
    if model is not None:
        st.success("✅ Model Loaded Successfully")
    else:
        st.error("❌ Model Not Found")

    st.markdown("---")
    st.markdown("### 📋 Application Info")
    st.markdown(
        """
        This application uses a **Balanced Random Forest**
        classifier to predict whether a borrower is likely
        to **default** on their loan.

        The model was pre-trained on historical loan data
        and saved as a `.pkl` file.
        """
    )

    st.markdown("---")
    st.markdown("### 🤖 Model Details")
    st.markdown(
        """
        | Property | Value |
        |---|---|
        | **Task** | Binary Classification |
        | **Target** | Default (0 / 1) |
        | **Algorithm** | Balanced Random Forest |
        | **Input Features** | 16 |
        """
    )

    st.markdown("---")
    st.markdown("### 📌 How to Use")
    st.markdown(
        """
        1. Fill in all sections of the form
        2. Double-check your input values
        3. Click **Predict Loan Default**
        4. View the risk assessment result
        """
    )

    st.markdown("---")
    # Reset / clear session state button
    if st.button("🔄 Reset All Inputs"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b;'>⚠️ For educational purposes only.<br>Not financial advice.</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="header-card">
        <h1>🏦 Loan Default Prediction</h1>
        <h3>AI-Powered Loan Risk Assessment</h3>
        <p>
            This application uses a pre-trained Machine Learning model to predict whether a borrower
            is likely to default on a loan. Fill in the applicant's details below and click
            <strong>Predict Loan Default</strong> to get an instant risk assessment.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Show model load error prominently if any
if load_error:
    st.error(f"⚠️ **Model Load Error**\n\n{load_error}")
    st.info(
        "**Fix:** Place your `.pkl` file at `loan_default_app/model/loan_default_model.pkl` "
        "and restart the app."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────────────

# ── SECTION A — Personal Information ──────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 Section A — Personal Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=35,
        step=1,
        help="Applicant's age in years (18–100)",
        key="age",
    )

with col2:
    education = st.selectbox(
        "Education Level",
        options=["High School", "Bachelor's", "Master's", "PhD"],
        index=1,
        help="Highest education level attained",
        key="education",
    )

with col3:
    marital_status = st.selectbox(
        "Marital Status",
        options=["Single", "Married", "Divorced"],
        index=0,
        help="Current marital status",
        key="marital_status",
    )

st.markdown("</div>", unsafe_allow_html=True)


# ── SECTION B — Employment & Financial Information ─────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">💼 Section B — Employment & Financial Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "Annual Income (\$)",
        min_value=0,
        max_value=10_000_000,
        value=60_000,
        step=1_000,
        help="Gross annual income in USD",
        key="income",
    )
    months_employed = st.number_input(
        "Months Employed",
        min_value=0,
        max_value=600,
        value=36,
        step=1,
        help="Number of months at current/last employer",
        key="months_employed",
    )

with col2:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=680,
        step=1,
        help="FICO credit score (300–850)",
        key="credit_score",
    )
    dti_ratio = st.number_input(
        "DTI Ratio (Debt-to-Income %)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=0.1,
        format="%.2f",
        help="Total monthly debt payments / gross monthly income × 100",
        key="dti_ratio",
    )

employment_type = st.selectbox(
    "Employment Type",
    options=["Full-time", "Part-time", "Self-employed", "Unemployed"],
    index=0,
    help="Current employment status",
    key="employment_type",
)

st.markdown("</div>", unsafe_allow_html=True)


# ── SECTION C — Loan Information ──────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">💳 Section C — Loan Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    loan_amount = st.number_input(
        "Loan Amount (\$)",
        min_value=0,
        max_value=1_000_000,
        value=20_000,
        step=500,
        help="Total loan amount requested in USD",
        key="loan_amount",
    )
    interest_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=8.5,
        step=0.1,
        format="%.2f",
        help="Annual interest rate on the loan",
        key="interest_rate",
    )

with col2:
    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=1,
        max_value=600,
        value=36,
        step=1,
        help="Duration of the loan in months",
        key="loan_term",
    )
    num_credit_lines = st.number_input(
        "Number of Credit Lines",
        min_value=0,
        max_value=50,
        value=3,
        step=1,
        help="Total number of open credit lines",
        key="num_credit_lines",
    )

loan_purpose = st.selectbox(
    "Loan Purpose",
    options=["Business", "Home", "Education", "Auto", "Other"],
    index=4,
    help="Primary purpose of the loan",
    key="loan_purpose",
)

st.markdown("</div>", unsafe_allow_html=True)


# ── SECTION D — Additional Information ────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📎 Section D — Additional Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    has_mortgage_input = st.selectbox(
        "Has Mortgage?",
        options=["No", "Yes"],
        index=0,
        help="Does the applicant currently have a mortgage?",
        key="has_mortgage",
    )

with col2:
    has_dependents_input = st.selectbox(
        "Has Dependents?",
        options=["No", "Yes"],
        index=0,
        help="Does the applicant have financial dependents?",
        key="has_dependents",
    )

with col3:
    has_cosigner_input = st.selectbox(
        "Has Co-Signer?",
        options=["No", "Yes"],
        index=0,
        help="Does the loan have a co-signer?",
        key="has_cosigner",
    )

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Predict Loan Default", use_container_width=True)

if predict_clicked:

    # ── FORM VALIDATION ───────────────────────────────────────────
    errors = []

    if age < 18 or age > 100:
        errors.append("Age must be between **18** and **100** years.")
    if income < 0:
        errors.append("Annual Income cannot be **negative**.")
    if loan_amount <= 0:
        errors.append("Loan Amount must be **greater than 0**.")
    if credit_score < 300 or credit_score > 850:
        errors.append("Credit Score must be between **300** and **850**.")
    if interest_rate < 0:
        errors.append("Interest Rate cannot be **negative**.")
    if loan_term < 1:
        errors.append("Loan Term must be at least **1 month**.")
    if dti_ratio < 0:
        errors.append("DTI Ratio cannot be **negative**.")
    if months_employed < 0:
        errors.append("Months Employed cannot be **negative**.")
    if num_credit_lines < 0:
        errors.append("Number of Credit Lines cannot be **negative**.")

    if errors:
        st.markdown("---")
        st.error("⚠️ **Please fix the following errors before predicting:**")
        for err in errors:
            st.markdown(f"- {err}")
    else:
        # ── CONVERT Yes/No to 1/0 ─────────────────────────────────
        # NOTE: If your pipeline already handles string "Yes"/"No", keep them as strings.
        # The mapping below converts to integer 0/1 which is commonly expected.
        has_mortgage = has_mortgage_input
        has_dependents = has_dependents_input
        has_cosigner = has_cosigner_input

        # ── BUILD INPUT DATAFRAME ─────────────────────────────────
        # Column names MUST exactly match those used during model training.
        input_data = pd.DataFrame({
            "Age":            [age],
            "Income":         [income],
            "LoanAmount":     [loan_amount],
            "CreditScore":    [credit_score],
            "MonthsEmployed": [months_employed],
            "NumCreditLines": [num_credit_lines],
            "InterestRate":   [interest_rate],
            "LoanTerm":       [loan_term],
            "DTIRatio":       [dti_ratio],
            "Education":      [education],
            "EmploymentType": [employment_type],
            "MaritalStatus":  [marital_status],
            "HasMortgage":    [has_mortgage],
            "HasDependents":  [has_dependents],
            "LoanPurpose":    [loan_purpose],
            "HasCoSigner":    [has_cosigner],
        })

        # ── RUN PREDICTION ────────────────────────────────────────
        try:
            with st.spinner("🔄 Running AI risk assessment…"):
                prediction = model.predict(input_data)[0]           # 0 or 1

                # Probability support (most sklearn estimators have this)
                prob_default    = None
                prob_no_default = None
                has_proba       = hasattr(model, "predict_proba")

                if has_proba:
                    proba           = model.predict_proba(input_data)[0]  # [P(0), P(1)]
                    prob_no_default = float(proba[0])
                    prob_default    = float(proba[1])

            # ── RESULT SECTION ────────────────────────────────────
            st.markdown("---")
            st.markdown("## 📊 Prediction Result")

            if prediction == 0:
                # ── LOW RISK ──────────────────────────────────────
                st.markdown(
                    """
                    <div class="result-safe">
                        <h2>✅ LOW DEFAULT RISK</h2>
                        <p>Prediction: The loan applicant is <strong>unlikely to default</strong>.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # ── HIGH RISK ─────────────────────────────────────
                st.markdown(
                    """
                    <div class="result-danger">
                        <h2>🚨 HIGH DEFAULT RISK</h2>
                        <p>Prediction: The loan applicant is <strong>likely to default</strong>.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── METRIC TILES ──────────────────────────────────────
            c1, c2, c3 = st.columns(3)

            with c1:
                risk_label = "🔴 HIGH" if prediction == 1 else "🟢 LOW"
                st.markdown(
                    f"""<div class="metric-tile">
                        <div class="label">Risk Level</div>
                        <div class="value">{risk_label}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            if has_proba:
                with c2:
                    st.markdown(
                        f"""<div class="metric-tile">
                            <div class="label">Default Probability</div>
                            <div class="value" style="color:{'#ef4444' if prob_default >= 0.5 else '#10b981'}">
                                {prob_default*100:.1f}%
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                with c3:
                    st.markdown(
                        f"""<div class="metric-tile">
                            <div class="label">No-Default Probability</div>
                            <div class="value" style="color:#10b981">
                                {prob_no_default*100:.1f}%
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            else:
                with c2:
                    st.markdown(
                        f"""<div class="metric-tile">
                            <div class="label">Prediction Class</div>
                            <div class="value">{int(prediction)}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            # ── PROBABILITY VISUALISATION ─────────────────────────
            if has_proba:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📈 Probability Breakdown")

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("**✅ No Default Probability**")
                    st.progress(float(prob_no_default), text=f"{prob_no_default*100:.1f}%")

                with col_b:
                    st.markdown("**🚨 Default Probability**")
                    st.progress(float(prob_default), text=f"{prob_default*100:.1f}%")

                # Bar chart
                chart_data = pd.DataFrame(
                    {
                        "Outcome": ["No Default", "Default"],
                        "Probability (%)": [
                            round(prob_no_default * 100, 2),
                            round(prob_default * 100, 2),
                        ],
                    }
                ).set_index("Outcome")

                st.markdown("<br>", unsafe_allow_html=True)
                st.bar_chart(chart_data, color=["#2563eb"], height=260)

            # ── INPUT SUMMARY ─────────────────────────────────────
            with st.expander("📋 View Input Summary", expanded=False):
                summary_df = input_data.copy()
                # Map binary back to labels for readability
                summary_df["HasMortgage"]   = "Yes" if has_mortgage   == 1 else "No"
                summary_df["HasDependents"] = "Yes" if has_dependents == 1 else "No"
                summary_df["HasCoSigner"]   = "Yes" if has_cosigner   == 1 else "No"
                st.dataframe(summary_df.T.rename(columns={0: "Value"}), use_container_width=True)

        except Exception as e:
            st.error(
                f"⚠️ **Prediction Error**\n\n"
                f"An error occurred while running the model:\n\n`{e}`\n\n"
                "**Possible causes:**\n"
                "- The `.pkl` file may have been trained with different column names.\n"
                "- The pipeline may expect different preprocessing.\n"
                "- Scikit-learn version mismatch between training and this environment.\n\n"
                "Please check the model's expected input format and update `app.py` accordingly."
            )


# ─────────────────────────────────────────────────────────────────
# ABOUT THE MODEL SECTION
# ─────────────────────────────────────────────────────────────────
st.markdown("---")

with st.expander("ℹ️ About the Model", expanded=False):
    st.markdown(
        """
        ### 🤖 Model Information

        | Property              | Details                              |
        |-----------------------|--------------------------------------|
        | **Problem**           | Loan Default Prediction              |
        | **ML Task**           | Binary Classification                |
        | **Algorithm**         | Balanced Random Forest               |
        | **Target Variable**   | `Default`                            |
        | **Class 0**           | No Default (loan repaid successfully)|
        | **Class 1**           | Default (borrower failed to repay)   |
        | **Input Features**    | 16 (numerical + categorical)         |

        ### 📦 How Prediction Works

        1. User fills in the 16 input features in the form.
        2. The app constructs a **pandas DataFrame** with exact column names
           used during model training.
        3. The pre-loaded sklearn **Pipeline** (or model) processes the data —
           this includes any encoders and scalers baked into the pipeline.
        4. `model.predict()` returns `0` (No Default) or `1` (Default).
        5. `model.predict_proba()` returns confidence probabilities for each class.

        ### 📊 Feature Descriptions

        | Feature         | Type        | Description                            |
        |-----------------|-------------|----------------------------------------|
        | Age             | Numerical   | Applicant's age                        |
        | Income          | Numerical   | Annual income (USD)                    |
        | LoanAmount      | Numerical   | Loan amount requested (USD)            |
        | CreditScore     | Numerical   | FICO credit score (300–850)            |
        | MonthsEmployed  | Numerical   | Months at current employer             |
        | NumCreditLines  | Numerical   | Open credit lines                      |
        | InterestRate    | Numerical   | Annual interest rate (%)               |
        | LoanTerm        | Numerical   | Loan duration (months)                 |
        | DTIRatio        | Numerical   | Debt-to-income ratio (%)               |
        | Education       | Categorical | Highest education level                |
        | EmploymentType  | Categorical | Employment status                      |
        | MaritalStatus   | Categorical | Marital status                         |
        | HasMortgage     | Binary      | 1 = Yes, 0 = No                        |
        | HasDependents   | Binary      | 1 = Yes, 0 = No                        |
        | LoanPurpose     | Categorical | Reason for the loan                    |
        | HasCoSigner     | Binary      | 1 = Yes, 0 = No                        |
        """
    )


# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        🏦 <strong>Loan Default Prediction</strong> | Machine Learning Project<br>
        Built with Python & Streamlit · For educational purposes only
    </div>
    """,
    unsafe_allow_html=True,
)
