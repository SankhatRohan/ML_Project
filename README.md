# 🏦 Loan Default Prediction — Streamlit Application

An **AI-powered Loan Risk Assessment** web application built with **Python + Streamlit**.  
It loads a pre-trained Balanced Random Forest model and predicts whether a borrower is likely to default on a loan.

---

## 📁 Project Structure

```
loan_default_app/
│
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
└── model/
    └── loan_default_model.pkl   ← ⬅ Place your trained model here
```

---

## 🚀 Quick Start

### Step 1 — Place Your Model File

Copy your `.pkl` file into the `model/` folder and rename it:

```
loan_default_app/model/loan_default_model.pkl
```

> **Note:** The app looks for the model at exactly this path.  
> If the file is missing, an error message will appear on startup.

---

### Step 2 — Install Dependencies

Open a terminal / command prompt, navigate to the project folder, and run:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install streamlit pandas numpy scikit-learn joblib imbalanced-learn
```

---

### Step 3 — Run the Application

```bash
streamlit run app.py
```

The app will open automatically in your default browser at:

```
http://localhost:8501
```

---

## 🤖 Model Requirements

| Property | Expected |
|---|---|
| **File format** | `.pkl` (joblib or pickle) |
| **Type** | sklearn Pipeline, ColumnTransformer, or direct model |
| **Input** | pandas DataFrame with 16 columns (see below) |
| **Output** | Binary prediction — `0` (No Default) or `1` (Default) |

### Expected Input Columns (in any order)

| Column | Type | Notes |
|---|---|---|
| `Age` | int | Years, 18–100 |
| `Income` | float | Annual USD |
| `LoanAmount` | float | USD |
| `CreditScore` | int | 300–850 |
| `MonthsEmployed` | int | At current employer |
| `NumCreditLines` | int | Open credit lines |
| `InterestRate` | float | % |
| `LoanTerm` | int | Months |
| `DTIRatio` | float | % |
| `Education` | str | `"High School"` / `"Bachelor's"` / `"Master's"` / `"PhD"` |
| `EmploymentType` | str | `"Full-time"` / `"Part-time"` / `"Self-employed"` / `"Unemployed"` |
| `MaritalStatus` | str | `"Single"` / `"Married"` / `"Divorced"` |
| `HasMortgage` | int | `1` = Yes, `0` = No |
| `HasDependents` | int | `1` = Yes, `0` = No |
| `LoanPurpose` | str | `"Business"` / `"Home"` / `"Education"` / `"Auto"` / `"Other"` |
| `HasCoSigner` | int | `1` = Yes, `0` = No |

---

## 🔄 Prediction Flow

```
User Input → pandas DataFrame → sklearn Pipeline (.pkl)
    → model.predict()       → 0 or 1
    → model.predict_proba() → [P(No Default), P(Default)]
    → Display Result Card + Probability Chart
```

1. User fills in **16 input fields** across 4 sections.
2. App validates all inputs and shows errors if invalid.
3. A `pandas DataFrame` is constructed with exact column names from training.
4. The pre-loaded model (cached with `@st.cache_resource`) runs prediction.
5. If `predict_proba()` is available, probabilities are shown alongside a bar chart.

---

## ⚠️ Troubleshooting

| Problem | Fix |
|---|---|
| `Model file not found` | Place your `.pkl` at `model/loan_default_model.pkl` |
| `Prediction Error` | Column names or data types don't match training data |
| `scikit-learn version mismatch` | Retrain the model with the same sklearn version |
| `ModuleNotFoundError: imbalanced-learn` | Run `pip install imbalanced-learn` |

---

## 🛠 Tech Stack

- **Python 3.9+**
- **Streamlit** — Web UI
- **scikit-learn** — ML model & pipeline
- **imbalanced-learn** — Balanced Random Forest
- **pandas / numpy** — Data handling
- **joblib** — Model serialization

---

## 📝 Notes

- This application **does not retrain** the model. It only loads and uses the saved `.pkl` file.
- All preprocessing (encoding, scaling) is assumed to be **inside the sklearn pipeline**.
- For educational / demonstration purposes only. Not financial advice.

---

*Loan Default Prediction | Machine Learning Project*
