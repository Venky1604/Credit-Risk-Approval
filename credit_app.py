import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ------------------- Streamlit Page Config -------------------
st.set_page_config(page_title="Credit Approval Predictor", layout="wide")
st.title("✅ Credit Approval Predictor (UCI Australian Dataset)")

MODEL_PATH = "model_rf.joblib"

# ------------------- Train / Load Model -------------------
@st.cache_resource
def load_or_train_model():
    if not os.path.exists(MODEL_PATH):
        # Load dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat"
        df = pd.read_csv(url, sep=" ", header=None)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Categorical and numerical column indices (from original dataset)
        cat_idx = [0, 3, 4, 5, 7, 8, 10, 11]
        num_idx = list(set(range(14)) - set(cat_idx))

        preprocessor = ColumnTransformer([
            ("num", SimpleImputer(strategy="mean"), num_idx),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("enc", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_idx)
        ])

        model = Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model.fit(X_train, y_train)

        # Metrics
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_proba),
            "f1": f1_score(y_test, y_pred),
        }

        joblib.dump(model, MODEL_PATH)
        return model, metrics
    else:
        model = joblib.load(MODEL_PATH)
        return model, None

model, metrics = load_or_train_model()

if metrics is not None:
    st.success(
        f"Model trained: Accuracy = {metrics['accuracy']:.2f}, "
        f"AUC = {metrics['auc']:.2f}, F1 = {metrics['f1']:.2f}"
    )
else:
    st.info("Loaded existing trained model from disk.")

# ------------------- Session State Setup -------------------
if "step" not in st.session_state:
    st.session_state.step = -1  # -1 = personal details page
if "answers" not in st.session_state:
    st.session_state.answers = [0.0] * 14
if "finished" not in st.session_state:
    st.session_state.finished = False
if "decision" not in st.session_state:
    st.session_state.decision = None
if "probability" not in st.session_state:
    st.session_state.probability = None
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_sex" not in st.session_state:
    st.session_state.user_sex = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

# ------------------- Reset Function -------------------
def reset_application():
    st.session_state.step = -1
    st.session_state.answers = [0.0] * 14
    st.session_state.finished = False
    st.session_state.decision = None
    st.session_state.probability = None
    st.session_state.user_name = ""
    st.session_state.user_sex = ""
    st.session_state.user_email = ""

# ------------------- Sidebar Controls -------------------
st.sidebar.header("Application Controls")
if st.sidebar.button("🔄 Start New Application"):
    reset_application()
    st.sidebar.success("New application started.")

# Progress info
if st.session_state.step >= 0:
    st.sidebar.markdown(f"**Current question:** {st.session_state.step + 1} / 14")
else:
    st.sidebar.markdown("**Current step:** Applicant Info")

# ------------------- PERSONAL DETAILS PAGE (step = -1) -------------------
if st.session_state.step == -1 and not st.session_state.finished:
    st.subheader("👤 Applicant Information")

    st.session_state.user_name = st.text_input(
        "Applicant Name", value=st.session_state.user_name
    )
    st.session_state.user_sex = st.selectbox(
        "Sex",
        ["Male", "Female", "Other"],
        index=["Male", "Female", "Other"].index(st.session_state.user_sex)
        if st.session_state.user_sex else 0
    )
    st.session_state.user_email = st.text_input(
        "Email Address", value=st.session_state.user_email
    )

    if st.button("Start Application ➡️"):
        if st.session_state.user_name.strip() == "" or st.session_state.user_email.strip() == "":
            st.error("Please enter both your name and email address.")
        else:
            st.session_state.step = 0
            st.experimental_rerun()

# ------------------- FEATURE METADATA (HUMAN-FRIENDLY NAMES) -------------------
# These are inferred / conventional interpretations of the anonymized features.
feature_meta = [
    {
        "label": "Q1: Marital / Status Category (Feature A1)",
        "help": "Categorical code representing your personal/relationship or account status (e.g., 0, 1, 2...)."
    },
    {
        "label": "Q2: Age / Financial Score (Feature A2)",
        "help": "Numeric value often interpreted as age or an internal financial score.",
    },
    {
        "label": "Q3: Existing Debt / Loan Amount (Feature A3)",
        "help": "Numeric amount representing your current debts or requested loan size.",
    },
    {
        "label": "Q4: Employment Category (Feature A4)",
        "help": "Categorical code for job type or employment status (e.g., full-time, part-time, self-employed).",
    },
    {
        "label": "Q5: Loan Purpose Category (Feature A5)",
        "help": "Categorical code indicating what the loan is for (personal, car, business, etc.).",
    },
    {
        "label": "Q6: Account Type / Financial Category (Feature A6)",
        "help": "Categorical code for account type or another internal risk category.",
    },
    {
        "label": "Q7: Years of Employment (Feature A7)",
        "help": "Approximate years you have been employed (e.g., 0–40).",
    },
    {
        "label": "Q8: Occupation / Job Class (Feature A8)",
        "help": "Categorical code representing occupation type or job class.",
    },
    {
        "label": "Q9: Housing Status (Feature A9)",
        "help": "Categorical code for housing situation (rent, own, mortgage, etc.).",
    },
    {
        "label": "Q10: Savings / Investment Amount (Feature A10)",
        "help": "Numeric value representing your savings/investments or similar assets.",
    },
    {
        "label": "Q11: Dependents / Household Category (Feature A11)",
        "help": "Categorical code for number of dependents or household composition.",
    },
    {
        "label": "Q12: Additional Loan / Work Category (Feature A12)",
        "help": "Categorical flag for another loan or employment-related category.",
    },
    {
        "label": "Q13: Existing Credit Balance (Feature A13)",
        "help": "Numeric value for current total credit balance or exposure.",
    },
    {
        "label": "Q14: Credit Score / Payment Index (Feature A14)",
        "help": "Numeric summary of your past payment behavior or internal credit index.",
    },
]

# ------------------- QUESTION FLOW FOR 14 FEATURES -------------------
if st.session_state.step >= 0 and not st.session_state.finished:
    step = st.session_state.step

    st.subheader("📝 Application Questionnaire")
    st.write("Answer each question carefully. Once you click **Next**, you move to the next step.")

    meta = feature_meta[step]
    question = meta["label"]
    help_text = meta["help"]
    current_val = st.session_state.answers[step]

    answer = st.number_input(
        question,
        value=float(current_val),
        step=1.0,
        help=help_text,
        key=f"feat_{step}"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Next ➡️"):
            st.session_state.answers[step] = float(answer)

            if step == 13:
                # Last question -> run prediction
                input_df = pd.DataFrame([st.session_state.answers])

                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
                decision = "✅ APPROVED" if prediction == 1 else "⛔ DENIED"

                st.session_state.decision = decision
                st.session_state.probability = float(probability)
                st.session_state.finished = True
            else:
                st.session_state.step += 1

            st.experimental_rerun()

    with col2:
        if step > 0:
            if st.button("⬅️ Previous"):
                st.session_state.answers[step] = float(answer)
                st.session_state.step -= 1
                st.experimental_rerun()

# ------------------- RESULT PAGE -------------------
if st.session_state.finished:
    name = st.session_state.user_name
    sex = st.session_state.user_sex
    email = st.session_state.user_email
    decision = st.session_state.decision
    probability = st.session_state.probability

    st.subheader("📌 Final Decision")

    if "APPROVED" in decision:
        st.success(
            f"🎉 Congratulations {name}! Your credit application has been **APPROVED**.\n\n"
            f"**Approval Probability:** `{probability:.3f}`\n\n"
            f"An email confirmation will be sent to: **{email}**."
        )
    else:
        st.error(
            f"⛔ Sorry {name}, your credit application has been **DENIED**.\n\n"
            f"**Approval Probability:** `{probability:.3f}`"
        )

        st.write("### 💡 Tips to Improve / Maintain Your Credit Score")
        st.markdown("""
1. **Pay bills on time** – Payment history is the biggest factor in your score.  
2. **Reduce credit utilization below 30%** – Try to keep used credit well below the limit.  
3. **Avoid too many new applications** – Each hard inquiry can slightly reduce your score.  
4. **Keep older accounts open** – Older accounts help build a longer credit history.  
5. **Check your credit reports regularly** – Correct any errors or fraudulent entries early.
        """)

    # ---------------- SHAP EXPLANATION -------------------
    st.write("### 🔍 SHAP Reason Codes for This Decision")

    input_df = pd.DataFrame([st.session_state.answers])
    transformed_input = model.named_steps["prep"].transform(input_df)
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    shap_values = explainer(transformed_input)

    shap.plots.bar(shap_values[0], max_display=5, show=False)
    st.pyplot(plt.gcf())

    st.write("### 📄 Executive Summary")
    st.markdown(f"""
- **Applicant:** {name} ({sex})  
- **Email:** {email}  
- **Decision:** {decision}  
- **Approval Probability:** `{probability:.2f}`  
- **ML Model:** RandomForest  
- **Dataset:** UCI Australian Credit Approval  
- **Top Contributing Factors:** shown above in the SHAP bar chart.
    """)

    st.button("🔄 Start Another Application", on_click=reset_application)
