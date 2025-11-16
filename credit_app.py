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

st.set_page_config(page_title="Credit Approval Predictor", layout="wide")
st.title("✅ Credit Approval Predictor (UCI Australian Dataset)")

MODEL_PATH = "model_rf.joblib"

# --------- Training section (runs only if model not found) ---------
if not os.path.exists(MODEL_PATH):
    st.info("Training model for the first time...")

    # Load dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat"
    df = pd.read_csv(url, sep=" ", header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Preprocessing
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)

    # Metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    st.success(f"Model trained: Accuracy = {accuracy_score(y_test, y_pred):.2f}, AUC = {roc_auc_score(y_test, y_proba):.2f}, F1 = {f1_score(y_test, y_pred):.2f}")

    joblib.dump(model, MODEL_PATH)
else:
    model = joblib.load(MODEL_PATH)

# --------- Input form ----------
st.sidebar.header("Applicant Features")
features = []
for i in range(14):
    features.append(st.sidebar.number_input(f"Feature A{i+1}", value=0.0, step=1.0))

input_df = pd.DataFrame([features])

if st.sidebar.button("Submit Application"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    decision = "✅ APPROVED" if prediction == 1 else "⛔ DENIED"
    st.subheader(f"Decision: {decision}")
    st.write(f"**Approval Probability:** {probability:.3f}")

    # Explainability
    st.write("### 🔍 Top SHAP Reason Codes")
    explainer = shap.Explainer(model.named_steps["clf"])
    transformed_input = model.named_steps["prep"].transform(input_df)
    shap_values = explainer(transformed_input)
    shap.plots.bar(shap_values[0], max_display=5, show=False)
    st.pyplot(plt.gcf())

    # Executive summary
    st.write("### 📄 Executive Summary")
    st.markdown(f"""
- **Prediction:** {decision}
- **Probability of approval:** {probability:.2f}
- **Model:** RandomForest on UCI Australian dataset
- **Top features:** shown above via SHAP
""")
