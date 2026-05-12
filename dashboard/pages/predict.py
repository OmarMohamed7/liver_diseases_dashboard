import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ..constants import (
    BINARY, BINARY_LABELS, CATEGORICAL, CLASS_COLORS, CLASS_LABELS,
    FEATURE_ORDER, MODEL_META,
)


def _proba_chart(model_name: str, proba: np.ndarray):
    prob_df = (
        pd.DataFrame({"Class": list(CLASS_LABELS.values()), "Probability": proba})
        .sort_values("Probability", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = [CLASS_COLORS.get(c, "#3498db") for c in prob_df["Class"]]
    bars = ax.barh(prob_df["Class"], prob_df["Probability"], color=colors)
    for bar, v in zip(bars, prob_df["Probability"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.1%}", va="center", fontsize=8)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Probability")
    ax.set_title(model_name)
    plt.tight_layout()
    return fig


def page_predict(cat_options, cat_encoders, _target_le, scaler, models):
    st.title("Liver Disease Prediction")
    st.markdown("Fill in the patient data and click **Predict** to get results from both models.")

    with st.form("prediction_form"):
        # ── Demographics & Lifestyle ──────────────────────────────────────────
        st.subheader("Demographics & Lifestyle")
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
            gender = st.selectbox("Gender", cat_options["Gender"])
            occupation = st.selectbox("Occupation", cat_options["Occupation"])
        with r1c2:
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            obesity_class = st.selectbox("Obesity Class", cat_options["Obesity_Class"])
            waist = st.number_input("Waist Circumference (cm)", min_value=40.0, max_value=200.0, value=85.0, step=0.1)
        with r1c3:
            diet = st.selectbox("Diet Quality", cat_options["Diet_Quality"])
            activity = st.selectbox("Physical Activity", cat_options["Physical_Activity"])
            sleep = st.number_input("Sleep Hours / Night", min_value=0.0, max_value=24.0, value=7.0, step=0.1)

        # ── Habits & History ─────────────────────────────────────────────────
        st.subheader("Habits & History")
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            smoking = st.selectbox("Smoking Status", cat_options["Smoking_Status"])
            alcohol = st.selectbox("Alcohol Consumption", cat_options["Alcohol_Consumption"])
        with r2c2:
            medication = st.selectbox("Medication History", cat_options["Medication_History"])
        with r2c3:
            source = st.selectbox("Source", cat_options["Source"])

        # ── Symptoms ─────────────────────────────────────────────────────────
        st.subheader("Symptoms")
        sym_features = [c for c in BINARY if c.startswith("Sym_")]
        sym_cols = st.columns(4)
        sym_values = {}
        for i, col in enumerate(sym_features):
            with sym_cols[i % 4]:
                sym_values[col] = int(st.checkbox(BINARY_LABELS[col], key=f"sym_{col}"))

        # ── Comorbidities ─────────────────────────────────────────────────────
        st.subheader("Comorbidities")
        com_features = [c for c in BINARY if c.startswith("Comorb_")]
        com_cols = st.columns(3)
        com_values = {}
        for i, col in enumerate(com_features):
            with com_cols[i % 3]:
                com_values[col] = int(st.checkbox(BINARY_LABELS[col], key=f"com_{col}"))

        # ── Lab Values ────────────────────────────────────────────────────────
        st.subheader("Lab Values")
        lab1, lab2, lab3, lab4 = st.columns(4)
        with lab1:
            alt = st.number_input("ALT (U/L)", min_value=0.0, max_value=500.0, value=35.0, step=0.1)
            ast = st.number_input("AST (U/L)", min_value=0.0, max_value=500.0, value=35.0, step=0.1)
        with lab2:
            bilirubin = st.number_input("Bilirubin (mg/dL)", min_value=0.0, max_value=20.0, value=1.0, step=0.01)
            albumin = st.number_input("Albumin (g/dL)", min_value=0.0, max_value=10.0, value=4.0, step=0.01)
        with lab3:
            platelets = st.number_input("Platelets (×10³/μL)", min_value=0, max_value=800, value=250)
            alk_phos = st.number_input("Alk. Phosphatase (U/L)", min_value=0, max_value=500, value=90)
        with lab4:
            ggt = st.number_input("GGT (U/L)", min_value=0, max_value=500, value=30)
            triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=0, max_value=1000, value=150)
            inr = st.number_input("INR", min_value=0.0, max_value=10.0, value=1.0, step=0.01)

        submitted = st.form_submit_button("Predict", width='stretch', type="primary")

    if not submitted:
        return

    # ── Encode & scale ────────────────────────────────────────────────────────
    def encode(col, val):
        return int(cat_encoders[col].transform([val])[0])

    input_data = {
        "Age": age,
        "Gender": encode("Gender", gender),
        "Occupation": encode("Occupation", occupation),
        "BMI": bmi,
        "Obesity_Class": encode("Obesity_Class", obesity_class),
        "Waist_Circumference": waist,
        "Diet_Quality": encode("Diet_Quality", diet),
        "Physical_Activity": encode("Physical_Activity", activity),
        "Sleep_Hours": sleep,
        "Smoking_Status": encode("Smoking_Status", smoking),
        "Alcohol_Consumption": encode("Alcohol_Consumption", alcohol),
        **sym_values,
        **com_values,
        "ALT": alt,
        "AST": ast,
        "Bilirubin": bilirubin,
        "Albumin": albumin,
        "Platelets": platelets,
        "Alk_Phosphatase": alk_phos,
        "GGT": ggt,
        "Triglycerides": triglycerides,
        "INR": inr,
        "Medication_History": encode("Medication_History", medication),
        "Source": encode("Source", source),
    }

    X_input = pd.DataFrame([input_data])[FEATURE_ORDER]
    X_scaled = scaler.transform(X_input)

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Prediction Results")

    result_cols = st.columns(len(models))
    for col, (model_name, model) in zip(result_cols, models.items()):
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        pred_label = CLASS_LABELS[pred]
        bg_color = CLASS_COLORS.get(pred_label, "#3498db")
        acc = MODEL_META[model_name]["accuracy"]

        with col:
            st.markdown(f"### {model_name}")
            st.markdown(
                f"<div style='background:{bg_color};padding:14px;border-radius:10px;"
                f"color:white;font-size:16px;font-weight:bold;text-align:center'>"
                f"{pred_label}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Training accuracy: {acc:.1%}")
            fig = _proba_chart(model_name, proba)
            st.pyplot(fig)
            plt.close()
