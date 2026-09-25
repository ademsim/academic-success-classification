from pathlib import Path

import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Academic Success Predictor")

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return pickle.load(open(BASE / "academic_model.pkl", "rb"))


@st.cache_resource
def load_meta():
    return pickle.load(open(BASE / "academic_meta.pkl", "rb"))


model = load_model()
meta = load_meta()
feature_cols = list(dict.fromkeys(meta["feature_cols"]))
defaults = meta["defaults"]

st.title("Academic Success Predictor")
st.write(
    "A Random Forest model (trained on the Kaggle 'Academic Success' dataset, based on real student records) "
    "predicts whether a student is likely to drop out, stay enrolled, or graduate, from the most important features. "
    "Every other feature is filled in with its typical (median) value from the training data."
)

col1, col2 = st.columns(2)
with col1:
    approved_2 = st.slider("Curricular units approved, 2nd semester", 0, 25, 6)
    grade_2 = st.slider("Average grade, 2nd semester (0-20)", 0.0, 20.0, 12.0, 0.1)
    approved_1 = st.slider("Curricular units approved, 1st semester", 0, 25, 6)
    grade_1 = st.slider("Average grade, 1st semester (0-20)", 0.0, 20.0, 12.0, 0.1)
    eval_2 = st.slider("Number of evaluations, 2nd semester", 0, 30, 8)
with col2:
    eval_1 = st.slider("Number of evaluations, 1st semester", 0, 30, 8)
    age = st.slider("Age at enrollment", 17, 70, 20)
    admission_grade = st.slider("Admission grade (0-200)", 0.0, 200.0, 130.0, 0.5)
    tuition_ok = st.checkbox("Tuition fees up to date", value=True)
    scholarship = st.checkbox("Scholarship holder", value=False)

if st.button("Predict"):
    row = dict(defaults)
    row.update({
        "Curricular units 2nd sem (approved)": approved_2,
        "Curricular units 2nd sem (grade)": grade_2,
        "Curricular units 1st sem (approved)": approved_1,
        "Curricular units 1st sem (grade)": grade_1,
        "Curricular units 2nd sem (evaluations)": eval_2,
        "Curricular units 1st sem (evaluations)": eval_1,
        "Age at enrollment": age,
        "Admission grade": admission_grade,
        "Tuition fees up to date": 1 if tuition_ok else 0,
        "Scholarship holder": 1 if scholarship else 0,
    })

    X = pd.DataFrame([row])
    X = X.reindex(columns=feature_cols, fill_value=0)

    pred = model.predict(X)[0]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0]))
    icon = {"Graduate": "🎓", "Enrolled": "📚", "Dropout": "⚠️"}.get(pred, "")
    st.success(f"{icon} Predicted outcome: **{pred}**")
    st.bar_chart(pd.Series(proba, name="probability"))

st.caption(
    "Model: Random Forest (validation accuracy ≈ 77-83% depending on dataset size). Units approved and grades "
    "in both semesters are by far the strongest predictors of dropping out or graduating."
)
