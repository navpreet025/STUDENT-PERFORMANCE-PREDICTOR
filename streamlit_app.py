# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.markdown("Select values from the choices and click **Predict** to see the performance category and score.")

# Load model (assumes model.pkl is in repository root)
MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("model.pkl not found. Please ensure model.pkl is in the repository root.")
    st.stop()

model = joblib.load(MODEL_PATH)

# ---- Input widgets (choice-based) ----
st.subheader("Student profile (choose options)")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.selectbox("Study hours per day", [0,1,2,3,4,5,6,7,8,9,10], index=5)
    sleep_quality_label = st.selectbox("Sleep quality", ["Poor","Below average","Average","Good","Excellent"])
    # map to number used when training
    sleep_quality_map = {"Poor":1,"Below average":2,"Average":3,"Good":4,"Excellent":5}
    sleep_quality = sleep_quality_map[sleep_quality_label]

    participation_label = st.selectbox("Class participation", ["Low","Moderate","Active"])
    participation_map = {"Low":1,"Moderate":3,"Active":5}
    participation = participation_map[participation_label]

with col2:
    activities_label = st.selectbox("Extracurricular activities", ["None","Occasional","Regular","Highly active"])
    activities_map = {"None":0,"Occasional":1,"Regular":2,"Highly active":3}
    activities = activities_map[activities_label]

    internet_label = st.selectbox("Internet usage (hrs/day)", ["<2","2–3","3–4","4–6",">6"])
    internet_map = {"<2":1,"2–3":2,"3–4":3,"4–6":4,">6":5}
    internet_usage = internet_map[internet_label]

    attendance_label = st.selectbox("Attendance", ["60–70%","75–85%","90–95%","Above 95%"])
    attendance_map = {"60–70%":65, "75–85%":80, "90–95%":92, "Above 95%":98}
    attendance = attendance_map[attendance_label]

assignments_label = st.selectbox("Assignments submitted", ["Few","Some","Most","All"])
assignments_map = {"Few":3,"Some":5,"Most":8,"All":10}
assignments_submitted = assignments_map[assignments_label]

# Optional: show chosen numeric features for transparency
if st.checkbox("Show numeric feature values"):
    st.write({
        "study_hours": study_hours,
        "sleep_quality": sleep_quality,
        "participation": participation,
        "activities": activities,
        "internet_usage": internet_usage,
        "attendance": attendance,
        "assignments_submitted": assignments_submitted
    })

# Predict button
if st.button("🔍 Predict Performance"):
    features = np.array([[study_hours, sleep_quality, participation, activities,
                          internet_usage, attendance, assignments_submitted]])
    try:
        score = float(model.predict(features)[0])
    except Exception as e:
        st.error("Model prediction failed: " + str(e))
        st.stop()

    score = max(0.0, min(100.0, score))  # clip to 0-100
    score_rounded = round(score, 2)

    # Map to performance category
    if score >= 85:
        perf = "Excellent 🌟"
        box_color = "#2ecc71"
    elif score >= 70:
        perf = "Good 🙂"
        box_color = "#3498db"
    elif score >= 50:
        perf = "Average 😐"
        box_color = "#f39c12"
    else:
        perf = "Needs Improvement ⚠️"
        box_color = "#e74c3c"

    # Show results
    st.markdown("### Result")
    st.markdown(f"<div style='padding:12px;border-radius:8px;background:{box_color};color:white;font-weight:600;'>"
                f"{perf} — Score: {score_rounded}%</div>", unsafe_allow_html=True)

    # Progress bar visualization
    st.progress(int(score_rounded))
    # small explanation
    st.write("**Tip:** This is a simple model trained on synthetic data. Replace `model.pkl` with a model trained on real data for production use.")
