from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        study_hours = int(request.form["study_hours"])
        sleep_quality = int(request.form["sleep_quality"])
        participation = int(request.form["participation"])
        activities = int(request.form["activities"])
        internet_usage = int(request.form["internet_usage"])
        attendance = int(request.form["attendance"])
        assignments_submitted = int(request.form["assignments_submitted"])

        features = np.array([[study_hours, sleep_quality, participation, activities,
                              internet_usage, attendance, assignments_submitted]])
        prediction = model.predict(features)[0]

        if prediction >= 85:
            category = "Excellent 🌟"
            color = "green"
        elif prediction >= 70:
            category = "Good 🙂"
            color = "blue"
        elif prediction >= 50:
            category = "Average 😐"
            color = "orange"
        else:
            category = "Needs Improvement ⚠️"
            color = "red"

        return render_template("result.html", score=round(prediction,2),
                               category=category, color=color)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)

