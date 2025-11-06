# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np

# ----- Simulated dataset -----
data = {
    "study_hours": [1,2,3,4,5,6,7,8,9,10],
    "sleep_quality": [1,2,3,4,5,3,4,5,4,3],  # new feature
    "participation": [1,2,3,4,5,3,4,5,5,5],
    "activities": [0,1,1,2,2,2,3,3,2,1],
    "internet_usage": [5,4,4,3,3,2,2,1,2,3],
    "attendance": [60,65,70,75,80,85,90,95,97,99],
    "assignments_submitted": [3,4,5,6,7,8,9,9,10,10],
}

df = pd.DataFrame(data)
np.random.seed(42)

df["final_score"] = (
    df["study_hours"] * 4.5
    + df["sleep_quality"] * 1.5
    + df["participation"] * 2
    + df["attendance"] * 0.4
    + df["assignments_submitted"] * 1.5
    - df["internet_usage"] * 1.2
    + np.random.normal(0, 3, len(df))
)

df["final_score"] = df["final_score"].clip(0, 100)

# ----- Train model -----
X = df.drop(columns=["final_score"])
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")


