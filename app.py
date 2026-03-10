from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# -----------------------------
# LOAD MODEL
# -----------------------------

model = pickle.load(open("salary_prediction_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_csv("Salary_Data.csv")

df = df.rename(columns={
    "Education Level": "Qualification",
    "Job Title": "Designation",
    "Years of Experience": "Work Experience"
})

# Remove missing values
df = df.dropna()

# Fix qualification names
df["Qualification"] = df["Qualification"].replace({
    "phD": "PhD",
    "PHD": "PhD",
    "Bachelor's": "Bachelor's Degree",
    "Master's": "Master's Degree"
})

# Dropdown options
designations = sorted(df["Designation"].unique())
qualifications = sorted(df["Qualification"].unique())
genders = sorted(df["Gender"].unique())


# -----------------------------
# HOME PAGE
# -----------------------------

@app.route("/")
def home():
    return render_template("home.html")


# -----------------------------
# PREDICT PAGE
# -----------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():

    prediction = None
    explanation = None

    if request.method == "POST":

        age = int(request.form["age"])
        gender = request.form["gender"]
        qualification = request.form["qualification"]
        designation = request.form["designation"]
        experience = int(request.form["experience"])

        # Create dataframe from input
        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Qualification": [qualification],
            "Designation": [designation],
            "Work Experience": [experience]
        })

        # Encode categorical variables
        input_encoded = pd.get_dummies(input_df)

        input_encoded = input_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )

        # Model prediction
        raw_prediction = model.predict(input_encoded)[0]

        # Convert salary to LPA
        prediction = round(raw_prediction / 10000, 2)

        # -----------------------------
        # AI EXPLANATION
        # -----------------------------

        exp_weight = min(60, experience * 8)
        qual_weight = 20
        role_weight = 15
        age_weight = 5

        total = exp_weight + qual_weight + role_weight + age_weight

        explanation = {
            "Experience": round(exp_weight / total * 100, 1),
            "Qualification": round(qual_weight / total * 100, 1),
            "Role": round(role_weight / total * 100, 1),
            "Age": round(age_weight / total * 100, 1)
        }

    return render_template(
        "predict.html",
        prediction=prediction,
        explanation=explanation,
        designations=designations,
        qualifications=qualifications,
        genders=genders
    )


# -----------------------------
# ANALYTICS DASHBOARD
# -----------------------------

@app.route("/analytics", methods=["GET", "POST"])
def analytics():

    predicted = None
    actual = None
    accuracy = None
    chart_type = "bar"

    # -----------------------------
    # DATASET INSIGHTS
    # -----------------------------

    salary_values = df["Salary"].dropna().values

    hist, bins = np.histogram(salary_values, bins=6)

    salary_bins = [
        f"{int(bins[i])}-{int(bins[i+1])}"
        for i in range(len(bins) - 1)
    ]

    salary_counts = hist.tolist()

    # Experience vs salary
    exp_salary = df.groupby("Work Experience")["Salary"].mean().reset_index()

    exp_labels = exp_salary["Work Experience"].tolist()
    exp_values = (exp_salary["Salary"] / 10000).round(2).tolist()

    # Qualification vs salary
    qual_salary = df.groupby("Qualification")["Salary"].mean().reset_index()

    qual_labels = qual_salary["Qualification"].tolist()
    qual_values = (qual_salary["Salary"] / 10000).round(2).tolist()

    # -----------------------------
    # MODEL COMPARISON
    # -----------------------------

    if request.method == "POST":

        predicted = float(request.form["predicted"])
        actual = float(request.form["actual"])
        chart_type = request.form["chartType"]

        if actual != 0:
            error = abs(actual - predicted)
            accuracy = round((1 - (error / actual)) * 100, 2)

    return render_template(
        "analytics.html",
        predicted=predicted,
        actual=actual,
        accuracy=accuracy,
        chart_type=chart_type,
        salary_bins=salary_bins,
        salary_counts=salary_counts,
        exp_labels=exp_labels,
        exp_values=exp_values,
        qual_labels=qual_labels,
        qual_values=qual_values
    )


# -----------------------------
# RUN APP
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)