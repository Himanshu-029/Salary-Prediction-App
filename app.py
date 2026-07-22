from flask import Flask, render_template, request, jsonify, session, flash, redirect, url_for, Response
import pandas as pd
import pickle
import numpy as np
import os
import io
import csv
import sqlite3
import hashlib
import re
from werkzeug.utils import secure_filename
from datetime import datetime
from functools import wraps

# Create required folders if they don't exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('resumes', exist_ok=True)




app = Flask(__name__)

# Required for session handling and file uploads
app.config['SECRET_KEY'] = 'your-super-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESUME_FOLDER'] = 'resumes'
app.config['DATABASE'] = 'database.db'

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESUME_FOLDER'], exist_ok=True)

# -----------------------------
# DATABASE SETUP
# -----------------------------

def get_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            age INTEGER,
            gender TEXT,
            qualification TEXT,
            designation TEXT,
            experience INTEGER,
            predicted_salary REAL,
            model_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    ''')
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# LOAD DEFAULT MODEL (kept as fallback)
# -----------------------------

model = pickle.load(open("salary_prediction_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

# -----------------------------
# LOAD DEFAULT DATASET (kept as fallback)
# -----------------------------

df = pd.read_csv("Salary_Data.csv")
df = df.rename(columns={
    "Education Level": "Qualification",
    "Job Title": "Designation",
    "Years of Experience": "Work Experience"
})
df = df.dropna()
df["Qualification"] = df["Qualification"].replace({
    "phD": "PhD",
    "PHD": "PhD",
    "Bachelor's": "Bachelor's Degree",
    "Master's": "Master's Degree"
})

designations = sorted(df["Designation"].unique())
qualifications = sorted(df["Qualification"].unique())
genders = sorted(df["Gender"].unique())

# Job role mapping for recommendation engine
JOB_ROLES = {
    "Software Engineer": {"skills": ["programming", "coding", "python", "java"], "avg_salary": 12},
    "Data Scientist": {"skills": ["machine learning", "statistics", "python", "sql"], "avg_salary": 18},
    "Data Analyst": {"skills": ["excel", "sql", "tableau", "analysis"], "avg_salary": 8},
    "Product Manager": {"skills": ["leadership", "strategy", "agile", "communication"], "avg_salary": 22},
    "UX Designer": {"skills": ["figma", "design", "user research", "prototyping"], "avg_salary": 10},
    "DevOps Engineer": {"skills": ["aws", "docker", "kubernetes", "ci/cd"], "avg_salary": 16},
    "Business Analyst": {"skills": ["requirements", "stakeholder", "excel", "documentation"], "avg_salary": 9},
    "ML Engineer": {"skills": ["deep learning", "tensorflow", "pytorch", "nlp"], "avg_salary": 20}
}

# -----------------------------
# AUTH DECORATOR
# -----------------------------

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'info')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# -----------------------------
# HELPER: Get current active model & data
# -----------------------------

def get_active_model():
    if session.get('is_custom_session'):
        return pickle.loads(session['custom_model'])
    return model

def get_active_columns():
    if session.get('is_custom_session'):
        return pickle.loads(session['custom_columns'])
    return model_columns

def get_active_data():
    if session.get('is_custom_session'):
        return pd.read_json(session['custom_data'])
    return df

# -----------------------------
# HOME PAGE
# -----------------------------

@app.route("/")
def home():
    return render_template("home.html")

# -----------------------------
# AUTH ROUTES
# -----------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        name = request.form.get("name", "").strip()

        if not email or not password:
            flash("Email and password are required", "error")
            return redirect(url_for("register"))

        conn = get_db()
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            flash("Email already registered", "error")
            conn.close()
            return redirect(url_for("register"))

        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        conn.execute("INSERT INTO users (email, password, name) VALUES (?, ?, ?)", (email, hashed_pw, name))
        conn.commit()
        conn.close()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, hashed_pw)).fetchone()
        conn.close()

        if user:
            session['user_id'] = user['id']
            session['user_email'] = user['email']
            session['user_name'] = user['name']
            flash(f"Welcome back, {user['name'] or user['email']}!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid email or password", "error")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop('user_id', None)
    session.pop('user_email', None)
    session.pop('user_name', None)
    flash("Logged out successfully", "info")
    return redirect(url_for("home"))

# -----------------------------
# FILE UPLOAD ROUTE
# -----------------------------

@app.route("/upload", methods=["GET", "POST"])
def upload_dataset():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            flash('Please upload a CSV or Excel file', 'error')
            return redirect(request.url)
        
        try:
            if file.filename.endswith('.csv'):
                uploaded_df = pd.read_csv(file)
            else:
                uploaded_df = pd.read_excel(file)
            
            required_cols = ['Salary', 'Work Experience']
            missing = [col for col in required_cols if col not in uploaded_df.columns]
            if missing:
                flash(f'Missing required columns: {", ".join(missing)}', 'error')
                return redirect(request.url)
            
            rename_map = {}
            if 'Years of Experience' in uploaded_df.columns:
                rename_map['Years of Experience'] = 'Work Experience'
            if 'Education Level' in uploaded_df.columns:
                rename_map['Education Level'] = 'Qualification'
            if 'Job Title' in uploaded_df.columns:
                rename_map['Job Title'] = 'Designation'
            uploaded_df = uploaded_df.rename(columns=rename_map)
            uploaded_df = uploaded_df.dropna()
            
            feature_cols = ['Age', 'Gender', 'Qualification', 'Designation', 'Work Experience']
            available_cols = [col for col in feature_cols if col in uploaded_df.columns]
            
            X = uploaded_df[available_cols]
            y = uploaded_df['Salary']
            
            X_encoded = pd.get_dummies(X, drop_first=True)
            custom_columns = X_encoded.columns.tolist()
            
            from sklearn.linear_model import LinearRegression
            custom_model = LinearRegression()
            custom_model.fit(X_encoded, y)
            
            session['custom_model'] = pickle.dumps(custom_model)
            session['custom_columns'] = pickle.dumps(custom_columns)
            session['custom_data'] = uploaded_df.to_json()
            session['is_custom_session'] = True
            
            flash(f'✅ Custom model trained successfully on {len(uploaded_df)} records!', 'success')
            return redirect(url_for('analytics'))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template("upload.html")

# -----------------------------
# RESUME PARSER ROUTE
# -----------------------------

@app.route("/resume", methods=["GET", "POST"])
def resume_parser():
    parsed_data = None
    
    if request.method == "POST":
        if 'resume' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['resume']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        ext = file.filename.rsplit('.', 1)[-1].lower()
        if ext not in ['pdf', 'docx', 'txt']:
            flash('Only PDF, DOCX, or TXT files allowed', 'error')
            return redirect(request.url)
        
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['RESUME_FOLDER'], filename)
            file.save(filepath)
            
            text = ""
            if ext == 'txt':
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            elif ext == 'pdf':
                try:
                    import PyPDF2
                    with open(filepath, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = " ".join([page.extract_text() or "" for page in reader.pages])
                except ImportError:
                    flash('PyPDF2 not installed. Run: pip install PyPDF2', 'error')
                    return redirect(request.url)
            elif ext == 'docx':
                try:
                    import docx
                    doc = docx.Document(filepath)
                    text = " ".join([para.text for para in doc.paragraphs])
                except ImportError:
                    flash('python-docx not installed. Run: pip install python-docx', 'error')
                    return redirect(request.url)
            
            text_lower = text.lower()
            
            # Extract email
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
            extracted_email = email_match.group(0) if email_match else "Not found"
            
            # Extract phone
            phone_match = re.search(r'\+?\d[\d\s-]{8,}\d', text)
            extracted_phone = phone_match.group(0).strip() if phone_match else "Not found"
            
            # Detect skills
            all_skills = ["python", "java", "sql", "machine learning", "excel", "tableau", 
                         "docker", "aws", "figma", "agile", "leadership", "communication"]
            found_skills = [s for s in all_skills if s in text_lower]
            
            # Detect experience years
            exp_years = 0
            exp_patterns = [
                r'(\d+)\+?\s*years?\s*(of)?\s*experience',
                r'experience\s*:?\s*(\d+)\+?\s*years?',
            ]
            for pattern in exp_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    exp_years = int(match.group(1))
                    break
            
            # Detect qualification
            qual = "Bachelor's Degree"
            if any(w in text_lower for w in ["phd", "doctorate"]):
                qual = "PhD"
            elif any(w in text_lower for w in ["master", "m.tech", "mba"]):
                qual = "Master's Degree"
            
            parsed_data = {
                'email': extracted_email,
                'phone': extracted_phone,
                'skills': found_skills,
                'experience': exp_years,
                'qualification': qual,
                'raw_text': text[:500] + "..." if len(text) > 500 else text
            }
            
            flash('Resume parsed successfully!', 'success')
            
        except Exception as e:
            flash(f'Error parsing resume: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template("resume.html", parsed_data=parsed_data, 
                          designations=designations, qualifications=qualifications, genders=genders)

# -----------------------------
# JOB ROLE RECOMMENDATION
# -----------------------------

def recommend_roles(skills, experience, qualification):
    scores = {}
    for role, data in JOB_ROLES.items():
        score = 0
        role_skills = data['skills']
        for skill in skills:
            if skill.lower() in [s.lower() for s in role_skills]:
                score += 25
        # Bonus for experience match
        if experience >= 5:
            score += 20
        elif experience >= 2:
            score += 10
        # Bonus for higher qualifications
        if qualification in ["Master's Degree", "PhD"]:
            score += 15
        scores[role] = min(score, 100)
    
    # Sort by score descending, take top 3
    sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return [{"role": role, "match": score, "avg_salary": JOB_ROLES[role]['avg_salary']} 
            for role, score in sorted_roles if score > 0]

# -----------------------------
# RESET SESSION ROUTE
# -----------------------------

@app.route("/reset")
def reset_session_route():
    session.pop('is_custom_session', None)
    session.pop('custom_model', None)
    session.pop('custom_columns', None)
    session.pop('custom_data', None)
    session.pop('last_prediction', None)
    flash('🔄 Switched back to default model and dataset', 'info')
    return redirect(url_for('home'))

# -----------------------------
# PREDICT PAGE
# -----------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    explanation = None
    recommendations = None

    if request.method == "POST":
        age = int(request.form["age"])
        gender = request.form["gender"]
        qualification = request.form["qualification"]
        designation = request.form["designation"]
        experience = int(request.form["experience"])

        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Qualification": [qualification],
            "Designation": [designation],
            "Work Experience": [experience]
        })

        input_encoded = pd.get_dummies(input_df)
        active_columns = get_active_columns()
        input_encoded = input_encoded.reindex(columns=active_columns, fill_value=0)

        active_model = get_active_model()
        raw_prediction = active_model.predict(input_encoded)[0]
        prediction = round(raw_prediction / 10000, 2)

        # AI Explanation
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
        
        # Job role recommendations based on designation + experience
        dummy_skills = []
        for role, data in JOB_ROLES.items():
            if role.lower() in designation.lower():
                dummy_skills = data['skills']
                break
        recommendations = recommend_roles(dummy_skills, experience, qualification)
        
        # Store last prediction in session
        session['last_prediction'] = {
            'age': age,
            'gender': gender,
            'qualification': qualification,
            'designation': designation,
            'experience': experience,
            'predicted_salary': prediction,
            'explanation': explanation,
            'model_type': 'custom' if session.get('is_custom_session') else 'default',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to DB if logged in
        if 'user_id' in session:
            conn = get_db()
            conn.execute('''INSERT INTO predictions (user_id, age, gender, qualification, designation, experience, predicted_salary, model_type)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                       (session['user_id'], age, gender, qualification, designation, experience, prediction,
                        session['last_prediction']['model_type']))
            conn.commit()
            conn.close()

    active_data = get_active_data()
    active_designations = sorted(active_data["Designation"].unique()) if "Designation" in active_data.columns else designations
    active_qualifications = sorted(active_data["Qualification"].unique()) if "Qualification" in active_data.columns else qualifications
    active_genders = sorted(active_data["Gender"].unique()) if "Gender" in active_data.columns else genders

    return render_template(
        "predict.html",
        prediction=prediction,
        explanation=explanation,
        recommendations=recommendations,
        designations=active_designations,
        qualifications=active_qualifications,
        genders=active_genders
    )

# -----------------------------
# PREDICTION HISTORY
# -----------------------------

@app.route("/history")
@login_required
def prediction_history():
    conn = get_db()
    history = conn.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 20",
                          (session['user_id'],)).fetchall()
    conn.close()
    return render_template("history.html", history=history)

# -----------------------------
# EMAIL REPORT
# -----------------------------

@app.route("/email_report", methods=["GET", "POST"])
def email_report():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        if not email:
            flash("Please enter an email address", "error")
            return redirect(url_for("email_report"))
        
        pred = session.get('last_prediction')
        if not pred:
            flash("No prediction found. Make a prediction first.", "error")
            return redirect(url_for("predict"))
        
        # In production, use Flask-Mail. Here we simulate success.
        flash(f"📧 Report sent to {email}! (Simulated — configure Flask-Mail for production)", "success")
        return redirect(url_for("predict"))
    
    return render_template("email_report.html")

# -----------------------------
# NEGOTIATION TIPS
# -----------------------------

@app.route("/negotiation_tips")
def negotiation_tips():
    pred = session.get('last_prediction')
    if not pred:
        flash("Make a prediction first to get tips", "info")
        return redirect(url_for("predict"))
    
    salary = pred['predicted_salary']
    experience = pred['experience']
    designation = pred['designation']
    
    tips = []
    if salary >= 20:
        tips.append("You're in the top salary bracket. Ask for equity/ESOPs in addition to base pay.")
        tips.append("Negotiate for a signing bonus — companies expect this at senior levels.")
    elif salary >= 10:
        tips.append("You're in a strong position. Ask for 10-15% above the initial offer.")
        tips.append("Highlight your unique skills during negotiation — specificity wins.")
    else:
        tips.append("Focus on growth opportunities and learning budgets if base pay is fixed.")
        tips.append("Ask about performance review timelines and raise cycles.")
    
    if experience >= 8:
        tips.append("With your experience, leadership roles command premiums. Don't undervalue soft skills.")
    
    return render_template("negotiation_tips.html", pred=pred, tips=tips)

# -----------------------------
# API PLAYGROUND
# -----------------------------

@app.route("/api_playground")
def api_playground():
    return render_template("api_playground.html")

# -----------------------------
# DOWNLOAD PREDICTION AS CSV
# -----------------------------

@app.route("/download_prediction")
def download_prediction():
    pred = session.get('last_prediction')
    if not pred:
        flash('No prediction found. Make a prediction first.', 'error')
        return redirect(url_for('predict'))
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Field', 'Value'])
    writer.writerow(['Age', pred['age']])
    writer.writerow(['Gender', pred['gender']])
    writer.writerow(['Qualification', pred['qualification']])
    writer.writerow(['Designation', pred['designation']])
    writer.writerow(['Work Experience (Years)', pred['experience']])
    writer.writerow(['Predicted Salary (LPA)', f"₹{pred['predicted_salary']}"])
    writer.writerow(['Model Used', pred['model_type'].capitalize()])
    writer.writerow(['Generated On', pred['timestamp']])
    writer.writerow([])
    writer.writerow(['Explanation Breakdown', ''])
    for factor, percentage in pred['explanation'].items():
        writer.writerow([f'{factor} Impact', f'{percentage}%'])
    
    output.seek(0)
    return Response(output.getvalue(), mimetype='text/csv',
                   headers={'Content-Disposition': 'attachment; filename=salary_prediction.csv'})

# -----------------------------
# REST API PREDICT ENDPOINT
# -----------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame({
            "Age": [int(data.get("age", 30))],
            "Gender": [data.get("gender", "Male")],
            "Qualification": [data.get("qualification", "Bachelor's Degree")],
            "Designation": [data.get("designation", "Software Engineer")],
            "Work Experience": [int(data.get("experience", 5))]
        })
        input_encoded = pd.get_dummies(input_df)
        active_columns = get_active_columns()
        input_encoded = input_encoded.reindex(columns=active_columns, fill_value=0)
        active_model = get_active_model()
        raw_prediction = active_model.predict(input_encoded)[0]
        prediction = round(raw_prediction / 10000, 2)
        
        return jsonify({
            'status': 'success',
            'predicted_salary_lpa': prediction,
            'model_type': 'custom' if session.get('is_custom_session') else 'default'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# -----------------------------
# ANALYTICS DASHBOARD
# -----------------------------

@app.route("/analytics", methods=["GET", "POST"])
def analytics():
    predicted = None
    actual = None
    accuracy = None
    chart_type = "bar"

    active_data = get_active_data()
    salary_values = active_data["Salary"].dropna().values
    hist, bins = np.histogram(salary_values, bins=6)
    salary_bins = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
    salary_counts = hist.tolist()

    if "Work Experience" in active_data.columns:
        exp_salary = active_data.groupby("Work Experience")["Salary"].mean().reset_index()
        exp_labels = exp_salary["Work Experience"].tolist()
        exp_values = (exp_salary["Salary"] / 10000).round(2).tolist()
    else:
        exp_labels, exp_values = [], []

    if "Qualification" in active_data.columns:
        qual_salary = active_data.groupby("Qualification")["Salary"].mean().reset_index()
        qual_labels = qual_salary["Qualification"].tolist()
        qual_values = (qual_salary["Salary"] / 10000).round(2).tolist()
    else:
        qual_labels, qual_values = [], []

    if request.method == "POST":
        predicted = float(request.form["predicted"])
        actual = float(request.form["actual"])
        chart_type = request.form["chartType"]
        if actual != 0:
            error = abs(actual - predicted)
            accuracy = round((1 - (error / actual)) * 100, 2)

    is_custom = session.get('is_custom_session', False)

    return render_template(
        "analytics.html",
        predicted=predicted, actual=actual, accuracy=accuracy, chart_type=chart_type,
        salary_bins=salary_bins, salary_counts=salary_counts,
        exp_labels=exp_labels, exp_values=exp_values,
        qual_labels=qual_labels, qual_values=qual_values, is_custom=is_custom
    )

# -----------------------------
# RUN APP
# -----------------------------

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

