import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# HTML Template with Embedded Modern CSS
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Eligibility Risk Assessment</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-blue: #38bdf8;
            --accent-gradient: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --border-color: #334155;
            --success-color: #10b981;
            --danger-color: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
            overflow: hidden;
        }

        .header {
            background: var(--accent-gradient);
            padding: 2rem;
            text-align: center;
        }

        .header h1 {
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }

        .header p {
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.95rem;
            margin-top: 0.4rem;
        }

        form {
            padding: 2rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .form-group input, .form-group select {
            background: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease;
        }

        .form-group input:focus, .form-group select:focus {
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: var(--accent-gradient);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 1rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s ease, transform 0.1s ease;
        }

        .btn-submit:hover {
            opacity: 0.95;
        }

        .btn-submit:active {
            transform: scale(0.99);
        }

        .result-box {
            margin: 0 2rem 2rem 2rem;
            padding: 1.25rem;
            border-radius: 8px;
            text-align: center;
            font-weight: 600;
            font-size: 1.1rem;
        }

        .result-approved {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid var(--success-color);
            color: var(--success-color);
        }

        .result-rejected {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid var(--danger-color);
            color: var(--danger-color);
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Loan Assessment AI</h1>
        <p>AdaBoost Classification Inference System</p>
    </div>

    <form action="/predict" method="POST">
        <div class="grid">
            <div class="form-group">
                <label>Age</label>
                <input type="number" name="person_age" value="28" required>
            </div>
            
            <div class="form-group">
                <label>Gender</label>
                <select name="person_gender">
                    <option value="0">Female (0)</option>
                    <option value="1">Male (1)</option>
                </select>
            </div>

            <div class="form-group">
                <label>Education</label>
                <select name="person_education">
                    <option value="0">High School (0)</option>
                    <option value="1">Associate (1)</option>
                    <option value="2">Bachelor (2)</option>
                    <option value="3">Master / PhD (3)</option>
                </select>
            </div>

            <div class="form-group">
                <label>Annual Income ($)</label>
                <input type="number" name="person_income" value="55000" step="500" required>
            </div>

            <div class="form-group">
                <label>Employment Exp (Years)</label>
                <input type="number" name="person_emp_exp" value="3" required>
            </div>

            <div class="form-group">
                <label>Home Ownership</label>
                <select name="person_home_ownership">
                    <option value="0">RENT (0)</option>
                    <option value="1">MORTGAGE (1)</option>
                    <option value="2">OWN (2)</option>
                    <option value="3">OTHER (3)</option>
                </select>
            </div>

            <div class="form-group">
                <label>Loan Amount ($)</label>
                <input type="number" name="loan_amnt" value="10000" step="500" required>
            </div>

            <div class="form-group">
                <label>Loan Intent</label>
                <select name="loan_intent">
                    <option value="0">PERSONAL (0)</option>
                    <option value="1">EDUCATION (1)</option>
                    <option value="2">MEDICAL (2)</option>
                    <option value="3">VENTURE (3)</option>
                    <option value="4">HOMEIMPROVEMENT (4)</option>
                    <option value="5">DEBTCONSOLIDATION (5)</option>
                </select>
            </div>

            <div class="form-group">
                <label>Interest Rate (%)</label>
                <input type="number" name="loan_int_rate" value="11.5" step="0.1" required>
            </div>

            <div class="form-group">
                <label>Loan % of Income</label>
                <input type="number" name="loan_percent_income" value="0.18" step="0.01" required>
            </div>

            <div class="form-group">
                <label>Credit History Length (Yrs)</label>
                <input type="number" name="cb_person_cred_hist_length" value="4" required>
            </div>

            <div class="form-group">
                <label>Credit Score</label>
                <input type="number" name="credit_score" value="680" required>
            </div>

            <div class="form-group">
                <label>Previous Defaults</label>
                <select name="previous_loan_defaults_on_file">
                    <option value="0">No (0)</option>
                    <option value="1">Yes (1)</option>
                </select>
            </div>

            <button type="submit" class="btn-submit">Evaluate Risk Profile</button>
        </div>
    </form>

    {% if prediction_text %}
        <div class="result-box {% if is_approved %}result-approved{% else %}result-rejected{% endif %}">
            {{ prediction_text }}
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_LAYOUT)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_LAYOUT, 
            prediction_text="Error: Model pickle file not found or failed to load.",
            is_approved=False
        )

    try:
        # Extract features matching model order
        feature_order = [
            "person_age", "person_gender", "person_education", "person_income", 
            "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent", 
            "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", 
            "credit_score", "previous_loan_defaults_on_file"
        ]

        raw_features = [float(request.form.get(col, 0)) for col in feature_order]
        input_df = pd.DataFrame([raw_features], columns=feature_order)

        # Inference
        prediction = model.predict(input_df)[0]
        
        # Determine status
        is_approved = (prediction == 0) # Assuming 0 = Low Risk / Approved, 1 = Default / Risk
        status_text = "Loan Profile Approved (Low Risk)" if is_approved else "Loan Profile Flagged (High Risk)"

        return render_template_string(
            HTML_LAYOUT, 
            prediction_text=f"Prediction Outcome: {status_text}", 
            is_approved=is_approved
        )
    except Exception as e:
        return render_template_string(
            HTML_LAYOUT, 
            prediction_text=f"Error processing input: {str(e)}", 
            is_approved=False
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
