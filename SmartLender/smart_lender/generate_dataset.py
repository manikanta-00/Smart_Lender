"""
generate_dataset.py
--------------------
Generates a synthetic loan-applicant dataset (data/loan_data.csv) that mimics
the classic "Loan Eligibility Prediction" dataset described in the Smart
Lender project brief:

    Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
    CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
    Property_Area, Loan_Status

Some values are intentionally left blank so that the preprocessing step
(mean / mode imputation) has real work to do, exactly as described in the
"Data Preprocessing & Feature Engineering" section of the project brief.

No third-party packages are used -- only Python's built-in `random` and
`csv` modules.
"""

import csv
import os
import random

random.seed(42)

GENDERS = ["Male", "Female"]
MARRIED = ["Yes", "No"]
DEPENDENTS = ["0", "1", "2", "3+"]
EDUCATION = ["Graduate", "Not Graduate"]
SELF_EMPLOYED = ["Yes", "No"]
PROPERTY_AREA = ["Urban", "Semiurban", "Rural"]
LOAN_TERMS = [360, 180, 120, 300, 240, 84, 60, 36, 12]


def _maybe_blank(value, blank_rate=0.05):
    """Randomly blank out a value to simulate missing data."""
    return "" if random.random() < blank_rate else value


def generate_row(loan_id):
    gender = random.choice(GENDERS)
    married = random.choice(MARRIED)
    dependents = random.choice(DEPENDENTS)
    education = random.choices(EDUCATION, weights=[0.78, 0.22])[0]
    self_employed = random.choices(SELF_EMPLOYED, weights=[0.14, 0.86])[0]

    # Income is correlated with education / employment status
    base_income = random.gauss(5500, 2500)
    if education == "Graduate":
        base_income *= 1.25
    if self_employed == "Yes":
        base_income *= random.uniform(0.7, 1.6)  # more variance
    applicant_income = max(150, round(base_income))

    coapplicant_income = 0
    if married == "Yes" and random.random() < 0.65:
        coapplicant_income = max(0, round(random.gauss(2200, 1500)))

    loan_amount = max(9, round((applicant_income + coapplicant_income) / random.uniform(18, 55)))
    loan_term = random.choice(LOAN_TERMS)
    credit_history = random.choices([1, 0], weights=[0.84, 0.16])[0]
    property_area = random.choice(PROPERTY_AREA)

    # ---- Determine loan status using a "hidden" rule (ground truth) ----
    # debt_ratio = requested loan (in dollars) relative to ANNUAL income --
    # this is the standard loan-to-income framing (typical mortgages run 2-5x).
    total_income = applicant_income + coapplicant_income
    annual_income = total_income * 12
    debt_ratio = (loan_amount * 1000) / max(annual_income, 1)

    score = 0
    score += 1.1 if credit_history == 1 else -1.5
    score += 0.45 if education == "Graduate" else -0.15
    score += 0.25 if married == "Yes" else 0
    score += 0.35 if property_area == "Semiurban" else (0.1 if property_area == "Urban" else -0.2)
    score -= debt_ratio * 0.5
    score += random.gauss(0, 1.0)  # noise -- keeps the problem realistically noisy/imperfect

    loan_status = "Y" if score > -0.6 else "N"

    row = {
        "Loan_ID": f"LP{100000 + loan_id}",
        "Gender": _maybe_blank(gender),
        "Married": _maybe_blank(married),
        "Dependents": _maybe_blank(dependents),
        "Education": education,
        "Self_Employed": _maybe_blank(self_employed, 0.08),
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": _maybe_blank(loan_amount, 0.06),
        "Loan_Amount_Term": _maybe_blank(loan_term, 0.05),
        "Credit_History": _maybe_blank(credit_history, 0.07),
        "Property_Area": property_area,
        "Loan_Status": loan_status,
    }
    return row


def generate_dataset(path, n_rows=600):
    fieldnames = [
        "Loan_ID", "Gender", "Married", "Dependents", "Education",
        "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
        "LoanAmount", "Loan_Amount_Term", "Credit_History",
        "Property_Area", "Loan_Status",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(1, n_rows + 1):
            writer.writerow(generate_row(i))
    print(f"[generate_dataset] Wrote {n_rows} rows to {path}")


if __name__ == "__main__":
    generate_dataset(os.path.join(os.path.dirname(__file__), "..", "data", "loan_data.csv"))
