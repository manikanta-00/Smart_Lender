# 💳 Smart Lender — Loan Eligibility Prediction System

A machine-learning-powered web app that predicts whether a loan applicant is
likely to be approved, built to match the **Smart Lender** project brief
(Entity Relationship Diagram, Epics 1–5, and all required scenarios/skills).

## ⭐ Why you can run this without installing anything

Every single piece of this project — data generation, exploratory data
analysis charts, all four machine-learning models (**Decision Tree, Random
Forest, K-Nearest Neighbors, and an XGBoost-style Gradient Boosting
classifier**), and the web application itself — is written using **only
Python's standard library**. There is no pandas, numpy, scikit-learn,
xgboost, matplotlib, seaborn, or Flask anywhere in this project. You do not
need to run `pip install` at all.

The only requirement is **Python 3.8 or newer** (which you need to open any
Python project in VS Code anyway).

## 🚀 How to run it in VS Code

1. Unzip this project and open the `SmartLender` folder in VS Code.
2. Open a terminal in VS Code (``Ctrl+` ``).
3. Run:
   ```bash
   python run.py
   ```
4. The first run will automatically:
   - Generate a synthetic 600-row loan dataset (`data/loan_data.csv`)
   - Generate the EDA charts (`static/eda/eda_report.html`)
   - Train all 4 models, print a comparison table, and save the best one
   - Start the web server
5. Open your browser to **http://127.0.0.1:5000**

On future runs, the saved model is reused automatically so the app starts
instantly. To force it to regenerate the dataset and retrain everything:
```bash
python run.py --retrain
```

> If `python` doesn't work on your system, try `python3 run.py`.

## 🗺️ App pages

| Route       | What it shows |
|-------------|----------------|
| `/`         | Project introduction + the 3 usage scenarios from the brief |
| `/predict`  | Form to enter an applicant's details and get a live prediction |
| `/models`   | Train/test accuracy comparison table for all 4 models |
| `/eda`      | Auto-generated count plots, distribution plots, and bar charts |

## 📁 Project structure

```
SmartLender/
├── run.py                      # SINGLE ENTRY POINT — run this file
├── requirements.txt             # (empty — no installs needed)
├── README.md
├── data/
│   └── loan_data.csv            # auto-generated synthetic dataset
├── model_store/
│   └── model.pkl                 # auto-generated: best trained model + encoders
├── static/
│   ├── style.css
│   └── eda/eda_report.html       # auto-generated EDA charts
├── templates/
│   ├── home.html
│   ├── predict.html
│   ├── result.html
│   └── models.html
└── smart_lender/
    ├── generate_dataset.py       # Epic 1 — Data Collection
    ├── eda.py                    # Epic 2 — Visualizing & Analysing the Data
    ├── data_utils.py             # Epic 3 — Data Pre-processing & Feature Engineering
    ├── models.py                 # Epic 4 — Machine Learning Model Building
    ├── train.py                  # Orchestrates the full pipeline
    └── app.py                    # Epic 5 — Application Building (web server)
```

## 🧠 What each Epic maps to

- **Epic 1 – Data Collection & Architecture design** → `generate_dataset.py`
  creates a realistic loan-applicant dataset with the same columns described
  in the brief (Gender, Married, Dependents, Education, Self_Employed,
  ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
  Credit_History, Property_Area, Loan_Status), including some missing values
  on purpose so preprocessing has real work to do.

- **Epic 2 – Visualizing and Analysing the Data** → `eda.py` generates count
  plots (Gender, Education, Property Area, Loan Status), a distribution plot
  (Applicant Income, Loan Amount), and bar charts (approval rate by Property
  Area / Education) — all as hand-drawn SVG, so no matplotlib/seaborn needed.

- **Epic 3 – Data Pre-processing** → `data_utils.py` fills missing numeric
  values with the column mean and missing categorical values with the column
  mode, then encodes categorical columns into numbers.

- **Epic 4 – Model Building** → `models.py` implements Decision Tree, Random
  Forest, KNN, and an XGBoost-style Gradient Boosting classifier completely
  from scratch (CART trees with Gini impurity for classification, MSE-based
  regression trees + logistic-loss gradient boosting for the XGBoost stand-in).

- **Epic 5 – Application Building** → `app.py` serves the home page, the
  prediction form, the live prediction result, and the model comparison page
  using Python's built-in `http.server` (a stand-in for Flask).

## 📊 Typical results

On a fresh synthetic run you should see accuracy roughly in the same range
as the original brief (Decision Tree/Random Forest/KNN around 80–90% test
accuracy, XGBoost-style model around 80–90% as well) — the exact numbers
change slightly each time you regenerate the dataset since it's randomly
sampled.

## ⚙️ System requirements (matches the brief's Technical Architecture)

- **Hardware:** Intel i3 or above, 4 GB RAM (8 GB recommended), 10 GB free
  space, 64-bit OS
- **Software:** Windows 10/11, Linux, or macOS; **Python 3.8+**; VS Code or
  any editor; a web browser (Chrome/Edge) to view the app
- **No Anaconda, no pip installs, no internet connection required to run it**
