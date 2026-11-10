# 🏦 Customer Segmentation & Churn Pattern Analytics in European Banking

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An enterprise-grade customer segmentation, churn prediction, and proactive retention analytics framework developed for the **Unified Mentor** project specifications and aligned with European retail banking standards.

---

## 📂 Repository Directory Layout

```plaintext
├── data/
│   ├── bank_churn_raw.csv           # Raw source dataset (10,000 customers)
│   ├── bank_churn_clean.csv         # Cleaned analytical base with derived metrics
│   └── customers_segmented.csv      # Scored dataset with segment tags & risk tiers
├── models/
│   ├── churn_pipeline.pkl           # Serialized Gradient Boosting + ColumnTransformer pipeline
│   ├── model_benchmarks.csv         # Performance metrics across LR, RF, and GB
│   ├── feature_importances.csv      # Ranked feature importances
│   ├── segment_profiles.csv         # Granular segment summaries and financial exposure
│   └── model_metrics.json           # Machine-readable evaluation parameters
├── figures/
│   ├── eda_demographics.png         # Age distribution, country share, gender split
│   ├── eda_bivariate_churn.png      # Regional, age-cohort, and product churn curves
│   ├── eda_correlation_matrix.png   # Pearson correlation matrix heatmap
│   ├── model_benchmark.png          # Algorithm comparison chart
│   ├── feature_importance.png       # Primary churn drivers
│   └── segment_profile_heatmap.png  # Segment profile comparison heatmap
├── deliverables/
│   ├── Research_Paper.pdf           # Deliverable 1: Complete analytical research report
│   └── Executive_Summary.pdf        # Deliverable 3: Executive brief for bank leadership
├── app.py                           # Deliverable 2: Production Streamlit interactive web application
├── build_reports.py                 # Automated ReportLab PDF compilation script
├── requirements.txt                 # Project environment dependencies
├── .gitignore                       # Standard Python / environment ignore rules
└── README.md                        # Documentation & setup manual
```

---

## 📌 Deliverables Summary

| Deliverable | Location | Description |
| :--- | :--- | :--- |
| **1. Research Paper / Detailed Report** | `deliverables/Research_Paper.pdf` | Comprehensive research report detailing EDA, econometric segmentation, ML model architectures, and financial retention strategies. |
| **2. Streamlit Web Application** | `app.py` | Multi-module web app with Portfolio Explorer, Individual Churn Calculator, What-If Simulator, and CSV export. |
| **3. Executive Summary** | `deliverables/Executive_Summary.pdf` | High-impact 2-page brief for bank executive leadership and ECB governance committees. |
| **4. Reproducible Pipelines** | `build_reports.py`, `models/` | Script generating both PDFs from source data, alongside serialized ML pipelines. |

---

## 🔍 Key Findings from the European Banking Portfolio

1. **Geographic Risk Asymmetry:**
   - **Germany** records an alarming **32.44%** churn rate—nearly double **France (16.15%)** and **Spain (16.67%)**.
   - German churn is concentrated among high-deposit accounts (averaging €119,730), threatening core liquidity.
2. **Age Cohort Vulnerability:**
   - Customers aged **46–60** suffer from a **51.12%** churn rate, compared to just **7.52%** for younger accounts (<30).
   - This middle-aged group represents the bank's core wealth-accumulating demographic.
3. **The Multi-Product Paradox:**
   - Customers with **2 products** demonstrate optimal stickiness (**7.58%** churn).
   - Churn surges to **82.71%** for customers with **3 products**, and reaches **100%** for customers with **4 products**, revealing severe fee fatigue and product bundling friction.
4. **Behavioral Inactivity:**
   - Inactive customers churn at **26.85%**, compared to **14.27%** among digitally active members.

---

## 🏆 Model Benchmark Results

Models were benchmarked using an 80/20 stratified split:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 81.50% | 63.31% | 21.62% | 32.23% | 0.7812 |
| **Random Forest Classifier** | 86.60% | 81.17% | 44.47% | 57.46% | 0.8600 |
| **Gradient Boosting (Production)** | **86.85%** | **78.57%** | **48.65%** | **60.09%** | **0.8670** |

---

## 💻 Local Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/european-banking-churn-analytics.git
cd european-banking-churn-analytics
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Rebuild Deliverable PDFs (Optional)
```bash
python build_reports.py
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🌐 Cloud Deployment (Streamlit Cloud / Render)

1. Push this entire repository to your GitHub account:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Complete European Banking Churn Analytics project"
   git branch -M main
   git remote add origin https://github.com/<your-username>/european-banking-churn-analytics.git
   git push -u origin main
   ```
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your repository, select `app.py`, and deploy.
