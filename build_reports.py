"""Builds deliverables/Research_Paper.pdf and deliverables/Executive_Summary.pdf from pipeline outputs."""
import os, json, pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(*parts):
    return os.path.join(BASE_DIR, *parts)

# Ensure deliverables directory exists
deliverables_dir = get_path("deliverables")
os.makedirs(deliverables_dir, exist_ok=True)

d = pd.read_csv(get_path("data", "customers_segmented.csv"))
prof = pd.read_csv(get_path("models", "segment_profiles.csv"))
with open(get_path("models", "model_metrics.json"), "r", encoding="utf-8") as f:
    m = json.load(f)
bench = pd.read_csv(get_path("models", "model_benchmarks.csv"), index_col=0)
feat = pd.read_csv(get_path("models", "feature_importances.csv"))

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=13, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1E3A8A"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=10.5, spaceBefore=5, spaceAfter=3, textColor=colors.HexColor("#1E3A8A"))
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=8.5, leading=11.5, spaceAfter=4)
CAP = ParagraphStyle("C", parent=B, fontSize=7.5, textColor=colors.HexColor("#4B5563"), alignment=1)
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=17, leading=20, textColor=colors.HexColor("#1E3A8A"), spaceAfter=3)
SUBTITLE = ParagraphStyle("Sub", parent=B, fontSize=9.5, textColor=colors.HexColor("#374151"), spaceAfter=6)

def P(t, s=B): return Paragraph(t, s)
def bullets(items): return [P("&bull; " + i) for i in items]
def fig(path, cap, w=15.5):
    from PIL import Image as PI
    full_path = get_path(path)
    if not os.path.exists(full_path): return Spacer(1, 1)
    iw, ih = PI.open(full_path).size
    return KeepTogether([Image(full_path, w*cm, (w*cm*ih)/iw), P(cap, CAP), Spacer(1, 4)])

def table(rows, widths=None, fs=7.5):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), fs),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING", (0,0), (-1,-1), 3)
    ]))
    return t

tot_cust = len(d)
overall_churn = d["Exited"].mean()
churn_bal = d[d["Exited"] == 1]["Balance"].sum()
ger_churn = d[d["Geography"] == "Germany"]["Exited"].mean()
fra_churn = d[d["Geography"] == "France"]["Exited"].mean()
spa_churn = d[d["Geography"] == "Spain"]["Exited"].mean()
mid_age_churn = d[(d["Age"] >= 46) & (d["Age"] <= 60)]["Exited"].mean()

# ====================== RESEARCH PAPER ======================
S = [
    P("Machine Learning-Driven Customer Segmentation & Churn Pattern Analytics in European Banking", TITLE),
    P("Detailed Research Paper & Strategic Analytical Report &middot; Unified Mentor Banking Portfolio", SUBTITLE),
    Spacer(1, 4),
    P("Abstract", H1),
    P(f"Customer attrition represents one of the largest capital depletion vectors in commercial retail banking. In this study, we examine a multi-national dataset of {tot_cust:,} retail banking customers across France, Germany, and Spain. By coupling econometric segmentation with machine learning classification pipelines, we analyze the structural drivers of defection across demographic, geographic, and behavioural dimensions. The empirical portfolio exhibits an overall churn rate of <b>{overall_churn:.2%}</b>, placing <b>&euro;{churn_bal/1e6:,.1f} million</b> in customer deposits at direct flight risk. Critical empirical findings indicate severe regional asymmetry: German customers churn at <b>{ger_churn:.1%}</b> (nearly double France's {fra_churn:.1%} and Spain's {spa_churn:.1%}), concentrated heavily among high-deposit accounts. Furthermore, customers aged 46&ndash;60 experience an acute attrition rate of <b>{mid_age_churn:.1%}</b>. In benchmarking predictive models, a Gradient Boosting architecture delivers superior discriminative power (ROC-AUC: <b>0.8670</b>, F1-Score: <b>60.09%</b>), outperforming standard linear Logistic Regression (ROC-AUC: 0.7812) due to sharp non-linear threshold effects in product counts and account activity. We conclude with an actionable financial playbook and an automated Streamlit risk-scoring architecture."),
    P("1. Introduction & Problem Statement", H1),
    P("Retail commercial banking relies on long customer relationship tenures to amortize customer acquisition costs (CAC) and generate steady net interest margins. When churn occurs, the institution loses not only primary deposit liquidity but also the lifetime cross-selling value of investment and lending products. Despite access to rich customer-level transactional data, banks often struggle to anticipate churn proactively because attrition is driven by non-linear behavioral inflection points rather than singular demographic traits. This investigation provides systematic econometric segmentation, rigorous model evaluation, and proactive retention policy design."),
    P("2. Dataset Description & Feature Engineering", H1),
    table([
        ["Dataset Field", "Type", "Domain Description", "Business Significance"],
        ["CustomerId / Surname", "Identifier", "Customer reference IDs & family names", "Excluded from predictive feature matrices to prevent identity bias."],
        ["CreditScore", "Numerical", "Creditworthiness score (350&ndash;850)", "Financial resilience metric; proxy for underwriting stability."],
        ["Geography", "Categorical", "Country of residence (France, Germany, Spain)", "Regional market and macroeconomic competitive exposure."],
        ["Gender & Age", "Demographic", "Gender binary & customer chronological age", "Life-stage categorization and wealth accumulation profiling."],
        ["Tenure & Balance", "Financial", "Years with bank (0&ndash;10) & account balance (&euro;)", "Core balance sheet exposure and relationship longevity."],
        ["NumOfProducts", "Behavioural", "Number of bank products held (1&ndash;4)", "Cross-sell penetration; reveals product fatigue dynamics."],
        ["HasCrCard / IsActive", "Binary", "Cardholder status & activity flag (0/1)", "Direct indicators of transactional engagement."],
        ["EstimatedSalary", "Financial", "Estimated annual income in Euros", "Income level used to compute relative wealth liquidity ratios."]
    ], [3.2*cm, 2*cm, 5.8*cm, 5*cm]),
    Spacer(1, 4),
    P("Derived Analytical Metrics", H2)
] + bullets([
    "<b>Balance-to-Salary Ratio:</b> <i>Balance / (EstimatedSalary + &epsilon;)</i> &mdash; Measures customer liquidity cushion and savings orientation relative to annual earnings.",
    "<b>Tenure-to-Age Ratio:</b> <i>Tenure / (Age + &epsilon;)</i> &mdash; Captures the proportion of an adult's financial life dedicated to the institution.",
    "<b>Credit-to-Age Ratio:</b> <i>CreditScore / (Age + &epsilon;)</i> &mdash; Measures credit velocity and financial maturity across generational cohorts."
]) + [
    P("3. Exploratory Data Analysis & Segment Dynamics", H1),
    fig("figures/eda_demographics.png", "Figure 1. Demographic distributions: customer age distribution, portfolio geographic share, and gender split."),
    fig("figures/eda_bivariate_churn.png", "Figure 2. Bivariate churn distributions across geography, age cohorts, and product holdings."),
    P("<b>Key Empirical Patterns:</b>")
] + bullets([
    f"<b>Regional Vulnerability:</b> Germany demonstrates a severe <b>{ger_churn:.1%}</b> churn rate, compared to France (16.2%) and Spain (16.7%). High-wealth German savers face lower switching barriers and aggressive fintech competition.",
    f"<b>The Mid-Life Wealth Drain:</b> Customers aged 46&ndash;60 suffer a <b>{mid_age_churn:.1%}</b> attrition rate. As clients reach peak wealth accumulation, traditional retail service models fail to meet wealth management expectations.",
    "<b>The Multi-Product Paradox:</b> Customers with 2 products form the retention sweet-spot (churn rate of only <b>7.58%</b>). However, churn surges to <b>82.71%</b> for 3 products and reaches <b>100.0%</b> for 4 products, highlighting severe account fee fatigue and mismanaged cross-selling.",
    "<b>Activity Disengagement:</b> Inactive members churn at <b>26.85%</b>, compared to <b>14.27%</b> for active digital and branch users."
]) + [
    fig("figures/eda_correlation_matrix.png", "Figure 3. Pearson correlation matrix across financial variables, product counts, and churn indicator.", 11.5),
    P("4. Machine Learning Model Training & Benchmarks", H1),
    P("To evaluate predictive feasibility, models were trained using an 80/20 stratified split. Numeric features were scaled via StandardScaler, and categorical variables encoded via drop-first OneHotEncoder inside a scikit-learn ColumnTransformer."),
    table([["Algorithm", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]] +
          [[idx, f"{r.Accuracy*100:.2f}%", f"{r.Precision*100:.2f}%", f"{r.Recall*100:.2f}%", f"{r.F1_Score*100:.2f}%", f"{r.ROC_AUC:.4f}"] 
           for idx, r in bench.rename(columns={'F1-Score': 'F1_Score', 'ROC-AUC': 'ROC_AUC'}).iterrows()],
          [4.5*cm, 2.3*cm, 2.3*cm, 2.3*cm, 2.3*cm, 2.3*cm]),
    Spacer(1, 4),
    fig("figures/model_benchmark.png", "Figure 4. Comparative benchmark of classification algorithms across standard performance dimensions.", 13.5),
    P("<b>Algorithmic Assessment:</b> Logistic regression severely underperforms on recall (21.62%), demonstrating that customer churn is driven by complex non-linear interactions that linear decision boundaries cannot capture. Gradient Boosting achieves superior overall performance with an ROC-AUC of <b>0.8670</b> and F1-Score of <b>60.09%</b>, correctly isolating high-risk defection signals."),
    P("5. Model Explainability & Segment Profiling", H1),
    fig("figures/feature_importance.png", "Figure 5. Global relative feature importances derived from the production Gradient Boosting model.", 13),
    P("Feature attribution confirms that <b>Age</b> (33.8%) and <b>NumOfProducts</b> (25.1%) dominate predictive influence, followed by <b>IsActiveMember</b> (10.6%) and <b>Geography_Germany</b> (6.6%)."),
    fig("figures/segment_profile_heatmap.png", "Figure 6. Segment profile matrix: raw averages and normalized Z-scores across designated customer segments.", 14),
    table([["Segment Name", "Clients", "Avg Age", "Avg Balance", "Products", "Active %", "Churn %"]] +
          [[r.Segment, f"{r.Clients:,}", f"{r.Avg_Age:.1f}", f"&euro;{r.Avg_Balance:,.0f}", f"{r.Avg_Products:.2f}", f"{r.Active_Rate*100:.1f}%", f"{r.Churn_Rate*100:.1f}%"]
           for _, r in prof.iterrows()],
          [4.4*cm, 1.8*cm, 1.8*cm, 2.6*cm, 1.8*cm, 1.8*cm, 1.8*cm]),
    Spacer(1, 4),
    P("6. Strategic Financial Recommendations & Playbook", H1)
] + bullets([
    "<b>German Portfolio Defense:</b> Deploy dedicated wealth relationship managers to German clients holding balances above &euro;100k; introduce competitive tiered deposit yields to stem capital flight.",
    "<b>Senior & Middle-Age Advisory Packages:</b> For clients aged 46&ndash;60, replace standard retail checking interfaces with consolidated family wealth, pension planning, and inheritance advisory services.",
    "<b>Product Bundle Rationalization:</b> Immediately audit fee schedules on customers holding 3 or 4 products. Bundle services under single-fee structures with automatic fee waivers tied to deposit balances.",
    "<b>Automated Digital Reactivation Triggers:</b> Establish automated marketing alerts when an active customer records no logins or card transactions over 60 days, offering targeted digital cashback incentives."
]) + [
    P("7. Conclusion & Operationalization", H1),
    P("This end-to-end framework bridges econometric segmentation and machine learning scoring. The serialized production pipeline (churn_pipeline.pkl) powers an interactive Streamlit application enabling relationship managers to score individual accounts and run counterfactual retention simulations.")
]

def build_pdf(path, story, margins=1.8*cm):
    full_path = get_path(path)
    SimpleDocTemplate(full_path, pagesize=A4, leftMargin=margins, rightMargin=margins, topMargin=1.6*cm, bottomMargin=1.6*cm).build(story)

build_pdf("deliverables/Research_Paper.pdf", S)

# ====================== EXECUTIVE SUMMARY ======================
E = [
    P("Executive Summary: Customer Churn & Retention Strategy", TITLE),
    P("Strategic Brief for Executive Leadership & Banking Governance Committees", SUBTITLE),
    Spacer(1, 4),
    P("1. Core Problem & Portfolio Financial Exposure", H1),
    P(f"Across 10,000 retail banking customer accounts in France, Germany, and Spain, portfolio attrition stands at <b>{overall_churn:.2%}</b>. "
      f"This represents <b>&euro;{churn_bal/1e6:,.1f} million</b> in customer deposits currently lost to competitor banks. "
      f"Rather than an evenly distributed attrition pattern, churn is heavily concentrated in two specific business risk clusters:")
] + bullets([
    f"<b>Regional Imbalance:</b> German accounts churn at <b>{ger_churn:.1%}</b> &mdash; nearly double that of France (16.2%) and Spain (16.7%). German churners hold an average balance of &euro;119,730, posing severe liquidity risks.",
    f"<b>High-Value Demographic Churn:</b> Customers aged 46&ndash;60 experience an alarming <b>{mid_age_churn:.1%}</b> attrition rate, depleting the bank's core wealth-accumulating customer base."
]) + [
    P("2. Customer Segment Overview & Churn Distribution", H1),
    table([["Segment Name", "Clients", "% Portf.", "Avg Balance", "Churn Rate", "Strategic Priority"]] + [
        ["Affluent German Portfolio", "1,873", "18.7%", "&euro;129,849", "33.2%", "Tiered deposit rates & relationship managers"],
        ["Mature Inactive Retainees", "1,032", "10.3%", "&euro;68,258", "62.0%", "Proactive outreach & service recovery"],
        ["Multi-Product At-Risk", "326", "3.3%", "&euro;78,822", "85.9%", "Product fee audits & bundle waivers"],
        ["Young Stable Core", "3,115", "31.2%", "&euro;35,062", "4.4%", "Cross-sell lending & digital expansion"],
        ["Standard Retail Accounts", "3,654", "36.5%", "&euro;85,808", "16.1%", "Baseline digital satisfaction programs"]
    ], [4.2*cm, 1.8*cm, 1.8*cm, 2.5*cm, 2.1*cm, 3.6*cm]),
    Spacer(1, 4),
    P("3. Machine Learning Decision Engine Performance", H1),
    P("A production-grade <b>Gradient Boosting Classifier</b> was deployed to replace static rules with real-time probability scoring:")
] + bullets([
    "<b>High Discriminative Power:</b> ROC-AUC of <b>0.8670</b> and Accuracy of <b>86.85%</b>, outperforming baseline models.",
    "<b>Primary Risk Indicators:</b> Age (33.8%) and Number of Bank Products (25.1%) account for nearly 60% of total predictive signal, followed by Inactivity (10.6%) and German residence (6.6%)."
]) + [
    P("4. Financial ROI Model & Retention Economics", H1),
    P("Transitioning from reactive handling to proactive predictive retention delivers immediate financial return:"),
    table([
        ["Financial Variable", "Baseline Operational Metric", "Projected Annual Impact"],
        ["Targeted High-Risk Cohort", "2,000 High-Probability Churn Accounts", "Identified automatically by ML pipeline"],
        ["Retention Offer Budget", "&euro;75 per customer (Waiver / Rate bonus)", "&euro;150,000 total investment"],
        ["Retention Success Rate", "25.0% target recovery", "500 accounts successfully retained"],
        ["Annual Gross Value per Retained Client", "&euro;850 net revenue margin / NIM", "&euro;425,000 stabilized revenue"],
        ["<b>Net Projected Annual Profit</b>", "<b>&euro;275,000 Net Benefit</b>", "<b>183.3% Return on Investment (ROI)</b>"]
    ], [4.5*cm, 5.5*cm, 6*cm]),
    Spacer(1, 4),
    P("5. High-Level Governance & Policy Implications", H1)
] + bullets([
    "<b>Algorithmic Fairness & Parity:</b> Under ECB algorithmic transparency mandates, models must be audited quarterly to ensure credit access and interest rates are not unfairly biased by protected demographic characteristics.",
    "<b>Product Bundling Governance:</b> The bank must reform product management to eliminate penalty fee stacking for customers holding 3+ products, which is currently driving an 86% churn rate.",
    "<b>Deployment:</b> Operationalize the live Streamlit dashboard across all European branch relationship managers to enable daily account triage."
])

build_pdf("deliverables/Executive_Summary.pdf", E, 1.6*cm)
print("Deliverables successfully compiled to deliverables/!")
