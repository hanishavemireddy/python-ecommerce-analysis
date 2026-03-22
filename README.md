# 🛒 E-Commerce Analysis in Python

## Project Overview
End-to-end data analysis and machine learning project on an e-commerce 
order dataset. Covers data engineering, EDA, visualization, binary 
classification, hyperparameter tuning and model explainability.

---

## Dataset
- 10 customers, 10 products, 15 orders
- Source: CSV files → loaded into SQLite database → queried via SQLAlchemy
- Target variable: order status (delivered vs returned)

---

## Project Structure
```
python-ecommerce-analysis/
│
├── ecommerce_analysis.ipynb   ← main notebook
├── customers.csv              ← raw data
├── products.csv               ← raw data
├── orders.csv                 ← raw data
├── ecommerce.db               ← SQLite database
└── README.md
```

---

## Skills Demonstrated

### Data Engineering
- Created and read CSV files with Python
- Connected Python to SQLite via SQLAlchemy
- Queried database using raw SQL from Python (pd.read_sql)

### Data Cleaning
- Fixed datetime types with pd.to_datetime()
- Extracted date parts with .dt accessor
- Engineered features: days_as_customer, email_type

### Exploratory Data Analysis
- Inspected shape, dtypes, missing values
- Grouped and aggregated with .groupby()
- Created crosstabs and pivot tables

### Visualization
- Bar charts, line charts, stacked bar charts
- Heatmaps, confusion matrix
- Multi-model comparison charts with error bars

### Machine Learning
- Binary classification: Random Forest vs XGBoost
- Stratified K-Fold cross validation
- Multi-metric evaluation: Accuracy, ROC-AUC, F1, Precision, Recall
- GridSearchCV with multiple scoring metrics
- RandomizedSearchCV over continuous parameter distributions

### Model Explainability
- Feature importance via MDI (Random Forest built-in)
- SHAP values — global and individual prediction explanations
- LIME — local linear approximations per prediction
- Side-by-side SHAP vs LIME comparison

---

## Key Findings

### Business Insights
- Houston shows a fulfillment issue — all orders stuck in pending
- New York has a disproportionately high return rate
- Days as customer is the strongest predictor of order outcome
- Gmail vs non-Gmail customers show similar order behavior

### Model Performance
- Both Random Forest and XGBoost evaluated via 3-fold Stratified CV
- 5-fold CV produced NaN ROC-AUC — reduced to 3-fold given minority 
  class size of 3 samples (rule: n_splits ≤ n_minority_samples)
- XGBoost tuned via GridSearchCV optimizing ROC-AUC across 72 
  parameter combinations

### Model Explainability Findings
- SHAP and MDI disagree on Email Type importance — SHAP preferred
  as it is theoretically grounded and computed on test data
- SHAP and LIME agree on most important feature (days_as_customer)
  but diverge on magnitude of Email Type due to differences between
  global and local feature distributions

### Learning Curves
- XGBoost validation scores flat at 0.5 (random baseline) across 
  all training sizes — expected given only 13 training samples
- Random Forest more robust to small training sets due to bootstrap 
  aggregation
- Learning curves on this dataset demonstrate the technique rather 
  than serve as a reliable diagnostic — 500+ samples recommended 
  for meaningful curves
- Immediate need for more training data in a production setting

---

## Technical Notes

### Why SQLite over MySQL?
SQLite requires zero server setup and stores data as a portable .db 
file — ideal for reproducible portfolio projects. Switching to MySQL 
or PostgreSQL in production requires changing only the SQLAlchemy 
connection string — all downstream code remains identical.

### Why 3-Fold Instead of 5-Fold CV?
With 13 samples and 3 minority class examples, 5-fold CV creates test 
folds too small to guarantee both classes are present — making ROC-AUC 
undefined. General rule: n_splits ≤ n_minority_class_samples.

### Why SHAP Over MDI for Feature Importance?
MDI (Mean Decrease in Impurity) is biased toward features used in early 
tree splits and computed on training data. SHAP is theoretically grounded 
in Shapley values, computed on test data, accounts for feature 
interactions and explains individual predictions.

---

## Tools & Libraries
| Library | Purpose |
|---|---|
| pandas | Data manipulation |
| numpy | Numerical operations |
| matplotlib / seaborn | Visualization |
| sqlalchemy | Python-SQL connection |
| scikit-learn | ML models, CV, tuning |
| xgboost | Gradient boosting |
| shap | Model explainability |
| lime | Local prediction explanation |

---

## How to Run
1. Clone the repository
2. Install dependencies: `pip install pandas numpy matplotlib seaborn 
   sqlalchemy scikit-learn xgboost shap lime`
3. Open `ecommerce_analysis.ipynb` in Jupyter or VS Code
4. Run all cells top to bottom