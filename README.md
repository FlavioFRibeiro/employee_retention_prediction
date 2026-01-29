# HR Analytics — Employee Attrition Prediction

## Project goal
This project aims to **predict voluntary employee attrition** and, more importantly, **identify which categorical groups are most associated with higher risk**. The final intent is to help HR teams understand *which segments* (e.g., department, job role, education field, employee source) deserve focused retention strategies.

## Why the notebook is structured this way
The notebook is deliberately organized to mirror a professional analytics workflow:

1) **Data loading**
   - Ensures the notebook is reproducible by using a local `data/dataset.csv` path.

2) **Data quality checks**
   - Missing values and duplicates are verified early to avoid model bias or leakage.

3) **Target distribution**
   - Attrition is typically imbalanced; visualizing the target guides metric choice.

4) **Feature engineering**
   - Derived variables like `PriorYearsOfExperience` and `AverageTenure` capture patterns that raw columns may miss.

5) **Target definition**
   - The project focuses on *voluntary* exits; the notebook automatically adapts to different label schemes.

6) **Preprocessing pipeline**
   - Uses `ColumnTransformer` + `OneHotEncoder` + `StandardScaler` to keep transformations consistent across models.

7) **Baseline model (Logistic Regression)**
   - Provides a strong, interpretable benchmark with balanced class weights.

8) **Decision Tree (interpretability)**
   - Serves as an explainability tool; pruned depth improves readability and avoids overfitting.

9) **XGBoost (performance)**
   - Captures nonlinear interactions and usually yields the best AUC.

10) **Threshold tuning**
   - For retention use cases, recall is critical. Tuning the decision threshold allows us to trade precision for capturing more at-risk employees.

11) **Model comparison & recommendation**
   - Metrics are summarized side-by-side to align the model choice with business priorities.

---

## Key conclusions (from the current notebook)
- **Best overall discrimination (ROC AUC)**: XGBoost consistently separates “stay” vs. “leave” better than other models.
- **Best recall for retention**: XGBoost with a tuned threshold captures the most employees at risk, which is ideal for proactive retention.
- **Best interpretability**: Logistic Regression remains a reliable baseline with clear directional insights.
- **Decision Tree** is mainly used for interpretability and storytelling, not for top performance.

---

## What this means for the business
The primary outcome is **segment-level understanding**: we can identify which categorical groups (e.g., department, job role, education field, employee source) show higher attrition risk and target them with focused interventions. This aligns directly with the project’s objective of explaining *which groups are more likely to leave*, not just predicting individual outcomes.

---

## How to run
1) Place the dataset in `data/dataset.csv`.
2) Open `RH Analytics_2.0.ipynb`.
3) Run all cells from top to bottom.

> Note: The XGBoost tree visualization requires Graphviz installed in your environment.

## Project structure
```
employee_retention_prediction/
  data/
    dataset.csv
  RH Analytics_2.0.ipynb
  README.md
```

## Dependencies
- Python 3.10+
- pandas, numpy, seaborn, matplotlib
- scikit-learn
- xgboost (optional, for best performance)
- graphviz (optional, for tree rendering)

