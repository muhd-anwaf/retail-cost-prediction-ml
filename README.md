# Retail Cost Prediction — Gradient Boosting Regression

A machine learning regression project that predicts product `cost` from retail store and product attributes using a `HistGradientBoostingRegressor`.

---

## Dataset

| File | Description |
|------|-------------|
| `Dataset/train.csv` | 360,336 rows with features + target (`cost`) |
| `Dataset/test.csv` | 240,224 rows without target (for submission) |
| `Dataset/sample_submission.csv` | Output predictions file |

**Features include:** store sales, unit sales, gross weight, store size, product attributes (low fat, recyclable), and store amenities (coffee bar, florist, etc.)

---

## Project Structure

```
├── Dataset/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── cost_prediction_gradient_boosting.ipynb
└── README.md
```

---

## Notebook Sections

1. Imports & Configuration
2. Data Loading
3. Data Inspection
4. Exploratory Data Analysis (EDA)
5. Preprocessing & Feature Engineering
6. Train / Validation / Test Split (70 / 15 / 15)
7. Model Training (Random Forest baseline + HistGradientBoosting)
8. Final Evaluation on Test Set
9. Feature Importance (Permutation Importance)
10. Generate Submission Predictions
11. Conclusion

---

## Model

**Primary:** `HistGradientBoostingRegressor`  
**Baseline:** `RandomForestRegressor`

HistGradientBoosting was chosen because:
- The features have near-zero linear correlation with the target — linear models (Lasso, Ridge) underfit
- It captures non-linear relationships and feature interactions
- Trains efficiently on large datasets via histogram-based binning
- Natively handles missing values

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| R²     | Proportion of variance explained |
| MAE    | Mean Absolute Error |
| RMSE   | Root Mean Squared Error |

---

## How to Run

1. Clone the repo and install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Open the notebook:

```bash
jupyter notebook cost_prediction_gradient_boosting.ipynb
```

3. Run all cells top to bottom.

---

## Key Design Decisions

- `id` column dropped — not a predictive feature
- Log transform applied **only** to `num_children_at_home` (skew ~1.85); all other features have |skew| < 1
- Target `cost` is nearly symmetric (skew ~0.02) — no transformation needed
- Hyperparameter tuning cell included but commented out (uncomment to run `RandomizedSearchCV`)

---

## Optional Improvements

- Hyperparameter tuning via `RandomizedSearchCV`
- Try `LightGBM` or `XGBoost` for potentially better performance
- Add k-fold cross-validation for more robust evaluation
- Engineer interaction features (e.g., `store_sqft × store_sales`)
- Use SHAP values for deeper model explainability
