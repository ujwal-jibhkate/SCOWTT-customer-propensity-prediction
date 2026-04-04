# Predicting Customer Purchase Propensity & Order Value

## Overview

This project models customer purchasing behavior on the Olist Brazilian E-Commerce platform using historical transactional data. The goal is to score each customer for:

1. **Purchase Propensity** : the probability (0–1) that a customer will place an order in the next 90 days.
2. **Predicted Order Value** : the expected monetary value of that order.

These two scores are combined into an **Expected Value** metric (propensity × predicted value) to rank customers for targeted advertising.

---

## Dataset

The dataset contains ~100K orders from 2016–2018 across 11 CSV files.

**Tables used (7):**

| Table | Rows | Description |
|---|---|---|
| `olist_customers_dataset` | 99,441 | Customer demographics and location |
| `olist_orders_dataset` | 99,441 | Order-level data with timestamps and status |
| `olist_order_items_dataset` | 112,650 | Item-level price and freight per order |
| `olist_order_payments_dataset` | 103,886 | Payment method, installments, value |
| `olist_order_reviews_dataset` | 99,224 | Review scores and text |
| `olist_products_dataset` | 32,951 | Product category and attributes |
| `product_category_name_translation` | 71 | Portuguese → English category mapping |

**Tables excluded (4):** Geolocation (redundant with customer city/state), sellers (low signal for customer behavior), closed deals and marketing leads (B2B seller acquisition funnel, not customer-facing).

---

## Methodology

### Temporal Split

To prevent data leakage, the dataset is split temporally:

- **Feature window:** 2016-09-04 to 2018-07-01 (22 months), used to compute customer-level features
- **Target window:** 2018-07-01 to 2018-10-01 (90 days), used to define labels

A 30-day target window was initially tested but yielded insufficient positive examples (all 9 qualifying orders were canceled). The 90-day window provides 298 repeat customers out of 83,748 total (0.35% positive rate).

### Feature Engineering

33 features were engineered across 6 groups, aggregated at the `customer_unique_id` level:

| Group | Features | Description |
|---|---|---|
| Order count & status | 10 | Total orders + count pivoted by each order status |
| Monetary & payment | 7 | Total spend, installments, payment value split by type (credit card, boleto, voucher, debit) |
| Item & product | 7 | Product count, seller diversity, order size, price, freight |
| Review engagement | 4 | Review count, average score, title length, body length |
| Recency | 4 | Days since last purchase, shipment, delivery, review |
| Preferences | 2 | Preferred product category, preferred payment method |

After pairwise correlation analysis (threshold |r| > 0.7), 7 redundant features were removed. Categorical features were one-hot encoded (top-10 product categories + other), resulting in **40 model-ready features**.

Full feature documentation is available in [`data_reports/Feature_Engineering.xlsx`](data_reports/Feature_Engineering.xlsx).

### Null Handling

| Feature(s) | Nulls | Strategy | Rationale |
|---|---|---|---|
| Payment columns | 1 each | Fill 0 | No payment record = no payment through that channel |
| avg_order_price, avg_order_freight | 620 | Fill median | Real customers with missing item data; median robust to skew |
| days_since_last_shipped | 1,473 | Fill column max | Never shipped = longest time; tree models interpret correctly |
| days_since_last_delivered | 2,482 | Fill column max | Same logic; delivery is downstream of shipping |
| days_since_last_review | 651 | Fill column max | No review = least engaged |

---

## Modeling

A two-stage framework was used: classification for purchase propensity, then regression for order value.

### Stage 1: Purchase Propensity (Classification)

All models were tuned via `RandomizedSearchCV` with 3-fold CV, optimizing for PR-AUC.

| Model | Best Hyperparameters | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|
| Logistic Regression | C=1, penalty=l2, solver=saga | 0.5995 | 0.0061 |
| Random Forest | n_estimators=200, max_depth=3 | 0.6040 | 0.0063 |
| XGBoost | n_estimators=100, max_depth=6, lr=0.01 | 0.5970 | 0.0066 |
| **Balanced Random Forest** | **n_estimators=100, max_depth=7, min_samples_leaf=10** | **0.6171** | **0.0077** |
| LightGBM | n_estimators=100, max_depth=3, lr=0.05, scale_pos_weight=100 | 0.6044 | 0.0073 |

**Selected model:** Balanced Random Forest (best ROC-AUC and tied for best PR-AUC).

![PR and ROC Curves](outputs/pr_roc_curves.png)
![Model Comparison](outputs/model_comparison.png)

**Primary metric:** PR-AUC, chosen over ROC-AUC because with 0.35% positive rate, ROC-AUC can appear deceptively high while precision remains poor. PR-AUC focuses on the model's ability to identify the rare positive class.

### Stage 2: Order Value (Regression)

Trained on the 234 customers who purchased in the target window. Target was log-transformed (`log(1 + value)`) to handle right skew. Metrics are reported on the original dollar scale after inverse transform.

| Model | Best Hyperparameters | MAE | RMSE |
|---|---|---|---|
| Ridge Regression | alpha=100, solver=lsqr | $60.24 | $94.38 |
| **Random Forest** | **n_estimators=200, max_depth=3, min_samples_leaf=5** | **$55.75** | **$91.80** |
| XGBoost | n_estimators=100, max_depth=5, lr=0.01 | $55.78 | $92.14 |

**Selected model:** Random Forest Regressor (lowest MAE).

![Predicted vs Actual](outputs/regression_pred_vs_actual.png)

### Top Features (Balanced Random Forest: Propensity)

| Rank | Feature | Importance |
|---|---|---|
| 1 | days_since_last_delivered | 0.163 |
| 2 | days_since_last_purchase | 0.155 |
| 3 | tot_order_freight_value | 0.103 |
| 4 | total_payment_value | 0.091 |
| 5 | tot_pymt_credit_card | 0.062 |

![Feature Importance](outputs/feature_importance.png)

Recency features dominate, confirming that how recently a customer interacted is the strongest predictor of repeat purchase. Payment behavior and order value features provide secondary signal.

---

## Inference Pipeline

The inference notebook (`04_inference.ipynb`) scores all 83,748 customers and produces a ranked output:

- **Propensity score:** 0–1 probability from the Balanced Random Forest
- **Predicted order value:** Dollar estimate from the Random Forest Regressor
- **Expected value:** Propensity × predicted value, the ranking metric for ad targeting

Output is saved to `outputs/customer_scores.csv`.

![Score Distributions](outputs/score_distributions.png)
---

## Key Findings

- **Repeat purchase prediction is inherently difficult** in this dataset. 97% of customers made exactly one purchase, making their feature profiles nearly identical regardless of whether they return. Five different model architectures (linear, bagging, boosting, balanced sampling) all converge to similar performance, confirming the signal ceiling is in the data, not the model.
- **Recency is the strongest signal.** Days since last delivery and last purchase are the top two features by importance, consistent with established RFM (Recency, Frequency, Monetary) frameworks.
- **Payment behavior carries meaningful signal.** Total payment value, credit card usage, and installment behavior are top-10 features, how a customer pays reveals their engagement level.
- **The two-stage propensity × value framework** provides a principled approach to customer ranking even with limited model discrimination, as it combines both likelihood and value into a single actionable score.

---

## Repository Structure

```
SCOWTT/
├── datasets/                              # Raw CSVs (not committed, see setup)
├── processed/                             # Aggregated modelling datasets
│   ├── modelling_dataset.csv
│   └── user_features.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb          # Schema, quality checks, profiling reports
│   ├── 02_feature_engineering.ipynb       # User-level feature construction
│   ├── 03_modelling.ipynb                 # EDA, model training, evaluation
│   └── 04_inference.ipynb                 # Score all customers, final output
├── data_reports/
│   ├── Feature_Engineering.xlsx           # Feature documentation (4 sheets)
│   └── *_report.html                     # ydata-profiling reports per source file
├── outputs/
│   ├── best_propensity_model.pkl          # Saved Balanced Random Forest
│   ├── best_value_model.pkl               # Saved Random Forest Regressor
│   ├── scaler.pkl                         # StandardScaler (for LR only)
│   ├── customer_scores.csv                # Final scored output
│   ├── pr_roc_curves.png                  # PR and ROC curves
│   ├── model_comparison.png               # Model comparison bar charts
│   ├── feature_importance.png             # Top 15 feature importances
│   ├── regression_pred_vs_actual.png      # Predicted vs actual scatter
│   └── score_distributions.png            # Propensity, value, expected value distributions
├── pyproject.toml                         # uv project configuration
├── uv.lock                                # Reproducible dependency lock
├── .gitignore
└── README.md
```

---

## Setup

1. Install [uv](https://docs.astral.sh/uv/):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone and install dependencies:
   ```bash
   git clone https://github.com/ujwal-jibhkate/SCOWTT.git
   cd SCOWTT
   uv sync
   ```

3. Place all CSVs in `datasets/`.

4. Run notebooks in order:
   ```
   01_data_exploration.ipynb → 02_feature_engineering.ipynb → 03_modelling.ipynb → 04_inference.ipynb
   ```

**Python version:** 3.12.12

---

## Future Improvements

- Rolling backtests across multiple monthly snapshots for more robust evaluation
- Customer × month grain to capture temporal behavioral shifts
- Status-pivoted payment and review features (payment value by order status, review count by status)
- SHAP-based feature attribution for model interpretability
- Calibration of propensity scores using Platt scaling or isotonic regression
- Delivery experience features (on-time vs late delivery, estimated vs actual delivery gap)

---

## Use of AI Tools

AI tools were used for:
- Debugging code and resolving library-specific issues
- Documentation and reporting assistance

All code was written, understood, and validated by the author.