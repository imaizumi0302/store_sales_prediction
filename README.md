📘 Japanese version → README_ja.md

# Store Sales Time Series Forecasting

## Project Summary

**Task**
Forecast daily sales for each *store × product family* for the next **16 days**.

**Approach**
Time-series regression models primarily based on **gradient boosting (LightGBM)**.

**Key Contributions**

* Carefully designed **time-series cross-validation**
* Leakage-aware **feature engineering**
* Experimental evaluation of **CNN-based time-series embeddings**

**Final Results (Public Leaderboard)**

* Single model: **0.42405**
* Ensemble model: **0.41542**
* Rank: **96 / 793** (Top ~12%)

This project emphasizes not only predictive performance, but also **reproducibility, validity, and decision-making processes** in modeling.

---

## Overview

This repository contains a solution for the Kaggle competition
**Store Sales – Time Series Forecasting**.

The objective is to predict daily sales for each *store × product family* using historical sales records and auxiliary external data.

Rather than relying on leaderboard-specific tricks, this project focuses on:

* Time-series–aware cross-validation
* Leakage-safe feature engineering
* Quantitative and reproducible model selection

---

## Results

* Final single model (Public LB): **0.42405**
* Ensemble model (Public LB): **0.41542**
* Rank: **96 / 793** (Top ~12%)

The ensemble model consistently outperformed the single model.
Moreover, the cross-validation scores showed good alignment with Public Leaderboard performance, indicating a reliable validation strategy.

---

## Dataset

The following datasets provided by Kaggle were used:

* `train.csv` / `test.csv`
  Daily sales for each *store × product family*
* `stores.csv`
  Store metadata (city, state, store type, cluster)
* `oil.csv`
  Daily oil prices (missing dates were imputed)
* `holidays_events.csv`
  Holiday information (national, regional, local)

To ensure consistency in feature generation:

* `train` and `test` were concatenated during preprocessing
* Any sales-dependent features were computed **using past data only**

---

## Validation Strategy

The most significant performance improvement in this project came from redesigning the **cross-validation (CV) strategy**.

### Initial Design (Before Improvement)

Initially, rolling and aggregated features (e.g., by store or product family) were computed as follows:

* Use only the period before the *earliest train_end* across all folds
* Apply the same features to the validation period of every fold

This approach was safe in terms of leakage prevention, but had a major drawback:

* It did **not fully utilize all available historical data** within each fold

---

### Improved Design (Final Approach)

To address this limitation, feature generation was moved **inside the CV loop**.

For each fold:

* All data up to the fold-specific `train_end` was used
* Rolling and aggregated features were recomputed per fold
* Validation periods used features generated exclusively for that fold

This allowed:

* Maximum use of available historical data per fold
* Leakage prevention with richer feature representations

---

### Effect

As a result:

* Cross-validation scores improved significantly
* Alignment between CV and Public LB scores improved
* Overall model performance became more stable and reliable

This experience reinforced the importance of **data splitting and information availability timing** in time-series tasks—often more critical than model choice itself.

---

## Cross-Validation Splits

| Fold | Training Period         | Validation Period       |
| ---- | ----------------------- | ----------------------- |
| 1    | 2013-01-01 → 2017-06-30 | 2017-07-01 → 2017-07-16 |
| 2    | 2013-01-01 → 2017-07-15 | 2017-07-17 → 2017-08-01 |
| 3    | 2013-01-01 → 2017-07-30 | 2017-07-31 → 2017-08-15 |

For all folds, any feature depending on sales values used **only data prior to `train_end`** to avoid leakage.

---

## Feature Engineering

### Fold-Independent Features

* **Date features**: year, month, day, weekday, weekend flag
* **Oil price features**: rolling means (30 / 90 / 180 days)
* **Holiday features**:

  * National / regional / local holiday flags
  * Unified holiday flag
  * Special workday flag

---

### Fold-Dependent (Leakage-Safe) Features

**Aggregated features (target encoding style)**:

* `store`
* `family`
* `store × family`
* `store type`
* `cluster`

**Rolling statistics**:

* Windows: 3 / 7 / 30 days
* Shifted to ensure only past values are referenced

---

## Model

* **Base model**: LightGBM
* **Objective**: Regression (evaluated using RMSLE in log space)
* Selected for robustness to large-scale, high-dimensional features
* Hyperparameters optimized using Optuna

---

## Hyperparameter Tuning

* **Framework**: Optuna with Median Pruner
* **Validation**: 3-fold time-series CV

### Strategy

* Full tuning without CNN features
* Best parameters reused as initialization
* Lightweight tuning when CNN embeddings were enabled

---

## CNN-Based Time-Series Embedding (Experimental)

To capture short-term temporal patterns not fully represented by rolling statistics, a **1D CNN-based time-series embedding** was implemented.

**Input (past 30 days)**:

* Promotion flags
* Holiday flags
* Day-of-week / date features
* Oil price

**Output**:

* Fixed-dimensional embedding vector

The CNN was used **only as a feature extractor** and not trained jointly with the model.

To reduce computational cost:

* Embeddings were cached per fold
* No recomputation occurred during Optuna optimization

---

## Evaluation and Decision

* CNN embeddings appeared in feature importance and were utilized by the model
* Training scores improved, but validation scores did not improve consistently

Based on quantitative evaluation:

* CNN features were excluded from the final model
* Priority was given to **reproducibility, simplicity, and inference cost**

---

## Final Training and Prediction

* Final model trained using all data up to **2017-08-15**
* Forecast period: **2017-08-16 → 2017-08-31**

**Submitted models**:

* Single final model
* Ensemble of CV fold models + final model

The ensemble achieved the best Public Leaderboard score.

---

## Key Takeaways

* In time-series tasks, **validation design strongly influences performance**
* Strong feature engineering can outperform more complex models
* High-cost features (e.g., CNNs) should be adopted only with quantitative justification
* Model complexity should be balanced against reproducibility and interpretability

---

## Repository Structure

* `store_sales_new.ipynb / .py`
  End-to-end pipeline from preprocessing to training and prediction
* `models/`
  LightGBM models trained for each fold
* `submission.csv`
  Submission using the single final model
* `submission_ensemble.csv`
  Ensemble submission file

---

## Notes

Rather than blindly adopting complex models for performance gains, this project emphasizes **decision-making based on validation results**.

The entire workflow, including these decisions, is published in a **reproducible form**.

---

## License

This project is released under the **MIT License**.
See the `LICENSE` file for details.



