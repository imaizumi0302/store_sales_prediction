📘 Japanese version → README_japanese.md

# Store Sales Time Series Forecasting

## Project Summary

- **Task**: Forecast daily sales for each *store × product family* for the next 16 days  
- **Approach**: Time series regression using **LightGBM**  
- **Key Contributions**:
  - Carefully designed **time-series cross-validation**
  - Leakage-free feature engineering
  - Quantitative evaluation of **CNN-based time-series embeddings**
- **Final Public Leaderboard Score**:
  - Single model: **0.42405**
  - **Ensemble model: 0.41542**
- **Rank**: **96 / 793** (Top ~12%)

This project emphasizes not only predictive accuracy, but also  
**reproducibility, validation integrity, and principled model selection**.

---

## Overview

This repository contains a complete solution for the Kaggle competition  
**Store Sales – Time Series Forecasting**.

The goal is to predict daily sales for each *store × product family*  
using historical sales data and auxiliary external information such as  
oil prices and holiday calendars.

The project focuses on:

- Time-aware validation strategies
- Strict prevention of data leakage
- Model and feature selection based on quantitative evidence

Rather than relying on leaderboard-specific tricks,  
the solution is designed with **real-world data science practices** in mind.

---

## Results

- **Final single model (Public LB)**: 0.42405  
- **Final ensemble model (Public LB)**: **0.41542**  
- **Rank**: 96 / 793 (Top ~12%)

The ensemble consistently outperformed the single model, and  
cross-validation scores showed strong alignment with Public Leaderboard results.

---

## Dataset

The following datasets provided by Kaggle were used:

- `train.csv` / `test.csv`: Daily sales by store and product family  
- `stores.csv`: Store metadata (city, state, type, cluster)  
- `oil.csv`: Daily oil prices (missing values imputed)  
- `holidays_events.csv`: National, regional, and local holidays  

For consistent preprocessing, training and test data were concatenated.  
Any features derived from sales values were computed **strictly using past data only**.

---

## Validation Strategy (Key Improvement)

The most significant performance improvement came from  
**revising the cross-validation (CV) design**.

### Initial Design (Before Improvement)

Originally, rolling and mean-based features were computed as follows:

- Statistics were calculated using data **prior to the earliest `train_end`** among all folds
- The same feature values were reused across all validation folds

While this approach avoided leakage, it suffered from a major drawback:

- Each fold failed to fully utilize all available historical data  
  within its own training period

---

### Revised Design (Final Approach)

To address this issue, feature generation was moved **inside the CV loop**:

- For each fold:
  - All data up to that fold’s `train_end` was used
  - Mean-based and rolling features were recomputed
- Validation data used fold-specific features only

This design:

- Maximized the use of available historical data
- Preserved strict leakage prevention
- Produced more informative features per fold

---

### Impact

- Significant improvement in cross-validation scores
- Better alignment between CV results and Public Leaderboard scores

This highlighted an important lesson:

> **In time-series tasks, validation design can have a larger impact than model choice itself.**

---

## Time-Series Cross-Validation Setup

| Fold | Training Period | Validation Period |
|---|---|---|
| 1 | 2013-01-01 → 2017-06-30 | 2017-07-01 → 2017-07-16 |
| 2 | 2013-01-01 → 2017-07-15 | 2017-07-17 → 2017-08-01 |
| 3 | 2013-01-01 → 2017-07-30 | 2017-07-31 → 2017-08-15 |

For all folds, sales-dependent features were computed using  
data **up to `train_end` only**.

---

## Feature Engineering

### Fold-Independent Features

- Date features: year, month, day, weekday, weekend flag  
- Oil price features: rolling averages (30 / 90 / 180 days)  
- Holiday features:
  - National / regional / local holiday flags
  - Aggregated holiday indicator
  - Special workday flag

### Fold-Dependent (Leakage-Safe) Features

- Mean-based (target encoding) features:
  - store, family, store × family, type, cluster
- Rolling sales statistics:
  - Windows: 3 / 7 / 30 days
  - Implemented with shifts to ensure past-only references

---

## Model

### Base Model: LightGBM

- Objective: regression (RMSLE evaluated in log space)
- Gradient boosting model chosen for robustness with large feature sets
- Hyperparameters optimized using Optuna

### Hyperparameter Optimization

- Optuna with Median Pruner
- 3-fold time-series cross-validation
- Strategy:
  1. Full hyperparameter search without CNN features
  2. Use the resulting best parameters as initialization
  3. Perform lightweight tuning when CNN embeddings are included

---

## SHAP Analysis and Feature Reduction

SHAP was used to analyze feature contributions to model predictions.

Key findings:

- Oil-related features showed **relatively high split importance**
- However, their **mean absolute SHAP values were extremely small**

This indicates that while oil features were frequently used for splits,
they did **not consistently move predictions in a single direction**.

### Feature Ablation Results

| Model Variant | CV Mean RMSLE | CV Std |
|---|---:|---:|
| All features | 0.40420 | 0.02295 |
| Without holiday features | 0.40091 | 0.02256 |
| **Without oil features** | **0.39794** | **0.01680** |
| Without oil & holiday features | 0.40012 | 0.01759 |

Removing oil-related features:

- Improved average CV performance
- Significantly reduced variance across folds

This suggests that oil features added noise rather than stable signal,
and their removal improved both **generalization and stability**.

---

## CNN-Based Time-Series Embedding (Experimental)

To capture short-term temporal patterns beyond rolling statistics,
a 1D CNN-based embedding was explored.

- Input: past 30 days of features
  - promotion flags
  - holiday flags
  - weekday / date features
  - oil prices
- Output: fixed-dimensional embedding vector
- CNN used as a **feature extractor only (no training)**

To reduce computation cost, embeddings were cached per fold during Optuna runs.

### Evaluation

- Training scores improved
- Validation scores did **not** show consistent gains

Based on quantitative evaluation, CNN embeddings were excluded from the final model
in favor of **simplicity, reproducibility, and inference efficiency**.

---

## Final Training and Prediction

- Final model trained on all data up to **2017-08-15**
- Forecast horizon: **2017-08-16 → 2017-08-31**
- Submission strategies:
  - Single final model
  - Ensemble of CV fold models + final model

The ensemble achieved the best Public Leaderboard score.

---

## Key Takeaways

- Validation design is critical in time-series problems
- SHAP-guided feature selection improves both accuracy and stability
- High-cost features (e.g., CNN embeddings) should be adopted only with clear validation gains
- Emphasizing reproducibility and interpretability leads to robust solutions

---

## Repository Structure

- `store_sales_new.ipynb` / `.py`: End-to-end pipeline (preprocessing → training → inference)
- `models/`: LightGBM models trained for each CV fold
- `submission.csv`: Single-model submission
- `submission_ensemble.csv`: Ensemble submission

---

## Notes

This repository intentionally documents not only the final model,
but also the **decision-making process** behind feature and model selection,
ensuring full reproducibility and transparency.

---

## License

This project is released under the **MIT License**.  
See the `LICENSE` file for details.

