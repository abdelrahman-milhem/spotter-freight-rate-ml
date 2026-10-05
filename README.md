# Spotter ML Engineer Assessment: Freight Rate Prediction

An end-to-end Machine Learning solution for predicting freight load rates (posted_rate) using domain-aware feature engineering, data quality remediation, and a multi-objective Gradient Boosted Decision Tree (GBDT) ensemble.

---

## 📌 Project Overview

This repository contains the full submission artifacts for the Spotter Machine Learning Engineer assessment:
- **alidation_predictions.csv**: Predicted rates for all 12,000 holdout loads in alidation.csv.
- **december-chart-inputs.csv**: Predicted rates for the fixed Lexington-to-Fort Wayne December 2025 benchmark route (31 days).
- **Freight_Rate_ML_Assessment_Report.pdf** & **.docx**: Formal technical report detailing methodology, validation strategy, and performance metrics.
- **score.py Validation**: Full compliance and verification pass via score.py.

---

## 📊 Key Results & Benchmarks

| Metric | Baseline Model | Optimized Ensemble | Improvement |
| :--- | :---: | :---: | :---: |
| **MAE ($)** | .31 | **.33** | **-45.7% (+/load saved)** |
| **Error %** | 5.51% | **3.00%** | **-2.51% absolute** |
| **R² Score** | 0.8634 | **0.8715** | **+0.0081 (+87.2% variance explained)** |
| **RMSE ($)** | .59 | **.13** | **-.46** |

---

## 🛠️ Data Quality & Hygiene

1. **Negative Weights**: 292 entries with negative weights were corrected via absolute values (.abs()).
2. **Missing Value Imputation**:
   - Missing weights (300 records) were imputed using equipment-specific medians.
   - Missing market indices (374 records) were imputed using training set medians to strictly prevent data leakage.
3. **December Benchmark Enrichment**: Enriched missing latitude/longitude coordinates and extracted daily spot market signals from alidation.csv.

---

## 🔬 Feature Engineering Highlights

- **Estimated Base Cost**: distance * quote_signal (core baseline cost anchor).
- **Market Rate Pressure**: quote_signal * market_index (spot market premium).
- **Ton-Miles Transport Energy**: weight * distance and weight / (distance + 1).
- **Geospatial Proximity**: Haversine spherical distance and Manhattan coordinate deltas.
- **Temporal Seasonality**: Month, Day, Day-of-Week, Day-of-Year, and Weekend flags capturing Q4 surge trends.

---

## 🧠 Model Architecture & Ensemble

We engineered a specialized ensemble of **HistGradientBoostingRegressor** models optimizing an L1 absolute error loss (loss=\'absolute_error\') for robust outlier resistance:
1. **Rate-per-Mile Model (55% weight)**: Predicts normalized rate per mile (posted_rate / distance), excelling on long-haul routes.
2. **Residual Model (35% weight)**: Predicts price delta over baseline (posted_rate - estimated_base), capturing spot market volatility.
3. **Direct L1 Model (10% weight)**: Regularizing anchor model predicting total rate directly.

---

## 🚀 Quickstart & Reproduction

### 1. Installation
`ash
pip install -r requirements.txt
`

### 2. Run Training Pipeline & Scoring
`ash
python train.py
`
This script will:
1. Clean data and engineer features.
2. Train the ensemble models on all 48,000 labeled records.
3. Generate alidation_predictions.csv (12,000 loads).
4. Update december-chart-inputs.csv (31 loads).
5. Execute score.py and output validation confirmation + chart to scorer_results/candidate_december.png.

### 3. Evaluate Cross-Validation Metrics
`ash
python clean.py
`
