# 🌡️ Short-Term Temperature Forecasting with High-Resolution Meteorological Sensors

### Hybrid Stacking Machine Learning Framework

This repository contains a machine learning framework for **short-term temperature forecasting** using high-resolution (10-minute interval) meteorological sensor data. The study compares classical time-series models (Prophet, SARIMA) with tree-based ML algorithms (Random Forest, XGBoost, CatBoost) and introduces a **Hybrid Stacking Ensemble** that delivers state-of-the-art performance.

---

## 🚀 Project Overview

* The dataset includes **20 atmospheric variables** collected from the Max Planck Institute weather station (2020).
* Target variable: **Temperature (T)**.
* Key preprocessing steps:

  * Timestamp conversion
  * Circular feature engineering for wind direction (sin/cos)
  * Missing-value interpolation
  * Strict chronological 80/20 train–test split
  * Min–Max scaling
* Models evaluated:

  * **SARIMA**, **Prophet**
  * **Random Forest**, **XGBoost**, **CatBoost**
  * **Hybrid Stacking Ensemble (RF + XGB + CatBoost)**
* Evaluation metrics: **MAE, RMSE, MAPE, R²**

---

## 🧠 Why Hybrid Stacking?

Based on results in the paper (Fig. 9–15), the Hybrid Ensemble:

* Achieved the **best overall performance**:

  * **MAE = 0.255°C**
  * **RMSE = 0.329°C**
  * **R² = 0.949**
* Outperformed individual ML models and dramatically surpassed Prophet/SARIMA.
* Captured rapid short-term fluctuations that traditional models completely smoothed out.
* Produced the **most stable and unbiased** error distribution.
* Maintained strong consistency even at extreme temperature values.

This demonstrates that ensemble learning is highly effective for dense, nonlinear meteorological time series.

---

## 📊 Dataset Description

The dataset contains 20 meteorological indicators, including:

* **T**: Temperature (°C)
* **rh**: Relative Humidity (%)
* **p**: Atmospheric Pressure (mbar)
* **wv**: Wind Speed (m/s)
* **rain**: Precipitation (mm)
* **SWDR**: Shortwave Radiation (W/m²)
* **PAR**: Photosynthetically Active Radiation
* Additional physical variables: VPmax, VPact, VPdef, sh, H2OC, rho, Tpot, etc.

Sampling frequency: **every 10 minutes**, yielding **52,560+ measurements per variable** in one year.

---

## ⚙️ Installation

```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt
```

Required libraries:

* pandas, numpy
* scikit-learn
* prophet
* xgboost
* catboost
* matplotlib, seaborn

---

## 🧪 Training the Models

Example usage:

```python
from models.hybrid import HybridStackingModel

model = HybridStackingModel()
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

---

## 📈 Results

* Prophet and SARIMA showed **significant underperformance** on high-frequency data (Prophet even produced **negative R²**).
* Random Forest, XGBoost, and CatBoost captured nonlinearities but had individual weaknesses at extremes.
* **Hybrid Stacking** combined their strengths and delivered the **most accurate and consistent** predictions.

For complete visual and numerical results, see the full paper:


---

## 🔮 Future Work

Potential improvements for next versions:

* Integration of CNN–Transformer hybrid architectures
* Use of satellite-based radiation estimations
* Multi-step forecasting with uncertainty quantification
* Real-time deployment with concept-drift adaptation
