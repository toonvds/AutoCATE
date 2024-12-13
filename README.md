<p align="center">
  <img src="https://github.com/user-attachments/assets/2d84a0cb-2fa8-4e4d-b44b-d6537923dfa8" alt="AutoCATE Logo" width="800">
</p>

<h1 align="center">AutoCATE: End-to-End, Automated Treatment Effect Estimation</h1>
<p align="center">
  <i>— AutoML for causal effect estimation —</i>
  </p>

<p align="center">
  <a href="https://github.com/AutoCATE"><img src="https://img.shields.io/github/stars/AutoCATE?color=gold"></a>
  <a href="https://github.com/AutoCATE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

**AutoCATE** is an open-source Python package for automated, end-to-end estimation of Conditional Average Treatment Effects (CATE). Designed to simplify the complexities of causal inference, AutoCATE enables practitioners and researchers to quickly build robust ML pipelines for causal effect/heterogeneous treatment effect estimation in only *four lines of code*.

## ✨ Key Features
✔️ **Automated ML Pipelines**: Automatically builds pipelines and handles preprocessing, ML algorithm selection, hyperparameter optimization, and ensembling.

✔️ **Custom Evaluation Protocols**: Incorporates different risk measures (e.g., DR-risk, T-risk) and risk metrics (e.g., AUQC) tailored for causal inference.

✔️ **Low-Code API**: Effortlessly predict treatment effects with minimal setup.

## 🛠 Quick start

```python
from src.AutoCATE import AutoCATE

# Initialize AutoCATE
autocate = AutoCATE()

# Fit model on training data
autocate.fit(X_train, t_train, yf_train)

# Predict CATE for new data
cate_pred = autocate.predict(X_test)
```

## ⚙️ How It Works
AutoCATE operates in three stages:

1️⃣ **Evaluation**: Optimize the ML pipelines underlying the risk measure(s) (e.g., R-risk) for robust pipeline selection.

2️⃣ **Estimation**: Optimizing ML pipelines for CATE estimation, combining preprocessors, metalearners, and baselearners. 

3️⃣ **Ensembling**: Combines the top-performing pipelines to create a final model. 

## ❓ Why Choose AutoCATE?

Estimating causal effects requires dealing with **unique challenges** 🚨 

  ❌ Evaluation is a challenge due to lack of ground truth CATE and covariate shift caused by confounding variables.

  ❌ Causal metalearners combine different ML algorithms and are complex to tune.

  ❌ No clear, established practices for preprocessing and ensembling.

AutoCATE eliminates these barriers by **automating the entire process**, making state-of-the-art CATE estimation accessible for everyone.
