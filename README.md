# AutoCATE: End-to-End, Automated Treatment Effect Estimation

![AutoCATE Logo](assets/AutoCATE Logo Name.jpg)

🚀 **AutoCATE** is an open-source Python package for automated, end-to-end estimation of Conditional Average Treatment Effects (CATE). Designed to simplify the complexities of causal inference, AutoCATE enables practitioners and researchers to quickly build robust ML pipelines for causal effect/heterogeneous treatment effect estimation in only *four lines of code*.

## ✨ Key Features
✔️ **Automated ML Pipelines**: Automatically builds pipelines and handles preprocessing, ML algorithm selection and hyperparameter optimization.

✔️ **Custom Evaluation Protocols**: Incorporates different risk measures (e.g., DR-risk, T-risk) and risk metrics (e.g., AUQC) tailored for causal inference.

✔️ **Flexible Ensemble Methods**: Selects and combines the optimal pipelines for improved robustness and accuracy.

✔️ **Low-Code API**: Effortlessly predict treatment effects with minimal setup.

## ⚙️ How It Works
AutoCATE operates in three stages:

1️⃣ **Evaluation**: Optimize the ML pipelines underlying the risk measure(s) (e.g., R-risk) for robust pipeline selection.

2️⃣ **Estimation**: Optimizing ML pipelines for CATE estimation, combining preprocessors, metalearners, and baselearners. 

3️⃣ **Ensembling**: Combines the top-performing pipelines to create a final model. 

## ❓ Why Choose AutoCATE?

🚨 Estimating causal effects requires dealing with **unique challenges**:

❌ Evaluation is a challenge due to lack of ground truth CATE and covariate shift caused by confounding variables.

❌ Causal metalearners combine different ML algorithms and are complex to tune.

❌ No clear, established practices for preprocessing and ensembling.

🎯 AutoCATE eliminates these barriers by **automating the entire process**, making state-of-the-art CATE estimation accessible for everyone.

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
