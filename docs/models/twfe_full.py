

# models/twfe_full.py

# Two-Way Fixed Effects Champion Model

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../data/final_simulated_panel.csv")

# Create log_sales
df["log_sales"] = np.log(df["sales"])

# Relative price
df["relative_price"] = df["competitor_price"] / df["price"]

# Model with store and week fixed effects
formula = (
    "log_sales ~ price + relative_price + ad_spend + ad_spend_lag + "
    "competitor_count + store_size + area_income + manager_experience + "
    "manager_vacant + C(store_id) + C(week)"
)

# Fit model
model = smf.ols(formula=formula, data=df).fit()

print(model.summary())

# Evaluation metrics

pred = model.predict(df)
rmse = np.sqrt(np.mean((pred - df["log_sales"])**2))
r2 = model.rsquared

true_betas = {
    "price": -1.2,
    "relative_price": -0.8,
    "ad_spend": 0.05,
    "ad_spend_lag": 0.02,
    "competitor_count": -0.15,
    "store_size": 0.3,
    "area_income": 0.02,
    "manager_experience": 0.01,
    "manager_vacant": -0.05
}

beta_errors = {}
for name, true_val in true_betas.items():
    if name in model.params:
        beta_errors[name] = abs(model.params[name] - true_val)

print("\n--- Evaluation Summary ---")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print("\nCoefficient Recovery (absolute error):")
for k, v in beta_errors.items():
    print(f"  {k}: {v:.4f}")

# Visualizations

plt.figure(figsize=(6,6))
plt.scatter(df["log_sales"], pred, alpha=0.4)
plt.xlabel("True log(Sales)")
plt.ylabel("Predicted log(Sales)")
plt.title("Predicted vs. True - TWFE Champion")
plt.savefig("../assets/figures/twfe_pred_vs_true.png")
plt.close()

plt.figure(figsize=(6,4))
plt.hist(model.resid, bins=40)
plt.title("Residual Distribution - TWFE Champion")
plt.savefig("../assets/figures/twfe_residual_hist.png")
plt.close()
