# OLS Naive Model

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the simulated dataset
df = pd.read_csv("../data/final_simulated_panel.csv")

# Create log_sales
df["log_sales"] = np.log(df["sales"])

# Define model predictors (intentionally naive)
X = df[[
    "price",
    "ad_spend",
    "store_size",
    "area_income"
]]

y = df["log_sales"]

# Add constant
X = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X).fit()

print(model.summary())

# Evaluation Metrics

# RMSE
rmse = np.sqrt(np.mean((model.predict(X) - y)**2))

# R-squared
r2 = model.rsquared

# True parameters from the DGP
true_betas = {
    "const": 0,
    "price": -1.2,
    "ad_spend": 0.05,
    "store_size": 0.3,
    "area_income": 0.02
}

# Coefficient recovery
beta_errors = {}

for param in model.params.index:
    if param in true_betas:
        beta_errors[param] = abs(model.params[param] - true_betas[param])

print("\n--- Evaluation Summary ---")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print("\nCoefficient Recovery (absolute error):")
for k, v in beta_errors.items():
    print(f"  {k}: {v:.4f}")

# Visualizations

# Predicted vs True log(sales)
plt.figure(figsize=(6,6))
plt.scatter(y, model.predict(X), alpha=0.4)
plt.xlabel("True log(Sales)")
plt.ylabel("Predicted log(Sales)")
plt.title("Predicted vs. True - OLS Naive")
plt.savefig("../assets/figures/ols_pred_vs_true.png")
plt.close()

# Residual histogram
plt.figure(figsize=(6,4))
plt.hist(model.resid, bins=40)
plt.title("Residual Distribution - OLS Naive")
plt.savefig("../assets/figures/ols_residual_hist.png")
plt.close()
