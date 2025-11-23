# Data Visualization Script for Simulated Store Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
base_dir = Path(__file__).resolve().parents[1]
df = pd.read_csv(base_dir / "data" / "final_simulated_panel.csv")

# Create output directory
fig_dir = base_dir / "assets" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# 1. Sales vs price scatterplot
plt.figure(figsize=(6, 5))
plt.scatter(df["price"], df["sales"], alpha=0.3)
plt.xlabel("Price")
plt.ylabel("Sales")
plt.title("Sales vs Price")
plt.savefig(fig_dir / "sales_vs_price.png")
plt.close()

# 2. Sales vs advertising scatterplot
plt.figure(figsize=(6, 5))
plt.scatter(df["ad_spend"], df["sales"], alpha=0.3)
plt.xlabel("Advertising Spend")
plt.ylabel("Sales")
plt.title("Sales vs Advertising Spend")
plt.savefig(fig_dir / "sales_vs_ad_spend.png")
plt.close()

# 3. Sales vs store size scatterplot
plt.figure(figsize=(6, 5))
plt.scatter(df["store_size"], df["sales"], alpha=0.3)
plt.xlabel("Store Size")
plt.ylabel("Sales")
plt.title("Sales vs Store Size")
plt.savefig(fig_dir / "sales_vs_store_size.png")
plt.close()

# 4. Sales vs manager experience with vacancy as an experience bin
df["experience_bin"] = pd.cut(
    df["manager_experience"],
    bins=[0, 2, 5, 10, 20, 50],
    labels=["0-2", "3-5", "6-10", "11-20", "21+"],
    include_lowest=True
)

df.loc[df["manager_vacant"] == 1, "experience_bin"] = "Vacant"

sales_by_exp = df.groupby("experience_bin")["sales"].mean().reindex(
    ["Vacant", "0-2", "3-5", "6-10", "11-20", "21+"]
)

plt.figure(figsize=(7, 5))
sales_by_exp.plot(kind="bar")
plt.xlabel("Manager Experience (years or vacant)")
plt.ylabel("Average Sales")
plt.title("Sales by Manager Experience or Vacancy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(fig_dir / "sales_by_manager_experience.png")
plt.close()

# 5. Sales vs relative price scatterplot
plt.figure(figsize=(6, 5))
plt.scatter(df["relative_price"], df["sales"], alpha=0.3)
plt.xlabel("Relative Price (Competitor Price / Own Price)")
plt.ylabel("Sales")
plt.title("Sales vs Relative Price")
plt.savefig(fig_dir / "sales_vs_relative_price.png")
plt.close()

# 6. Sales by binned relative price
df["relative_price_bin"] = pd.cut(
    df["relative_price"],
    bins=[0.0, 0.8, 0.95, 1.05, 1.2, 5.0],
    labels=["<0.8", "0.80-0.95", "0.95-1.05", "1.05-1.20", "1.20+"],
    include_lowest=True
)

sales_by_rel_price = df.groupby("relative_price_bin")["sales"].mean()

plt.figure(figsize=(7, 5))
sales_by_rel_price.plot(kind="bar")
plt.xlabel("Relative Price Range")
plt.ylabel("Average Sales")
plt.title("Sales by Relative Price Bin")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(fig_dir / "sales_by_relative_price_bin.png")
plt.close()

# 7. Sales over time for 5 random stores
unique_stores = df["store_id"].unique()
np.random.seed(42)
sample_stores = np.random.choice(unique_stores, size=5, replace=False)

plt.figure(figsize=(10, 6))
for s in sample_stores:
    sub = df[df["store_id"] == s].sort_values("week")
    plt.plot(sub["week"], sub["sales"], label=f"Store {s}")

plt.xlabel("Week")
plt.ylabel("Sales")
plt.title("Sales Over Time for 5 Random Stores")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "sales_over_time_5_stores.png")
plt.close()

print("Data visualizations complete. Figures saved to assets/figures/.")
