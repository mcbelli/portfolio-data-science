# Data Visualization Script for Simulated Store Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Load data
# -----------------------------
base_dir = Path(__file__).resolve().parents[1]
print("base_dir is:")
print(base_dir)

df = pd.read_csv(base_dir / "data" / "final_simulated_panel.csv")

print("df columns:")
for col in df.columns:
    print(col)
print("End of df.columns\n")

# Work from a single analysis dataframe
df_panel = df.copy()

# -----------------------------
# Derive time variables
# -----------------------------
df_panel["year"] = df_panel["week"] // 52 + 1
df_panel["week_of_year"] = df_panel["week"] % 52 + 1

# Define month as 4-week blocks (13 months/year)
df_panel["month"] = (df_panel["week_of_year"] - 1) // 4 + 1

# -----------------------------
# Ad spend aggregation
# -----------------------------
ad_cols = [
    "spend_paid_search",
    "spend_paid_social",
    "spend_broadcast_tv",
    "spend_stream_tv",
    "spend_direct_mail",
    "spend_print",
    "spend_other"
]

df_panel["total_ad_spend"] = df_panel[ad_cols].sum(axis=1)

# -----------------------------
# SANITY CHECK 1: Weekly sales by year
# -----------------------------
weekly_sales = (
    df_panel
    .groupby(["year", "week_of_year"], observed=True)["sales"]
    .sum()
    .reset_index()
)

fig_dir = base_dir / "assets" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 6))

for yr in sorted(weekly_sales["year"].unique()):
    sub = weekly_sales[weekly_sales["year"] == yr]
    plt.plot(sub["week_of_year"], sub["sales"], label=f"Year {yr}")

plt.xlabel("Week of Year")
plt.ylabel("Total Sales")
plt.title("Weekly Total Sales by Year")
plt.ylim(bottom=0)
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "weekly_sales_by_year.png")
plt.close()

# -----------------------------
# SANITY CHECK 2: Monthly sales and spend
# -----------------------------
monthly = (
    df_panel
    .groupby(["year", "month"], observed=True)
    .agg(
        total_sales=("sales", "sum"),
        total_ad_spend=("total_ad_spend", "sum")
    )
    .reset_index()
)

# Monthly sales
plt.figure(figsize=(10, 6))

for y in sorted(monthly["year"].unique()):
    sub = monthly[monthly["year"] == y]
    plt.plot(sub["month"], sub["total_sales"], marker="o", label=f"Year {y}")

plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Monthly Total Sales by Year")
plt.ylim(bottom=0)
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "monthly_total_sales_by_year.png")
plt.close()

# Monthly ad spend
plt.figure(figsize=(10, 6))

for y in sorted(monthly["year"].unique()):
    sub = monthly[monthly["year"] == y]
    plt.plot(sub["month"], sub["total_ad_spend"], marker="o", label=f"Year {y}")

plt.xlabel("Month")
plt.ylabel("Total Ad Spend")
plt.title("Monthly Total Ad Spend by Year")
plt.ylim(bottom=0)
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "monthly_total_ad_spend_by_year.png")
plt.close()


# weekly sales and spend
# -----------------------------
# SANITY CHECK 3: Weekly ad spend by year
# -----------------------------
weekly_spend = (
    df_panel
    .groupby(["year", "week_of_year"], observed=True)["total_ad_spend"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(10, 6))

for year, sub in weekly_spend.groupby("year"):
    plt.plot(
        sub["week_of_year"],
        sub["total_ad_spend"],
        label=f"Year {year}"
    )

plt.xlabel("Week of Year")
plt.ylabel("Total Advertising Spend")
plt.title("Total Weekly Advertising Spend by Year")
plt.ylim(bottom=0)
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "total_weekly_ad_spend_by_year.png")
plt.close()

# more checks

# 1. Sales vs price scatterplot
plt.figure(figsize=(6, 5))
plt.scatter(df_panel["price"], df_panel["sales"], alpha=0.3)
plt.xlabel("Price")
plt.ylabel("Sales")
plt.title("Sales vs Price")
plt.savefig(fig_dir / "sales_vs_price.png")
plt.close()

#for col in df_panel:
#    print(col)

# 2. Sales vs advertising scatterplot
plt.figure(figsize=(6, 5))
plt.scatter(df_panel["total_ad_spend"], df_panel["sales"], alpha=0.3)
plt.xlabel("Advertising Spend")
plt.ylabel("Sales")
plt.title("Sales vs Advertising Spend")
plt.savefig(fig_dir / "sales_vs_ad_spend.png")
plt.close()



# 3. Sales vs store size scatterplot
plt.figure(figsize=(6, 5))
plt.scatter(df_panel["store_size"], df_panel["sales"], alpha=0.3)
plt.xlabel("Store Size")
plt.ylabel("Sales")
plt.title("Sales vs Store Size")
plt.savefig(fig_dir / "sales_vs_store_size.png")
plt.close()


# 4. Sales vs manager experience with vacancy as an experience bin
df_panel["experience_bin"] = (
    pd.cut(
        df["manager_experience"],
        bins=[0, 2, 5, 10, 20, 50],
        labels=["0-2", "3-5", "6-10", "11-20", "21+"],
        include_lowest=True
    )
    .cat.add_categories("Vacant")
)

df_panel["experience_bin"] = df_panel["experience_bin"].where(
    df_panel["manager_vacant"] == 0, "Vacant"
)


#sales_by_exp = df.groupby("experience_bin")["sales"].mean().reindex(
#    ["Vacant", "0-2", "3-5", "6-10", "11-20", "21+"]
#)
sales_by_exp = (
    df_panel.groupby("experience_bin", observed=False)["sales"]
      .mean()
      .reindex(["Vacant", "0-2", "3-5", "6-10", "11-20", "21+"])
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
plt.scatter(df_panel["relative_price"], df_panel["sales"], alpha=0.3)
plt.xlabel("Relative Price (Competitor Price / Own Price)")
plt.ylabel("Sales")
plt.title("Sales vs Relative Price")
plt.savefig(fig_dir / "sales_vs_relative_price.png")
plt.close()

# 6. Sales by binned relative price

# 6. Sales by binned relative price
df_panel["relative_price_bin"] = pd.cut( df_panel["relative_price"],
    bins=[0.0, 0.8, 0.95, 1.05, 1.2, 5.0],
    labels=["<0.8", "0.80-0.95", "0.95-1.05", "1.05-1.20", "1.20+"],
     include_lowest=True )

#sales_by_rel_price = df_panel.groupby("relative_price_bin")["sales"].mean()
sales_by_rel_price = df_panel.groupby(
    "relative_price_bin", observed=False
)["sales"].mean()


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
unique_stores = df_panel["store_id"].unique()
np.random.seed(42)
sample_stores = np.random.choice(unique_stores, size=5, replace=False)

plt.figure(figsize=(10, 6))
for s in sample_stores:
    sub = df_panel[df_panel["store_id"] == s].sort_values("week")
    plt.plot(sub["week"], sub["sales"], label=f"Store {s}")

plt.xlabel("Week")
plt.ylabel("Sales")
plt.title("Sales Over Time for 5 Random Stores")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "sales_over_time_5_stores.png")
plt.close()

print("Data visualizations complete. Figures saved to assets/figures/.")
