# Data Generation Process

This project uses a synthetic panel dataset that simulates weekly sales for multiple retail stores. The goal is to define a known data generating process (DGP) so that different econometric models can be evaluated on how well they recover the true parameters.

The DGP includes:

- Multiple stores observed over multiple weeks
- Store-level characteristics
- Time-varying marketing decisions
- Competitor conditions
- Management characteristics
- Random noise

The output of the DGP is a panel dataset saved as:

- data/final_simulated_panel.csv

and a JSON file with the true parameters:

- data/true_params.json

These are used to evaluate model accuracy.

## Variables

The main variables are:

Store and time identifiers:
- store_id: store identifier
- week: time period (e.g., week index)

Store-level characteristics:
- store_size: size of the store (e.g., square footage, scaled)
- area_income: average income in the area
- manager_experience: years of experience of the manager
- manager_vacant: indicator for vacancy (1 means no manager)

Marketing and pricing:
- price: own price for the focal product or basket
- ad_spend: current period advertising spend
- ad_spend_lag: lagged advertising spend for the same store

Competition:
- competitor_price: price level of competitor
- competitor_count: number of competing stores in the area
- relative_price: competitor_price divided by own price

Outcome:
- sales: simulated sales level
- log_sales: natural log of sales

## Structural Equation

Sales are generated from a log-linear model:

log_sales_it = intercept
    + b_price * price_it
    + b_relprice * relative_price_it
    + b_ad * ad_spend_it
    + b_adlag * ad_spend_lag_it
    + b_comp * competitor_count_it
    + b_size * store_size_i
    + b_income * area_income_i
    + b_mgr * manager_experience_i
    + b_vac * manager_vacant_i
    + store_effect_i
    + week_effect_t
    + epsilon_it

where epsilon_it is random noise.

The true parameters are stored in data/true_params.json so that model estimates can be compared to the true values.

## Python Code

The script below generates the panel data and writes both the CSV and the JSON file. Save this as data_gen/data_gen.py.

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import json

@dataclass
class DGPConfig:
    n_stores: int = 50
    n_weeks: int = 52
    seed: int = 42

    intercept: float = 4.5

    beta_price: float = -1.2
    beta_relprice: float = -0.8
    beta_ad: float = 0.05
    beta_adlag: float = 0.02
    beta_competitors: float = -0.15
    beta_size: float = 0.3
    beta_income: float = 0.02
    beta_mgr_exp: float = 0.01
    beta_mgr_vacant: float = -0.05

    sigma_store_fe: float = 0.4
    sigma_week_fe: float = 0.2
    sigma_eps: float = 0.3

def generate_store_level_data(cfg: DGPConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    store_ids = np.arange(cfg.n_stores)

    store_size = rng.normal(loc=1.0, scale=0.2, size=cfg.n_stores)
    store_size = np.clip(store_size, 0.5, None)

    area_income = rng.normal(loc=60.0, scale=10.0, size=cfg.n_stores)

    manager_experience = rng.normal(loc=5.0, scale=2.0, size=cfg.n_stores)
    manager_experience = np.clip(manager_experience, 0.0, None)

    manager_vacant = rng.binomial(n=1, p=0.1, size=cfg.n_stores)

    base_competitors = rng.poisson(lam=3.0, size=cfg.n_stores)

    store_effects = rng.normal(loc=0.0, scale=cfg.sigma_store_fe, size=cfg.n_stores)

    df_store = pd.DataFrame({
        "store_id": store_ids,
        "store_size": store_size,
        "area_income": area_income,
        "manager_experience": manager_experience,
        "manager_vacant": manager_vacant,
        "base_competitors": base_competitors,
        "store_fe": store_effects
    })

    return df_store

def generate_panel_data(cfg: DGPConfig, df_store: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)

    weeks = np.arange(cfg.n_weeks)
    week_effects = rng.normal(loc=0.0, scale=cfg.sigma_week_fe, size=cfg.n_weeks)

    df = (
        df_store
        .assign(key=1)
        .merge(
            pd.DataFrame({"week": weeks, "key": 1, "week_fe": week_effects}),
            on="key",
            how="outer"
        )
        .drop(columns=["key"])
    )

    rng = np.random.default_rng(cfg.seed + 2)

    base_price = rng.normal(loc=10.0, scale=1.0, size=len(df))
    price_trend = 0.01 * df["week"].values
    df["price"] = base_price + 0.1 * (df["area_income"] - df["area_income"].mean()) / 10.0 + price_trend

    comp_price_noise = rng.normal(loc=0.0, scale=0.5, size=len(df))
    df["competitor_price"] = df["price"] + comp_price_noise

    df["competitor_count"] = df["base_competitors"] + rng.poisson(lam=0.2, size=len(df))
    df["competitor_count"] = df["competitor_count"].clip(lower=0)

    df["ad_spend"] = (
        5.0
        + 0.5 * df["store_size"]
        + 0.05 * df["area_income"]
        + rng.normal(loc=0.0, scale=2.0, size=len(df))
    )
    df["ad_spend"] = df["ad_spend"].clip(lower=0.0)

    df = df.sort_values(["store_id", "week"]).reset_index(drop=True)
    df["ad_spend_lag"] = df.groupby("store_id")["ad_spend"].shift(1)
    df["ad_spend_lag"] = df["ad_spend_lag"].fillna(0.0)

    df["relative_price"] = df["competitor_price"] / df["price"]

    eps = rng.normal(loc=0.0, scale=cfg.sigma_eps, size=len(df))

    log_sales = (
        cfg.intercept
        + cfg.beta_price * df["price"]
        + cfg.beta_relprice * df["relative_price"]
        + cfg.beta_ad * df["ad_spend"]
        + cfg.beta_adlag * df["ad_spend_lag"]
        + cfg.beta_competitors * df["competitor_count"]
        + cfg.beta_size * df["store_size"]
        + cfg.beta_income * df["area_income"]
        + cfg.beta_mgr_exp * df["manager_experience"]
        + cfg.beta_mgr_vacant * df["manager_vacant"]
        + df["store_fe"]
        + df["week_fe"]
        + eps
    )

    df["log_sales"] = log_sales
    df["sales"] = np.exp(log_sales)

    return df

def save_outputs(df: pd.DataFrame, cfg: DGPConfig, base_dir: Path) -> None:
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "final_simulated_panel.csv"
    df.to_csv(csv_path, index=False)

    true_params = {
        "intercept": cfg.intercept,
        "price": cfg.beta_price,
        "relative_price": cfg.beta_relprice,
        "ad_spend": cfg.beta_ad,
        "ad_spend_lag": cfg.beta_adlag,
        "competitor_count": cfg.beta_competitors,
        "store_size": cfg.beta_size,
        "area_income": cfg.beta_income,
        "manager_experience": cfg.beta_mgr_exp,
        "manager_vacant": cfg.beta_mgr_vacant
    }

    json_path = data_dir / "true_params.json"
    with open(json_path, "w") as f:
        json.dump(true_params, f, indent=2)

def main():
    base_dir = Path(__file__).resolve().parents[1]
    cfg = DGPConfig()

    df_store = generate_store_level_data(cfg)
    df_panel = generate_panel_data(cfg, df_store)
    save_outputs(df_panel, cfg, base_dir)

    print("Data generation complete.")
    print("Rows:", len(df_panel))
    print("Output written to data/final_simulated_panel.csv and data/true_params.json")

if __name__ == "__main__":
    main()
