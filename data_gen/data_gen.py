# 12/20/2025 big change: 3 years and 7 ad spend categories with ad stock on broadcast TV and online TV

# Current version corrected the size of sales and made the product more price elastic
# this script 1) creates a dataclass to make setting and changing the details of the data gen easier and more reliable
# 2) generates store-level data and store-specific characteristics


from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
import pandas as pd


# -------------------------------
# CONFIGURATION
# -------------------------------

@dataclass
class DGPConfig:
    n_stores: int = 50
    n_weeks: int = 156  # 3 years
    seed: int = 42

    # Sales scale: tuned previously for ~$50k/week average
    intercept: float = 22.0

    # Price effects (keep from your tuned version)
    beta_price: float = -0.12
    beta_relprice: float = -1.31  # calibrated behavior you wanted

    # Non-marketing controls
    beta_competitors: float = -0.15
    beta_size: float = 0.3
    beta_income: float = 0.02
    beta_mgr_exp: float = 0.01
    beta_mgr_vacant: float = -0.05

    # Fixed-effect and noise scales
    sigma_store_fe: float = 0.4
    sigma_week_fe: float = 0.2
    sigma_eps: float = 0.3

    # -----------------------
    # Channel spend design
    # -----------------------
    channels: tuple = (
        "paid_search",
        "paid_social",
        "broadcast_tv",
        "stream_tv",
        "direct_mail",
        "print",
        "other",
    )

    # Year 1 quarterly means (per week)
    # Weeks 1-13 = Q1, 14-26 = Q2, 27-39 = Q3, 40-52 = Q4 (repeats each year)
    quarter_means_year1: dict = field(default_factory=lambda: {
        "paid_search":   [100, 150, 100, 200],
        "paid_social":   [100, 150, 100, 200],
        "broadcast_tv":  [50,  75,  50,  100],   # half of paid search
        "stream_tv":     [200, 300, 200, 400],   # double paid search
        "direct_mail":   [50,  75,  50,  100],   # same as broadcast tv
        "print":         [25,  37.5, 25,  50],   # half of broadcast tv
        "other":         [50,  125, 10,  140],
    })

    # Year scaling: Year2 is +20%, Year3 is -20% vs Year1
    year_scalars: tuple = (1.0, 1.2, 0.8)

    # Tight within-quarter variation (relative std dev)
    # e.g., 0.08 = ~8% coefficient of variation around quarter mean
    within_qtr_cv: float = 0.08

    # -----------------------
    # Diminishing returns + interactions
    # -----------------------

    # Diminishing returns: we’ll use a saturating function eff = log1p(spend / k)
    # Smaller k => saturates sooner.
    sat_k: dict = field(default_factory=lambda: {
        "paid_search": 200.0,
        "paid_social": 250.0,
        "broadcast_tv": 600.0,
        "stream_tv": 500.0,
        "direct_mail": 300.0,
        "print": 200.0,
        "other": 250.0,
    })

    # Base channel coefficients (effects on log_sales)
    beta_channel: dict = field(default_factory=lambda: {
        "paid_search": 0.25,
        "paid_social": 0.18,
        "broadcast_tv": 0.10,   # via adstock (separate)
        "stream_tv": 0.12,      # via adstock (separate)
        "direct_mail": 0.15,
        "print": 0.06,
        "other": 0.05,
    })

    # Interaction: TV/print/other increase effectiveness of (search/social/direct_mail)
    # multiplier = 1 + interaction_strength * (upper_funnel_index)
    interaction_strength: float = 0.35

    # How to build the "upper funnel index" from TV/print/other effectiveness
    upper_funnel_weights: dict = field(default_factory=lambda: {
        "broadcast_tv": 0.45,
        "stream_tv": 0.45,
        "print": 0.05,
        "other": 0.05,
    })

    # Adstock (geometric) decay rates for broadcast_tv and stream_tv
    adstock_decay_broadcast_tv: float = 0.75
    adstock_decay_stream_tv: float = 0.65


# -------------------------------
# STORE-LEVEL DATA
# -------------------------------

def generate_store_level_data(cfg: DGPConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    store_ids = np.arange(cfg.n_stores)

    # store_size is a normalized latent size factor (centered around 1)
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


# -------------------------------
# HELPERS: QUARTERS + ADSTOCK + SATURATION
# -------------------------------

def week_to_year_and_quarter(week_index_zero_based: int) -> tuple[int, int]:
    """
    week_index_zero_based: 0..(n_weeks-1)
    Returns: (year_number_1_based, quarter_number_1_based)
    Assumes 52-week years and 13-week quarters.
    """
    year_1_based = (week_index_zero_based // 52) + 1
    week_in_year = week_index_zero_based % 52
    quarter_1_based = (week_in_year // 13) + 1
    return year_1_based, quarter_1_based


def saturating_effect(spend: pd.Series, k: float) -> pd.Series:
    """
    Diminishing returns function.
    eff = log1p(spend / k)
    - spend=0 => 0
    - increasing but concave
    """
    return np.log1p(spend / k)


def geometric_adstock(spend: pd.Series, decay: float) -> pd.Series:
    """
    Standard geometric adstock:
    stock_t = spend_t + decay * stock_{t-1}
    """
    out = np.zeros(len(spend), dtype=float)
    carry = 0.0
    for i, x in enumerate(spend.to_numpy()):
        carry = x + decay * carry
        out[i] = carry
    return pd.Series(out, index=spend.index)


# -------------------------------
# PANEL-LEVEL DATA
# -------------------------------

def generate_panel_data(cfg: DGPConfig, df_store: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)

    weeks = np.arange(cfg.n_weeks)
    week_effects = rng.normal(loc=0.0, scale=cfg.sigma_week_fe, size=cfg.n_weeks)

    # Expand stores × weeks
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

    df = df.sort_values(["store_id", "week"]).reset_index(drop=True)

    rng2 = np.random.default_rng(cfg.seed + 2)

    # Price variation (same idea as before)
    base_price = rng2.normal(loc=10.0, scale=1.0, size=len(df))
    price_trend = 0.01 * df["week"].values
    df["price"] = base_price + 0.1 * (df["area_income"] - df["area_income"].mean()) / 10.0 + price_trend

    # Competitor price
    comp_price_noise = rng2.normal(loc=0.0, scale=0.5, size=len(df))
    df["competitor_price"] = df["price"] + comp_price_noise

    # Competitor count
    df["competitor_count"] = df["base_competitors"] + rng2.poisson(lam=0.2, size=len(df))
    df["competitor_count"] = df["competitor_count"].clip(lower=0)

    # Intuitive relative price: own / competitor
    df["relative_price"] = df["price"] / df["competitor_price"]

    # -----------------------
    # Multi-channel spend by quarter, year scaling, tight within-quarter variation
    # -----------------------
    # Build per-row quarter mean for each channel
    years = []
    quarters = []
    for w in df["week"].to_numpy():
        y, q = week_to_year_and_quarter(int(w))
        years.append(y)
        quarters.append(q)
    df["year"] = years
    df["quarter"] = quarters

    # Map year scalar
    year_scalar_map = {1: cfg.year_scalars[0], 2: cfg.year_scalars[1], 3: cfg.year_scalars[2]}
    df["year_scalar"] = df["year"].map(year_scalar_map).astype(float)

    # Generate spend columns
    for ch in cfg.channels:
        q_means = cfg.quarter_means_year1[ch]  # list of 4
        # quarter mean for each row (year1 baseline) * year scalar
        base_mean = df["quarter"].map({1: q_means[0], 2: q_means[1], 3: q_means[2], 4: q_means[3]}).astype(float)
        mean_scaled = base_mean * df["year_scalar"]

        # tight within-quarter variation: multiplicative noise around mean
        # lognormal-ish but approximately normal for small cv:
        # spend = mean * (1 + noise), clipped at 0
        noise = rng2.normal(loc=0.0, scale=cfg.within_qtr_cv, size=len(df))
        spend = mean_scaled * (1.0 + noise)

        # Mild store-level tilt: bigger stores spend a little more across all channels
        spend = spend * (0.9 + 0.2 * df["store_size"])

        df[f"spend_{ch}"] = np.clip(spend, 0.0, None)

    # -----------------------
    # Adstock for broadcast_tv and stream_tv (per store separately)
    # -----------------------
    df["adstock_broadcast_tv"] = (
        df.groupby("store_id")["spend_broadcast_tv"]
        .apply(lambda s: geometric_adstock(s, cfg.adstock_decay_broadcast_tv))
        .reset_index(level=0, drop=True)
    )

    df["adstock_stream_tv"] = (
        df.groupby("store_id")["spend_stream_tv"]
        .apply(lambda s: geometric_adstock(s, cfg.adstock_decay_stream_tv))
        .reset_index(level=0, drop=True)
    )

    # -----------------------
    # Diminishing returns (nonlinear within-week) for each channel
    # -----------------------
    for ch in cfg.channels:
        df[f"eff_{ch}"] = saturating_effect(df[f"spend_{ch}"], cfg.sat_k[ch])

    # For TV channels, we will also use adstock versions in the model:
    df["eff_adstock_broadcast_tv"] = saturating_effect(df["adstock_broadcast_tv"], cfg.sat_k["broadcast_tv"])
    df["eff_adstock_stream_tv"] = saturating_effect(df["adstock_stream_tv"], cfg.sat_k["stream_tv"])

    # -----------------------
    # Cross-channel interactions:
    # TV/print/other make search/social/direct_mail more effective
    # -----------------------
    upper_funnel_index = (
        cfg.upper_funnel_weights["broadcast_tv"] * df["eff_adstock_broadcast_tv"]
        + cfg.upper_funnel_weights["stream_tv"] * df["eff_adstock_stream_tv"]
        + cfg.upper_funnel_weights["print"] * df["eff_print"]
        + cfg.upper_funnel_weights["other"] * df["eff_other"]
    )

    # Multiplier applied to lower-funnel channels
    # Example: multiplier = 1 + 0.35 * upper_funnel_index
    df["lower_funnel_multiplier"] = 1.0 + cfg.interaction_strength * upper_funnel_index

    # Apply to lower-funnel effective spend
    df["eff_paid_search_adj"] = df["eff_paid_search"] * df["lower_funnel_multiplier"]
    df["eff_paid_social_adj"] = df["eff_paid_social"] * df["lower_funnel_multiplier"]
    df["eff_direct_mail_adj"] = df["eff_direct_mail"] * df["lower_funnel_multiplier"]

    # -----------------------
    # Noise
    # -----------------------
    eps = rng2.normal(loc=0.0, scale=cfg.sigma_eps, size=len(df))

    # -----------------------
    # Demand model: replace single ad_spend with channel-specific effects + adstock
    # -----------------------
    log_sales = (
        cfg.intercept
        + cfg.beta_price * df["price"]
        + cfg.beta_relprice * df["relative_price"]
        + cfg.beta_competitors * df["competitor_count"]
        + cfg.beta_size * df["store_size"]
        + cfg.beta_income * df["area_income"]
        + cfg.beta_mgr_exp * df["manager_experience"]
        + cfg.beta_mgr_vacant * df["manager_vacant"]
        # Lower-funnel channels with interaction boost
        + cfg.beta_channel["paid_search"] * df["eff_paid_search_adj"]
        + cfg.beta_channel["paid_social"] * df["eff_paid_social_adj"]
        + cfg.beta_channel["direct_mail"] * df["eff_direct_mail_adj"]
        # Upper-funnel channels (TV via adstock, print/other direct)
        + cfg.beta_channel["broadcast_tv"] * df["eff_adstock_broadcast_tv"]
        + cfg.beta_channel["stream_tv"] * df["eff_adstock_stream_tv"]
        + cfg.beta_channel["print"] * df["eff_print"]
        + cfg.beta_channel["other"] * df["eff_other"]
        # Fixed effects + noise
        + df["store_fe"]
        + df["week_fe"]
        + eps
    )

    df["log_sales"] = log_sales
    df["sales"] = np.exp(log_sales)

    return df


# -------------------------------
# OUTPUT WRITER
# -------------------------------

def save_outputs(df: pd.DataFrame, cfg: DGPConfig, base_dir: Path) -> None:
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "final_simulated_panel.csv"
    df.to_csv(csv_path, index=False)

    true_params = {
        "intercept": cfg.intercept,
        "price": cfg.beta_price,
        "relative_price": cfg.beta_relprice,
        "competitor_count": cfg.beta_competitors,
        "store_size": cfg.beta_size,
        "area_income": cfg.beta_income,
        "manager_experience": cfg.beta_mgr_exp,
        "manager_vacant": cfg.beta_mgr_vacant,
        "beta_channel": cfg.beta_channel,
        "interaction_strength": cfg.interaction_strength,
        "adstock_decay_broadcast_tv": cfg.adstock_decay_broadcast_tv,
        "adstock_decay_stream_tv": cfg.adstock_decay_stream_tv,
        "sat_k": cfg.sat_k,
        "quarter_means_year1": cfg.quarter_means_year1,
        "year_scalars": cfg.year_scalars,
        "within_qtr_cv": cfg.within_qtr_cv,
    }

    json_path = data_dir / "true_params.json"
    with open(json_path, "w") as f:
        json.dump(true_params, f, indent=2)


# -------------------------------
# MAIN
# -------------------------------

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


