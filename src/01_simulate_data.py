"""
STEP 0 (stand-in only): generate a synthetic M5-shaped dataset.

You do NOT need this file once you download the real data. It exists so the
whole pipeline is runnable today, before you touch Kaggle.

Real data instead:  https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
    sales_train_evaluation.csv, calendar.csv, sell_prices.csv
Swap it in with:  python 01_simulate_data.py --m5  (then the loaders find it)
The synthetic data mimics M5 structure: stores x items x days, with
weekday seasonality, yearly seasonality, promotions, SNAP flags, price
changes, intermittent demand, short history, and stockout zeros.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

N_STORES = 10
N_ITEMS = 120
N_DAYS = 730                      # 2 years of history
HORIZON = 28                      # M5-style 28-day holdout

stores = [f"CA_{i+1:02d}" for i in range(4)] + [f"TX_{i+1:02d}" for i in range(3)] + [f"WI_{i+1:02d}" for i in range(3)]
depts = ["FOODS_1", "FOODS_2", "FOODS_3", "HOBBIES_1", "HOUSEHOLD_1", "HOUSEHOLD_2"]
items = []
for i in range(N_ITEMS):
    items.append({
        "item_id": f"ITEM_{i:04d}",
        "dept_id": depts[i % len(depts)],
        "base": float(rng.gamma(2.0, 2.2)),          # baseline demand level
        "trend": float(rng.normal(0, 0.0004)),       # slow drift
        "weekly_amp": float(rng.uniform(0.15, 0.55)),
        "yearly_amp": float(rng.uniform(0.05, 0.35)),
        "promo_lift": float(rng.uniform(0.4, 1.8)),
        "intermittent": bool(rng.random() < 0.30),   # slow movers -> many zeros
    })
items = pd.DataFrame(items)

dates = pd.date_range("2023-01-01", periods=N_DAYS, freq="D")
cal = pd.DataFrame({
    "date": dates,
    "d": [f"d_{i+1}" for i in range(N_DAYS)],
    "wm_yr_wk": dates.isocalendar().week.astype(int) + (dates.year - 2023) * 52,
    "weekday": dates.day_name(),
    "month": dates.month,
    "year": dates.year,
    "snap": ((dates.dayofweek == 6) & np.isin([d for d in dates], dates)).astype(int),
    "event_name_1": np.where((dates.month == 11) & (dates.day.isin([24, 25, 26])), "Thanksgiving", ""),
    "event_type_1": np.where((dates.month == 11) & (dates.day.isin([24, 25, 26])), "National", ""),
})
# SNAP: food-stamp days cluster early in the month
cal["snap"] = ((cal["month"].isin([1, 2, 3]) & (cal["date"].dt.day <= 10)) |
               (cal["date"].dt.day.isin([1, 15]))).astype(int)

rows = []
for s in stores:
    for it in items.itertuples():
        t = np.arange(N_DAYS)
        dow = dates.dayofweek.values
        weekly = 1 + it.weekly_amp * np.sin(2 * np.pi * (dow - 2) / 7)
        yearly = 1 + it.yearly_amp * np.sin(2 * np.pi * (t / 365.25) + rng.uniform(0, 6.28))
        trend = 1 + it.trend * t
        lam = it.base * weekly * yearly * trend
        if it.intermittent:
            lam = lam * 0.25
        promo = rng.random((N_DAYS,)) < 0.08
        lam = lam * (1 + it.promo_lift * promo)
        noise = rng.gamma(shape=4.0, scale=lam / 4.0)
        sales = rng.poisson(np.clip(noise, 0, None)).astype(int)
        # price: weekly, with promo markdowns
        price = np.round(1.5 + it.base * 0.8 + rng.normal(0, 0.15, N_DAYS), 2)
        price = np.where(promo, price * rng.uniform(0.6, 0.85, N_DAYS), price)
        lead = int(rng.integers(2, 15))
        rows.append(pd.DataFrame({
            "store_id": s, "item_id": it.item_id, "dept_id": it.dept_id,
            "date": dates, "d": cal["d"].values, "sales": sales,
            "sell_price": price, "promo": promo.astype(int),
            "snap": cal["snap"].values, "lead_time_days": lead,
            "unit_cost": np.round(price * rng.uniform(0.55, 0.7), 2),
            "unit_price": np.round(price * rng.uniform(1.3, 1.6), 2),
        }))
long = pd.concat(rows, ignore_index=True)
long.to_parquet(DATA / "sales_long.parquet", index=False)
cal.drop(columns=["weekday"]).to_parquet(DATA / "calendar.parquet", index=False)
print(f"wrote {len(long):,} rows -> data/sales_long.parquet")
print(f"series: {long.groupby(['store_id','item_id']).ngroups:,} | days: {N_DAYS} | horizon: {HORIZON}")
print(long.head(3).to_string())
