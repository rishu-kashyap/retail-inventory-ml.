"""
STEP 1: leakage-safe feature engineering for a 28-day-ahead forecast.

The rule: every feature must be knowable at the forecast origin (28 days before
the day predicted). All lags/rollings are shifted by >= 28. Calendar, planned
promos and price are allowed to be "future" - they are known in advance in real
retail.

Memory-lean: works on ~1-2M rows in float32 and frees intermediates as it goes,
so it runs on a 2GB machine. Use --days to shorten history for very large scopes.

Run: python 02_features.py                 # full history
     python 02_features.py --days 1200     # last 1200 days only (lighter)
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HORIZON = 28

FEATURES = [
    "lag_28", "lag_35", "lag_42", "lag_56", "lag_364",
    "rmean_7", "rmean_28", "rmean_56", "rmean_91",
    "rstd_28", "rstd_56", "rmedian_28", "zero_share_28", "mom_28",
    "price_lag28", "price_rel", "price_target",
    "dow", "month", "day", "weekofyear", "is_weekend", "in_promo", "is_event",
    "dept_code", "store_code", "item_code", "item_mean", "item_store_mean",
]


def load():
    p = DATA / "sales_long"
    if p.is_dir():
        df = pd.concat([pd.read_parquet(f) for f in sorted(p.glob("part-*.parquet"))],
                       ignore_index=True)
    else:
        df = pd.read_parquet(DATA / "sales_long.parquet")

    for c in ["sales", "promo", "snap", "lead_time_days"]:
        if c in df.columns:
            df[c] = df[c].astype("int16")
    for c in ["sell_price", "unit_cost", "unit_price"]:
        if c in df.columns:
            df[c] = df[c].astype("float32")
    for c in ["store_id", "item_id", "dept_id"]:
        df[c] = df[c].astype("category")
    df = df.sort_values(["store_id", "item_id", "d"], kind="stable").reset_index(drop=True)
    df["promo_plan"] = df["promo"].astype("int16")
    return df


def build(df, max_days=None):
    if max_days:
        cutoff = df["d"].str[2:].astype("int32").max() - max_days
        keep = df["d"].str[2:].astype("int32") > cutoff
        df = df[keep].reset_index(drop=True)
        print(f"  trimmed to last {max_days} days -> {len(df):,} rows")

    df["sid"] = pd.factorize(df["store_id"].astype(str) + "_" + df["item_id"].astype(str))[0]
    df = df.sort_values(["sid", "d"], kind="stable").reset_index(drop=True)

    sg = df.groupby("sid", sort=False, observed=True)["sales"]
    s = sg.ngroup()  # noqa - keeps groupby warm

    def shifted(L):
        return sg.shift(L).astype("float32")

    def rolled(w, how="mean", mp=None):
        mp = mp or max(3, w // 4)
        return sg.transform(
            lambda x: x.shift(HORIZON).rolling(w, min_periods=mp).agg(how)
        ).astype("float32")

    for L in (28, 35, 42, 56, 364):
        df[f"lag_{L}"] = shifted(L)

    for w in (7, 28, 56, 91):
        df[f"rmean_{w}"] = rolled(w, "mean")
    df["rstd_28"] = rolled(28, "std", mp=7)
    df["rstd_56"] = rolled(56, "std", mp=14)
    df["rmedian_28"] = rolled(28, "median", mp=7)

    # intermittency: share of zero-sales days - computed as a rolling mean of the
    # zero indicator (cheap) rather than a lambda over every window (slow)
    zero = (df["sales"] == 0).astype("float32")
    df["zero_share_28"] = (zero.groupby(df["sid"], sort=False)
                           .transform(lambda x: x.shift(HORIZON).rolling(28, min_periods=7).mean())
                           .astype("float32"))
    del zero

    df["mom_28"] = (df["rmean_28"] / (df["rmean_56"] + 1e-6)).astype("float32")

    # price: level at origin, and cheap/expensive relative to own recent history
    pg = df.groupby("sid", sort=False, observed=True)["sell_price"]
    df["price_lag28"] = pg.shift(HORIZON).astype("float32")
    hist = pg.transform(lambda x: x.shift(HORIZON).rolling(28, min_periods=7).mean()).astype("float32")
    df["price_rel"] = (df["price_lag28"] / (hist + 1e-6)).astype("float32")
    del hist
    df["price_target"] = df["sell_price"].astype("float32")

    d = df["date"].dt
    df["dow"] = d.dayofweek.astype("int8")
    df["month"] = d.month.astype("int8")
    df["day"] = d.day.astype("int8")
    df["weekofyear"] = d.isocalendar().week.astype("int16")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["in_promo"] = df["promo_plan"].astype("int8")
    df["is_event"] = (df["snap"] == 1).astype("int8")

    df["dept_code"] = df["dept_id"].astype("category").cat.codes.astype("int16")
    df["store_code"] = df["store_id"].astype("category").cat.codes.astype("int16")
    df["item_code"] = df["item_id"].astype("category").cat.codes.astype("int16")

    # series-level means: computed on train rows only, inside the splitter below
    return df


def add_series_means(df, train_mask):
    """item_mean / item_store_mean must be computed from TRAIN rows only (no leakage)."""
    tr = df.loc[train_mask]
    im = tr.groupby("item_code", observed=True)["sales"].mean().astype("float32")
    ism = tr.groupby("sid", observed=True)["sales"].mean().astype("float32")
    df["item_mean"] = df["item_code"].map(im).astype("float32")
    df["item_store_mean"] = df["sid"].map(ism).astype("float32")
    # global fallback for any unseen code
    df["item_mean"] = df["item_mean"].fillna(float(tr["sales"].mean()))
    df["item_store_mean"] = df["item_store_mean"].fillna(float(tr["sales"].mean()))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None, help="keep only the last N days")
    a = ap.parse_args()

    df = load()
    n_series = df["sid"].nunique() if "sid" in df else df.groupby(
        ["store_id", "item_id"], observed=True).ngroups
    print(f"loaded {len(df):,} rows, {n_series:,} series")

    df = build(df, a.days)
    df = df.dropna(subset=["lag_28", "lag_364", "rmean_28"])

    last = df["date"].max()
    split = last - pd.Timedelta(days=HORIZON - 1)
    train_mask = df["date"] < split
    df = add_series_means(df, train_mask)

    keep = (["store_id", "item_id", "sid", "date", "d", "sales"] + FEATURES
            + ["snap", "unit_cost", "unit_price", "lead_time_days", "price_imputed"])
    df = df[[c for c in keep if c in df.columns]]

    tr = df[df["date"] < split].reset_index(drop=True)
    va = df[df["date"] >= split].reset_index(drop=True)
    tr.to_parquet(DATA / "train.parquet", index=False)
    va.to_parquet(DATA / "valid.parquet", index=False)

    print(f"features: {len(FEATURES)}")
    print(f"train {tr.shape}  (< {split.date()})")
    print(f"valid {va.shape}  (>= {split.date()}, the 28-day holdout)")
    print(f"train date range: {tr['date'].min().date()} -> {tr['date'].max().date()}")
    print(f"zero-sale share in valid: {(va['sales'] == 0).mean():.1%}")
    print("\nleakage sanity - corr(feature, target) on the holdout:")
    c = va[FEATURES + ["sales"]].corrwith(va["sales"]).drop("sales", errors="ignore")
    print(c.sort_values(key=abs, ascending=False).head(8).round(3).to_string())


if __name__ == "__main__":
    main()
