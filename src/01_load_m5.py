"""
STEP 0 (REAL DATA): M5 -> the long table the rest of the pipeline expects.

Memory-frugal by design: the wide file is read in row chunks, filtered to the
chosen scope *before* melting, and written chunk-by-chunk into partitioned
parquet. Full M5 is never held in RAM.

Input  (data/raw/): sales_train_evaluation.csv, calendar.csv, sell_prices.csv
Output (data/):     sales_long.parquet (partitioned by store_id), calendar.parquet

Run: python 01_load_m5.py                          # 1 store, 2 depts (default, laptop-safe)
     python 01_load_m5.py --stores CA_1 --depts FOODS_3
     python 01_load_m5.py --stores CA_1 TX_1 WI_1 --depts FOODS_1 FOODS_2 FOODS_3
     python 01_load_m5.py --all                    # needs ~16GB RAM

ASSUMPTIONS (M5 has no inventory/lead-time/cost data - document these in your report)
"""
import argparse
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data"

ASSUMPTIONS = {
    "lead_time_days": "Deterministic per store: CA=5, TX=7, WI=4 days (illustrative placeholders). "
                      "Replace with supplier lead times or predict them (DataCo dataset).",
    "unit_cost": "sell_price x 0.65  (i.e. 35% gross margin).",
    "unit_price": "sell_price (no extra markup assumed).",
    "promo": "M5 has no promo flag. Proxy = sell_price at least 5% below that "
             "item-store's trailing 28-day average price.",
    "snap": "snap_CA/TX/WI collapsed to one flag (task says single store per state).",
}
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
LEAD_MAP = {"CA": 5, "TX": 7, "WI": 4}
CHUNK = 1500


def load_calendar():
    cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    cal["snap"] = cal[["snap_CA", "snap_TX", "snap_WI"]].max(axis=1).astype("int8")
    for c in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        cal[c] = cal[c].fillna("")
    cal = cal[["date", "d", "wm_yr_wk", "weekday", "wday", "month", "year",
               "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap"]]
    cal["wm_yr_wk"] = cal["wm_yr_wk"].astype("int16")
    cal["wday"] = cal["wday"].astype("int8")
    cal["month"] = cal["month"].astype("int8")
    cal["year"] = cal["year"].astype("int16")
    return cal


def day_columns():
    with open(RAW / "sales_train_evaluation.csv") as f:
        header = f.readline().strip().split(",")
    return [c for c in header if c.startswith("d_")]


def load_prices(stores, items):
    p = pd.read_csv(RAW / "sell_prices.csv",
                    dtype={"store_id": "category", "item_id": "category",
                           "wm_yr_wk": "int16", "sell_price": "float32"})
    p = p[p["store_id"].isin(stores) & p["item_id"].isin(items)]
    print(f"  prices in scope: {len(p):,} rows")
    return p.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+", default=["CA_1"])
    ap.add_argument("--depts", nargs="+", default=["FOODS_1", "FOODS_3"])
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    stores = (["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3",
               "WI_1", "WI_2", "WI_3"] if a.all else a.stores)
    depts = None if a.all else a.depts

    cal = load_calendar()
    cal.to_parquet(DATA / "calendar.parquet", index=False)
    print(f"calendar: {len(cal)} days, {cal['date'].min().date()} -> {cal['date'].max().date()}")

    # ---- pass 1: which series are in scope? (ids only, cheap) ----
    ids = pd.read_csv(RAW / "sales_train_evaluation.csv", usecols=ID_COLS)
    mask = ids["store_id"].isin(stores)
    if depts:
        mask &= ids["dept_id"].isin(depts)
    ids = ids[mask].reset_index(drop=True)
    items = set(ids["item_id"])
    print(f"scope: stores={stores} depts={depts} -> {len(ids):,} series")

    prices = load_prices(stores, items)
    dcols = day_columns()
    print(f"  day columns: {len(dcols)}")

    cal_small = cal[["d", "date", "snap", "wm_yr_wk"]]
    outdir = DATA / "sales_long"
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written = 0
    reader = pd.read_csv(RAW / "sales_train_evaluation.csv",
                         usecols=ID_COLS + dcols, chunksize=CHUNK)
    part = 0
    for chunk in reader:
        m = chunk["store_id"].isin(stores)
        if depts:
            m &= chunk["dept_id"].isin(depts)
        if not m.any():
            continue
        c = chunk[m].copy()
        for d in dcols:
            c[d] = c[d].astype("int16")
        long = c.melt(id_vars=ID_COLS, value_vars=dcols, var_name="d", value_name="sales")
        del c
        long = long.merge(cal_small, on="d", how="left")
        long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        for col in ["store_id", "item_id", "dept_id", "cat_id", "state_id"]:
            long[col] = long[col].astype("category")
        long["date"] = pd.to_datetime(long["date"])

        # price carry-forward within the chunk's series
        long = long.sort_values(["store_id", "item_id", "d"], kind="stable")
        long["price_imputed"] = long["sell_price"].isna().astype("int8")
        g = long.groupby(["store_id", "item_id"], observed=True)["sell_price"]
        long["sell_price"] = g.ffill().bfill()

        long["lead_time_days"] = long["state_id"].astype(str).map(LEAD_MAP).astype("int8")
        long["unit_cost"] = (long["sell_price"] * 0.65).astype("float32")
        long["unit_price"] = long["sell_price"]
        base = long.groupby(["store_id", "item_id"], observed=True)["sell_price"] \
                   .transform(lambda s: s.rolling(28, min_periods=7).mean().shift(1))
        long["promo"] = (long["sell_price"] <= base * 0.95).fillna(False).astype("int8")
        long["sales"] = long["sales"].astype("int16")

        keep = ["store_id", "item_id", "dept_id", "cat_id", "state_id", "date", "d",
                "sales", "sell_price", "promo", "snap", "lead_time_days",
                "unit_cost", "unit_price", "price_imputed"]
        long = long[keep]
        long.to_parquet(outdir / f"part-{part:03d}.parquet", index=False)
        written += len(long); part += 1
        print(f"  chunk {part}: +{len(long):,} rows (total {written:,})")
        del long

    print(f"\nwrote {written:,} rows -> data/sales_long/ ({part} part files)")
    print("ASSUMPTIONS (state these in your report):")
    for k, v in ASSUMPTIONS.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
