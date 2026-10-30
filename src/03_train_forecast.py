"""
STEP 2: forecast + evaluate with a retail-appropriate metric.

Why not plain RMSE: in retail, one series can be a 500-unit/day bestseller and
another a 0.1-unit/day slow mover. Plain RMSE lets the big series dominate and
hides how badly you do on the long tail - which is exactly where stockouts and
dead stock live. M5's WRMSSE weights every series by its dollar importance and
scales the error by that series' own historical naive error.

Also: a naive "same day last week" baseline is included on purpose. If your
model cannot beat it, you have no project yet.

Run: python 03_train_forecast.py            (synthetic data)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
HORIZON = 28
FEATURES = json.loads((ROOT / "src" / "features.json").read_text()) if (ROOT / "src" / "features.json").exists() else None


def smape(y, yhat):
    d = (np.abs(y) + np.abs(yhat)) / 2
    return float(np.mean(np.where(d == 0, 0, np.abs(y - yhat) / np.where(d == 0, 1, d))) * 100)


def wrmsse(df, pred_col, train_full):
    """Series-scaled, dollar-weighted RMSSE on the 28-day holdout (M5-style)."""
    m = df.copy()
    m["err2"] = (m["sales"] - m[pred_col]) ** 2
    key = "sid" if "sid" in m.columns else None
    grp = [key] if key else ["store_id", "item_id"]
    mse = m.groupby(grp, observed=True)["err2"].mean().rename("mse")
    tf = train_full.sort_values(grp + ["d"]) if "d" in train_full.columns else train_full
    diff = tf.groupby(grp, observed=True)["sales"].diff()
    naive = (diff ** 2).groupby(tf[grp[0]], observed=True).mean().rename("naive_mse").replace(0, np.nan)
    rev = tf.groupby(grp, observed=True)["sales"].sum().rename("revenue_units")
    r = pd.concat([mse, naive, rev], axis=1)
    scale = np.sqrt(r["mse"] / r["naive_mse"])
    r["contrib"] = (r["revenue_units"] / r["revenue_units"].sum()) * scale
    return float(r["contrib"].sum())


def main():
    import lightgbm as lgb
    from sklearn.ensemble import HistGradientBoostingRegressor

    tr = pd.read_parquet(DATA / "train.parquet")
    va = pd.read_parquet(DATA / "valid.parquet")
    full = pd.concat([tr, va], ignore_index=True)

    drop = {"store_id", "item_id", "date", "sales", "d", "sid", "price_imputed"}
    feats = [c for c in tr.columns if c not in drop]
    print(f"using {len(feats)} features")

    # --- baseline 1: seasonal naive (same weekday, 28 days earlier) ---
    va["naive_28"] = va["lag_28"]
    # --- baseline 2: mean of recent 28 days ---
    va["mean_28"] = va["rmean_28"]

    Xtr, ytr = tr[feats], tr["sales"].astype("float64")
    Xva, yva = va[feats], va["sales"].astype("float64")

    # --- model 1: log1p target + log-normal bias correction -----------------
    # Naive expm1() of a log-scale prediction is biased low: E[y] = exp(mu + sigma^2/2).
    # Retail demand is sparse and skewed, so this bias is large and it is exactly the
    # wrong direction for inventory (you under-order and stock out).
    m_log = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.05, num_leaves=63,
                              min_child_samples=40, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=42, n_jobs=2, verbose=-1)
    m_log.fit(Xtr, np.log1p(ytr), eval_set=[(Xva, np.log1p(yva))],
              eval_metric="rmse", callbacks=[lgb.early_stopping(60, verbose=False)])
    resid = np.log1p(ytr) - m_log.predict(Xtr)
    sigma2 = float(np.mean(resid ** 2))
    va["lgb_log"] = np.clip(np.expm1(m_log.predict(Xva) + sigma2 / 2), 0, None)

    # --- model 2: Tweedie objective, fitted directly on the count scale -----
    # Tweedie (1 < p < 2) is the standard choice for intermittent, non-negative
    # demand: it is a Poisson-Gamma compound, i.e. many exact zeros plus a
    # continuous positive spend. No back-transform bias at all.
    m_tw = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2,
                             n_estimators=1500, learning_rate=0.05, num_leaves=63,
                             min_child_samples=40, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=42, n_jobs=2, verbose=-1)
    m_tw.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="rmse",
             callbacks=[lgb.early_stopping(60, verbose=False)])
    va["lgb"] = np.clip(m_tw.predict(Xva), 0, None)

    # --- model 3: Poisson objective (variance == mean) ----------------------
    m_po = lgb.LGBMRegressor(objective="poisson", n_estimators=1500, learning_rate=0.05,
                             num_leaves=63, min_child_samples=40, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=42, n_jobs=2, verbose=-1)
    m_po.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="rmse",
             callbacks=[lgb.early_stopping(60, verbose=False)])
    va["lgb_pois"] = np.clip(m_po.predict(Xva), 0, None)

    # --- model 4: per-department Tweedie (hierarchical split, M5 lesson) ---
    va["lgb_dept"] = 0.0
    for dept, idx in va.groupby("dept_code").groups.items():
        mask = (tr["dept_code"] == dept).values
        if mask.sum() < 500:
            va.loc[idx, "lgb_dept"] = va.loc[idx, "lgb"]
            continue
        m2 = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2,
                               n_estimators=500, learning_rate=0.05, num_leaves=31,
                               min_child_samples=30, random_state=42, n_jobs=2, verbose=-1)
        m2.fit(tr.loc[mask, feats], tr.loc[mask, "sales"].astype("float64"))
        va.loc[idx, "lgb_dept"] = m2.predict(va.loc[idx, feats])
    va["lgb_dept"] = np.clip(va["lgb_dept"], 0, None)

    # --- model 5: blend (average of the count models) ----------------------
    va["blend"] = (va["lgb"] + va["lgb_pois"] + va["lgb_log"]) / 3

    # --- model 3: hedge - take max of model & baseline (retail-safe bias) ---
    va["hedged"] = np.maximum(va["lgb"], va["naive_28"])

    results = []
    for col in ["naive_28", "mean_28", "lgb_log", "lgb", "lgb_pois", "lgb_dept", "blend", "hedged"]:
        yhat = va[col].round().clip(lower=0)
        results.append({
            "model": col,
            "MAE": round(float(np.mean(np.abs(yva - yhat))), 3),
            "RMSE": round(float(np.sqrt(np.mean((yva - yhat) ** 2))), 3),
            "RMSSE": round(float(np.sqrt(np.mean((yva - yhat) ** 2) / np.mean((yva - va["lag_28"]) ** 2))), 3),
            "sMAPE_%": round(smape(yva, yhat), 2),
            "bias": round(float(np.mean(yhat - yva)), 3),
        })
        va[f"{col}_r"] = yhat

    res = pd.DataFrame(results).sort_values("RMSSE")
    print("\n" + res.to_string(index=False))

    # WRMSSE on the model variants
    wr = {c: round(wrmsse(va, f"{c}_r", full), 4)
          for c in ["naive_28", "mean_28", "lgb_log", "lgb", "lgb_pois", "lgb_dept", "blend", "hedged"]}
    print("\nWRMSSE (lower better):", json.dumps(wr, indent=None))

    imp = pd.DataFrame({"feature": feats, "gain": m_tw.booster_.feature_importance("gain")}) \
        .sort_values("gain", ascending=False)
    imp.to_csv(REPORTS / "feature_importance.csv", index=False)
    va.to_parquet(DATA / "forecast_valid.parquet", index=False)
    res.to_csv(REPORTS / "leaderboard.csv", index=False)
    (REPORTS / "metrics.json").write_text(json.dumps({"leaderboard": results, "wrmssse": wr}, indent=2))
    print("\ntop features:\n" + imp.head(10).to_string(index=False))
    print(f"\nbest_iteration: tweedie={m_tw.best_iteration_} poisson={m_po.best_iteration_} log={m_log.best_iteration_}")


if __name__ == "__main__":
    FEATURES = None
    main()
