"""
Model training and serialization script for Retail Inventory ML.
Trains the LightGBM Tweedie demand forecasting model and saves artifacts to models/.
"""
import json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def train_and_export():
    print("Loading datasets...")
    tr = pd.read_parquet(DATA / "train.parquet")
    va = pd.read_parquet(DATA / "valid.parquet")

    drop = {"store_id", "item_id", "date", "sales", "d", "sid", "price_imputed"}
    feats = [c for c in tr.columns if c not in drop]
    print(f"Training with {len(feats)} features: {feats}")

    Xtr, ytr = tr[feats], tr["sales"].astype("float64")
    Xva, yva = va[feats], va["sales"].astype("float64")

    print("Fitting LightGBM Tweedie Regressor...")
    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.2,
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        Xtr,
        ytr,
        eval_set=[(Xva, yva)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )

    val_preds = np.clip(model.predict(Xva), 0, None)
    residuals = yva - val_preds
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    overall_err_std = float(np.std(residuals))

    print(f"Validation MAE: {mae:.3f}, RMSE: {rmse:.3f}, Resid Std: {overall_err_std:.3f}")

    # Save model artifact
    model_path = MODELS / "lgb_inventory_model.joblib"
    joblib.dump(model, model_path, compress=3)
    print(f"Saved model artifact to {model_path} ({model_path.stat().st_size / (1024*1024):.2f} MB)")

    # Extract metadata for UI and inference
    stores = sorted(list(tr["store_id"].dropna().unique().astype(str)))
    items = sorted(list(tr["item_id"].dropna().unique().astype(str)))
    
    # Store-item summary statistics for fast lookup & defaults
    lookup_df = tr.groupby(["store_id", "item_id"], observed=True).agg({
        "sales": ["mean", "std"],
        "unit_cost": "mean",
        "unit_price": "mean",
        "lead_time_days": "mean",
        "dept_code": "first",
        "store_code": "first",
        "item_code": "first",
        "item_mean": "first",
        "item_store_mean": "first"
    }).reset_index()
    lookup_df.columns = [
        "store_id", "item_id", "hist_mean_sales", "hist_std_sales",
        "unit_cost", "unit_price", "lead_time_days", "dept_code",
        "store_code", "item_code", "item_mean", "item_store_mean"
    ]
    lookup_df["store_id"] = lookup_df["store_id"].astype(str)
    lookup_df["item_id"] = lookup_df["item_id"].astype(str)
    for c in ["hist_mean_sales", "hist_std_sales", "unit_cost", "unit_price", "lead_time_days", "dept_code", "store_code", "item_code", "item_mean", "item_store_mean"]:
        lookup_df[c] = pd.to_numeric(lookup_df[c], errors="coerce").fillna(0)

    # Dept mapping
    dept_map = {
        0: "FOODS_1",
        1: "FOODS_2",
        2: "FOODS_3",
        3: "HOBBIES_1",
        4: "HOUSEHOLD_1",
        5: "HOUSEHOLD_2"
    }

    metadata = {
        "features": feats,
        "best_iteration": int(model.best_iteration_ or 100),
        "validation_metrics": {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "std_error": round(overall_err_std, 3)
        },
        "stores": stores,
        "items": items[:100],  # top items for dropdown selector
        "dept_map": dept_map,
        "default_lead_time": 5,
        "default_order_cost": 25.0,
        "default_holding_rate": 0.25,
        "default_markdown_rate": 0.30
    }

    metadata_path = MODELS / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata to {metadata_path}")

    # Save a lightweight fast lookup table for the web application
    lookup_path = MODELS / "item_lookup.parquet"
    lookup_df.to_parquet(lookup_path, index=False)
    print(f"Saved item lookup to {lookup_path}")


if __name__ == "__main__":
    train_and_export()
