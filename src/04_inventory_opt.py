"""
STEP 3: forecast -> inventory policy -> rupees.

This is the layer that makes it an "optimized inventory management system"
rather than a forecasting notebook.

Per store-item it computes:
  safety stock   SS = z * sigma_d * sqrt(L)         (sigma = forecast error std)
  reorder point  ROP = mu_d * L + SS
  order qty      EOQ = sqrt(2 * D * S / (h * C))    (D=annual demand, S=order cost, h=holding rate, C=unit cost)
  service level  z from the newsvendor critical fractile  Cu/(Cu+Co)
                 Cu = lost margin per unit, Co = holding cost per unit per season
Then it simulates 28 days of ordering under (a) a naive weekly order rule and
(b) the optimized (s, S) policy, and reports the cost difference.

Run: python 04_inventory_opt.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS, FIG = ROOT / "data", ROOT / "reports", ROOT / "figures"
REPORTS.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

ORDER_COST = 25.0        # S: cost to place/receive one order line (Rs)
HOLD_RATE = 0.25         # h: annual holding cost as fraction of unit cost
MARKDOWN_RATE = 0.30     # Co: markdown/handling loss on a unit left unsold
MAX_DAYS_SUPPLY = 28.0   # practical cap: no single order may exceed 28 days of demand
REVIEW_DAYS = 7          # weekly ordering cycle
SIM_DAYS = 28
DAYS_YEAR = 365
rng = np.random.default_rng(7)


def norm_ppf(p):
    from scipy.stats import norm
    return float(norm.ppf(np.clip(p, 0.5, 0.999)))


BEST = "lgb_r"


def build_policies(va, full, seed=7):
    g = va.groupby(["store_id", "item_id"], observed=True)
    agg = g.agg(
        mu_d=(BEST, "mean"),
        actual_mean=("sales", "mean"),
        resid_std=("sales", lambda s: 0.0),   # replaced below
        lead=("lead_time_days", "first"),
        cost=("unit_cost", "mean"),
        price=("unit_price", "mean"),
    ).reset_index()
    # forecast error std from the holdout, floored by within-series demand variability
    r = va.assign(resid=va["sales"] - va[BEST]).groupby(["store_id", "item_id"], observed=True)["resid"].std().rename("err_std")
    v = va.groupby(["store_id", "item_id"], observed=True)["sales"].std().rename("dem_std")
    agg = agg.drop(columns=["resid_std"]).merge(r, on=["store_id", "item_id"]).merge(v, on=["store_id", "item_id"])
    agg["sigma_d"] = np.maximum(agg["err_std"], 0.5 * agg["dem_std"]).fillna(0.0)

    agg["margin"] = (agg["price"] - agg["cost"]).clip(lower=0.01)
    agg["overstock_cost"] = agg["cost"] * MARKDOWN_RATE
    # newsvendor: overstock cost = holding a unit for the season, understock = lost margin
    agg["critical"] = agg["margin"] / (agg["margin"] + agg["overstock_cost"])
    agg["z"] = agg["critical"].apply(norm_ppf)

    agg["D_annual"] = agg["mu_d"] * DAYS_YEAR
    agg["SS"] = (agg["z"] * agg["sigma_d"] * np.sqrt(agg["lead"])).round().clip(lower=0)
    agg["ROP"] = (agg["mu_d"] * agg["lead"] + agg["SS"]).round()
    agg["EOQ"] = np.sqrt(2 * agg["D_annual"] * ORDER_COST / (HOLD_RATE * agg["cost"].clip(lower=0.01)))
    # cap textbook EOQ by shelf-life / storage reality, otherwise the formula
    # happily tells a shop to buy 90 days of a 3-units-a-day item
    agg["days_supply_EOQ"] = agg["EOQ"] / agg["mu_d"].clip(lower=0.01)
    agg["EOQ"] = np.minimum(agg["EOQ"], agg["mu_d"] * MAX_DAYS_SUPPLY).clip(lower=1).round()
    agg["S_level"] = (agg["ROP"] + agg["EOQ"]).round()

    # ABC (revenue importance) x XYZ (demand variability) segmentation
    agg["revenue"] = agg["mu_d"] * agg["price"] * SIM_DAYS
    agg = agg.sort_values("revenue", ascending=False)
    agg["cum_share"] = agg["revenue"].cumsum() / agg["revenue"].sum()
    agg["ABC"] = np.where(agg["cum_share"] <= 0.70, "A", np.where(agg["cum_share"] <= 0.90, "B", "C"))
    agg["cv"] = (agg["sigma_d"] / agg["mu_d"].replace(0, np.nan)).fillna(9.9)
    agg["XYZ"] = np.where(agg["cv"] < 0.5, "X", np.where(agg["cv"] < 1.0, "Y", "Z"))
    agg["segment"] = agg["ABC"] + agg["XYZ"]
    return agg


def simulate(agg, actuals, mode="optimized"):
    """Vectorised 28-day (s,S) simulation. actuals: dict -> matrix [n_series, SIM_DAYS]."""
    n = len(agg)
    mu, sig, lead = agg["mu_d"].values, agg["sigma_d"].values, agg["lead"].values.astype(int)
    ROP, SS, EOQ = agg["ROP"].values, agg["SS"].values, agg["EOQ"].values.astype(int)
    cost = agg["cost"].values
    margin = agg["margin"].values

    # identical opening stock for both policies, so the comparison is about
    # the ordering rule, not about who got free starting inventory
    # each policy opens at its own steady-state level so we compare rules,
    # not who was handed the fuller shelf
    if mode == "optimized":
        on_hand = (ROP + EOQ).astype(float)
    elif mode == "eoq":
        on_hand = EOQ.astype(float)
    else:
        on_hand = np.maximum(mu * REVIEW_DAYS, 1.0)
    T = SIM_DAYS + int(lead.max()) + 8
    on_order = np.zeros((n, T))

    held, stockout, orders_placed = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(SIM_DAYS):
        arrivals = on_order[:, t]
        on_hand = on_hand + arrivals
        demand = actuals[:, t]
        sold = np.minimum(on_hand, demand)
        lost = demand - sold
        on_hand = on_hand - sold
        held += on_hand
        stockout += lost

        inv_pos = on_hand + on_order[:, t:min(T, t + int(lead.max()) + 1)].sum(axis=1)

        if mode == "optimized":
            # continuous review (s, S): when position hits ROP, refill to ROP + EOQ
            need = inv_pos <= ROP
            qty = np.where(need, np.maximum(ROP + EOQ - inv_pos, 0), 0)
        elif mode == "eoq":
            # textbook EOQ with no reorder point: order only when stock is gone
            need = inv_pos <= 0
            qty = np.where(need, EOQ, 0)
        else:
            # rule of thumb: every Monday order one week of forecast demand,
            # no safety stock and no lead-time buffer
            need = np.full(n, (t % REVIEW_DAYS) == 0)
            qty = np.where(need, mu * REVIEW_DAYS, 0)

        for k in range(n):
            if need[k] and qty[k] > 0:
                arr = t + max(1, lead[k])
                if arr < T:
                    on_order[k, arr] += qty[k]
                    orders_placed[k] += 1

    avg_inv = held / SIM_DAYS
    res = pd.DataFrame({
        "store_id": agg["store_id"], "item_id": agg["item_id"],
        "holding_cost": avg_inv * cost * HOLD_RATE * (SIM_DAYS / DAYS_YEAR),
        "stockout_units": stockout,
        "lost_margin": stockout * margin,
        "avg_inventory_units": avg_inv,
        "n_orders": orders_placed,
    })
    res["inventory_value_rs"] = avg_inv * cost
    res["order_cost"] = res["n_orders"] * ORDER_COST
    res["total_cost"] = res["holding_cost"] + res["lost_margin"] + res["order_cost"]
    res["service_level"] = 1 - res["stockout_units"] / np.maximum(actuals.sum(axis=1), 1e-9)
    return res


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    va = pd.read_parquet(DATA / "forecast_valid.parquet")
    full = pd.concat([pd.read_parquet(DATA / "train.parquet"), va[["store_id","item_id","date","sales"]]], ignore_index=True)

    agg = build_policies(va, full)
    agg = agg[agg["mu_d"] > 0].reset_index(drop=True)
    print(f"series with non-zero forecast: {len(agg):,} of {va.groupby(['store_id','item_id'], observed=True).ngroups:,}")
    agg.to_csv(REPORTS / "inventory_policy.csv", index=False)

    # actual demand matrix for the holdout window
    piv = va.pivot_table(index=["store_id", "item_id"], columns="date", values="sales").fillna(0)
    piv = piv.reindex(pd.MultiIndex.from_frame(agg[["store_id", "item_id"]])).values
    actuals = piv[:, :SIM_DAYS]

    sims = {"naive weekly order": simulate(agg, actuals, "naive"),
            "textbook EOQ (no ROP)": simulate(agg, actuals, "eoq"),
            "optimized (s,S) + safety stock": simulate(agg, actuals, "optimized")}

    cols = ["holding_cost", "lost_margin", "order_cost", "total_cost", "avg_inventory_units", "inventory_value_rs"]
    rows = []
    for name, sim in sims.items():
        r = {"policy": name}
        r.update(sim[cols].sum().round(0).to_dict())
        r["fill_rate"] = round(1 - sim["stockout_units"].sum() / actuals.sum(), 4)
        r["cost_per_unit_sold"] = round(sim["total_cost"].sum() / max(1.0, actuals.sum() - sim["stockout_units"].sum()), 2)
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(REPORTS / "policy_comparison.csv", index=False)
    print("\n=== 28-day cost simulation (M5: CA_1, FOODS_1+FOODS_3) ===")
    print(summary.to_string(index=False))
    base = summary.loc[0, "total_cost"]; best = summary["total_cost"].min()
    bi = summary["total_cost"].idxmin()
    print(f"\nbest policy: {summary.loc[bi,'policy']}  |  "
          f"total cost {100*(best-base)/base:+.1f}% vs naive  |  "
          f"fill rate {summary.loc[0,'fill_rate']:.3f} -> {summary.loc[bi,'fill_rate']:.3f}  |  "
          f"avg inventory {100*(summary.loc[bi,'avg_inventory_units']-summary.loc[0,'avg_inventory_units'])/summary.loc[0,'avg_inventory_units']:+.1f}%")
    sim_naive, sim_opt = sims["naive weekly order"], sims["optimized (s,S) + safety stock"]

    # segmentation table
    seg = agg.groupby("segment", observed=True).agg(
        series=("item_id", "count"),
        share_of_revenue=("revenue", lambda s: 100 * s.sum() / agg["revenue"].sum()),
        avg_service_target=("critical", "mean"),
        avg_safety_stock=("SS", "mean"),
        avg_lead=("lead", "mean"),
    ).round(2).sort_values("share_of_revenue", ascending=False)
    print("\n=== ABC-XYZ segments (who gets the attention) ===")
    print(seg.to_string())
    seg.to_csv(REPORTS / "segments.csv")

    # figure 1: forecast vs actual for 3 example series
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    pick = agg.sort_values("revenue", ascending=False).head(3)
    for ax, (_, row) in zip(axes, pick.iterrows()):
        s = va[(va.store_id == row.store_id) & (va.item_id == row.item_id)].sort_values("date")
        ax.plot(s["date"], s["sales"], label="actual", lw=1.4, color="#1f3a5f")
        ax.plot(s["date"], s[BEST], label="forecast", lw=1.6, color="#e07b39")
        ax.plot(s["date"], s["naive_28"], label="seasonal naive", lw=1, ls="--", color="#9aa5b1")
        ax.axhline(row["ROP"] / max(1, row["lead"]), color="#c0392b", lw=1, ls=":", label="ROP/lead-time rate")
        ax.set_title(f"{row.store_id} / {row.item_id}  ({row.segment})", fontsize=10, loc="left")
        ax.legend(fontsize=7, ncols=4); ax.grid(alpha=.25)
    fig.suptitle("28-day holdout: forecast vs actual", y=.995)
    fig.tight_layout(); fig.savefig(FIG / "forecast_vs_actual.png", dpi=140); plt.close(fig)

    # figure 2: naive vs optimized cost breakdown + segment bubble
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    parts = ["holding_cost", "lost_margin", "order_cost"]
    colors = ["#2e86ab", "#e07b39", "#6c757d"]
    bottom = np.zeros(len(sims))
    for p, c in zip(parts, colors):
        vals = [sims[k][p].sum() for k in sims]
        a1.bar([k.split(" (")[0] for k in sims], vals, bottom=bottom, label=p, color=c)
        bottom += np.array(vals)
    a1.set_title("28-day cost breakdown by policy"); a1.legend(fontsize=8); a1.grid(axis="y", alpha=.25)
    a1.tick_params(axis="x", labelsize=8)
    a2.scatter(agg["cv"].clip(upper=3), agg["revenue"], c=agg["margin"], s=18, cmap="viridis", alpha=.75)
    a2.set_xlabel("demand CV (variability)"); a2.set_ylabel("28-day revenue")
    a2.set_title("Where the money and the risk are"); a2.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "optimization.png", dpi=140); plt.close(fig)

    # figure 3: feature importance
    imp = pd.read_csv(REPORTS / "feature_importance.csv").head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp["feature"], imp["gain"] / imp["gain"].max(), color="#2e86ab")
    ax.set_title("What drives demand (gain, normalised)"); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "feature_importance.png", dpi=140); plt.close(fig)

    print(f"\nwrote: reports/inventory_policy.csv, reports/policy_comparison.csv, figures/*.png")


if __name__ == "__main__":
    main()
