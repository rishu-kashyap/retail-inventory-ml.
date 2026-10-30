# Optimized Retail Inventory Management with ML — Dataset Choice & Build Guide

A project blueprint, plus a **working reference implementation** in this folder that you can
run today and then re-point at the real dataset.

---

## 1. Which dataset should you use?

There is no single "best" one — it depends on which half of the problem you want to show.
"Optimized inventory management" = **forecasting + decision-making under uncertainty**. Pick
the dataset that lets you do both.

| Dataset | Size | What you get | Best for | Watch out |
|---|---|---|---|---|
| **M5 Forecasting — Accuracy** (Walmart, Kaggle) ⭐ | 30,490 item-store series × 1,941 days, 10 stores | Daily unit sales, sell price, promos, SNAP/payday flags, calendar events, item/dept/category hierarchy | **The benchmark.** Hierarchy lets you do reconciliation; rich covariates; published leaderboard to compare against | ~450 MB; must downcast dtypes; `sell_prices.csv` has gaps |
| **Corporación Favorita** (Kaggle) | ~125M rows, 4,100 SKUs × 54 stores, 5 yrs | Daily sales, promos, transactions, oil price, holidays | Promo-heavy grocery; great for feature engineering depth | Huge — needs DuckDB/polars or a subset; sales are *log1p*-transformed and per-family |
| **Rossmann Store Sales** (Kaggle) | 1,017 stores × 2.5 yrs | Store-day sales, promo, holiday, store metadata | Store-level forecasting, calendar effects | **No SKU level** — so no real per-item replenishment story |
| **DataCo Smart Supply Chain** (Kaggle) | 180,519 orders, 53 cols | Orders, shipping times, late-delivery risk, regions, lead times | **Best for the operations side**: lead-time + delay prediction, EOQ/ROP per category | No inventory balance or stock ledger; demand is order-driven, not shelf-driven |
| **Online Retail II / UCI** | ~1M transactions | Invoice-level UK e-commerce | RFM, churn, basket analysis | Returns/negatives to clean; no stock levels |

### My recommendation

**Use M5 as your core, and add DataCo for the operational layer if you want a bigger project.**
That combination is uncommon, so it reads as more than a copy-paste Kaggle notebook:

- M5 gives you legitimate **item-store daily demand** → forecasting + safety stock.
- DataCo gives you **lead times and late-delivery risk** → your safety stock stops being a
  constant `L` and becomes a *predicted* `L`, which is a real research-flavoured contribution.

If your deadline is tight (2–3 weeks) or it is a single-semester course: **M5 alone, one store,
one or two departments.** Do not try to model all 42,840 series on a laptop.

> Kaggle CLI: `kaggle competitions download -c m5-forecasting-accuracy -p data/raw`
> (needs `~/.kaggle/kaggle.json` from Account → Create New API Token)

---

## 2. The architecture you should build

The mistake almost everyone makes is stopping at "I trained a model, here is my RMSE."
An *inventory* project needs the last two layers — that is where the marks and the story are.

```
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌────────────────┐   ┌──────────────┐
│ 1. Ingest   │──▶│ 2. Features  │──▶│ 3. Forecast   │──▶│ 4. Policy      │──▶│ 5. Decision  │
│ sales, cal, │   │ lags, rolls, │   │ LightGBM /    │   │ safety stock,  │   │ order qty,   │
│ price, lead │   │ promo, price │   │ quantiles     │   │ ROP, EOQ, ABC  │   │ ₹ saved,     │
│ times       │   │ (leak-safe)  │   │ + baselines   │   │ + simulation   │   │ dashboard    │
└─────────────┘   └──────────────┘   └───────────────┘   └────────────────┘   └──────────────┘
                                          │                      ▲
                                          └── σ of forecast ─────┘   ← errors drive safety stock
```

**Why quantiles matter:** point forecasts can't size a safety buffer. You need an *upper*
quantile at your target service level, or a residual-error σ. This is the single most
under-appreciated step, and it's what separates a real inventory project from a sales-prediction
notebook.

---

## 3. How to do it — phased plan

### Phase 0 — Scope (½ day)
- Pick **1 store × 1–2 departments** (e.g. `CA_1` + `FOODS_3`) or 2–4k series.
- Define the decision: *"weekly order quantity per store-item, 28-day horizon, target 95% fill rate."*
- Write down your business assumption costs: order cost `S`, holding rate `h`, markdown loss.

### Phase 1 — EDA that actually informs the model (2–3 days)
Look for these specific things, each of which becomes a feature:
- Weekly seasonality → autocorrelation peaks at lags 7/14/21
- Payday/SNAP spikes → early-month effect, holiday dummies
- % of zero-sales days per series → intermittent demand needs separate handling
- Promo lift distribution → promo flag + price ratio
- Price–demand relation (finite elasticity) → price relative to 28-day average
- **Stockout zeros** (sales = 0 because the shelf was empty, not because nobody wanted it) →
  flag them, or you'll train your model to under-order

### Phase 2 — Leakage-safe features (3–4 days)
The cardinal rule: **every feature must be knowable at the forecast origin** (28 days before the
day you predict). All lags and rollings shifted by ≥ 28. Calendar, planned promos, and price
are legitimately "future" information — they're known in advance in real retail.

Feature set that works (see `src/02_features.py`):
- Lags: 28, 35, 42, 56, 364
- Rolling mean/std/median over 7/28/56/91, **shifted 28**
- Momentum: `rmean_28 / rmean_56`
- Intermittency: share of zeros in last 28 days
- Price: lagged price, price relative to its own 28-day mean, promo/planned price
- Calendar: weekday, month, day, week-of-year, weekend, event, SNAP
- Static: item/dept/store encodings, series mean (computed **only on train**)

### Phase 3 — Model + honest baselines (4–5 days)
Always report these three baselines — reviewers will ask:
1. **Seasonal naive**: `sales[t-28]`
2. **Moving average** of last 28 days
3. **Your model**

Use LightGBM (global model, or one per store/dept — per-state models were the biggest single
gain in top M5 solutions). Train on `log1p(sales)` for count data, clip predictions at 0.

**Metrics — use the right ones:**
- `RMSSE` (M5): error scaled by each series' own naive error → one giant series can't dominate
- `WRMSSE`: additionally dollar-weighted by series revenue → this is the one that matters
- `sMAPE`, `bias` (mean error — a systematic negative bias means chronic understocking)
- **Interval coverage**: does your 95% interval actually contain 95% of actuals?

> Generic RMSE/accuracy is the most common grading criticism. `src/03_train_forecast.py`
> computes all of the above plus a WRMSSE implementation.

### Phase 4 — Forecast → inventory policy (3–4 days)
Per store-item:

```
σ_d  = std of forecast residuals (fall back to demand std for sparse series)
z    = Φ⁻¹( Cu / (Cu + Co) )        Cu = lost margin, Co = markdown/handling loss
SS   = z · σ_d · √L                 safety stock
ROP  = μ_d · L + SS                 reorder point
EOQ  = √(2 · D · S / (h · C))       capped by shelf life / days-of-supply
```

Then **simulate 28 days** of ordering under (a) a naive weekly rule, (b) textbook EOQ with no
ROP, (c) the optimized (s, S) policy — and report holding cost, lost margin, order cost, fill
rate. This simulation is your headline result. `src/04_inventory_opt.py` does exactly this.

Segment with **ABC-XYZ** (revenue importance × demand variability) to say *where* to spend
attention: A-items get tight service targets, CZ-items get simple rules.

### Phase 5 — Evaluation & write-up (3–4 days)
Report a table like (numbers below are from the synthetic stand-in in this repo):

| Model | MAE | RMSSE | sMAPE | bias |
|---|---|---|---|---|
| Seasonal naive | 2.452 | 1.000 | 90.8 | +0.07 |
| Moving average 28 | 1.924 | 0.762 | 78.4 | +0.04 |
| **LightGBM (global)** | **1.662** | **0.682** | **71.9** | −0.58 |

| Policy | Total cost | Fill rate | Avg inventory |
|---|---|---|---|
| Naive weekly order | ₹226,704 | 76.0% | 9,681 units |
| Textbook EOQ, no ROP | ₹105,281 | 84.3% | 35,298 units |
| **Optimized (s, S)** | **₹38,209** | **98.8%** | 62,121 units |

### Phase 6 — Deployment (optional, 2–3 days, big presentation value)
Streamlit / FastAPI app: pick store + item → show forecast fan chart, current ROP, recommended
order, days-of-cover, and the ₹ impact. Even a static Power BI/Plotly dashboard scores well.

---

## 4. Pitfalls that cost marks

1. **Leakage**: using `sales[t-1]` to predict `sales[t+28]`. Fix: shift everything by horizon.
2. **Scaling on the full dataset** before splitting — fit scalers/encoders/means on train only.
3. **Stockout censoring**: zero sales ≠ zero demand. Cap or flag at-history.
4. **Plain RMSE on skewed demand**: a few bestsellers dominate; slow movers silently fail.
5. **No baseline**: "my model got RMSE 3.0" means nothing without the naive comparison.
6. **Unbounded EOQ**: textbook EOQ for a cheap slow mover suggests absurd order sizes — cap by
   days-of-supply (done in `src/04_inventory_opt.py`).
7. **Explaining inventory only, not *service level trade-off***: always show the cost-vs-fill-rate
   frontier. That's the "optimized" part.
8. **No uncertainty**: a point forecast cannot justify safety stock.

---

## 5. Extensions (pick one for a stronger grade)

- **Predicted lead time** (train on DataCo) instead of a constant `L` — directly improves SS.
- **Quantile regression** (LightGBM `objective='quantile'`) → asymmetric cost-aware ordering.
- **Newsvendor optimisation of the service level** instead of a fixed 95%.
- **Multi-echelon**: warehouse → store, with OR-Tools `pywraplp` (transportation/allocation LP).
- **Reinforcement learning** for ordering: DQN/PPO with state = (on-hand, pipeline, forecast,
  day-of-week), reward = −(holding + stockout + order cost). Great talking point, but be careful
  to benchmark it against the (s,S) baseline — RL often loses to it, and *saying so honestly with
  evidence* is a strong result.
- **Hierarchical reconciliation** (bottom-up vs MinT) across item → dept → store.

---

## 6. Results on the real M5 data

Scope: **store CA_1, departments FOODS_1 + FOODS_3 → 1,039 item-store series**, 1,551 training
days (2011-01-30 → 2016-04-24), 28-day holdout **2016-04-25 → 2016-05-22**.

### Forecast accuracy

| Model | MAE | RMSE | RMSSE | sMAPE % | bias |
|---|---|---|---|---|---|
| Seasonal naive (lag 28) | 2.316 | 4.692 | 1.000 | 100.3 | −0.25 |
| Moving average 28d | 1.719 | 3.356 | 0.715 | 92.5 | −0.28 |
| LightGBM, log1p + bias correction | 1.698 | 3.407 | 0.726 | 97.1 | −0.50 |
| LightGBM, Tweedie (p=1.2) | 1.714 | 3.307 | 0.705 | 98.6 | −0.33 |
| **LightGBM, Poisson** | 1.706 | 3.208 | **0.684** | 100.6 | −0.26 |
| LightGBM per-department Tweedie | **1.684** | **3.209** | **0.684** | 92.9 | −0.34 |
| Blend (Tweedie + Poisson + log) | 1.702 | 3.281 | 0.699 | 99.2 | −0.36 |

WRMSSE (M5's dollar-weighted metric): naive 1.050 → **Poisson 0.698**, per-dept 0.701.

**Two lessons this data taught us** (both worth writing up — they are exactly the kind of
finding that earns marks):

1. **The log1p + `expm1` trick is biased low.** On skewed, sparse demand, `E[y] = exp(μ + σ²/2)`,
   so a naive back-transform systematically under-forecasts — the wrong direction for inventory.
   Adding the correction moved log-target WRMSSE 0.729 → 0.711.
2. **A plain 28-day moving average initially beat LightGBM** (RMSSE 0.715 vs 0.774). The model
   only won after switching to a **count-appropriate objective (Poisson / Tweedie)**. If you
   don't report the baseline honestly you'll never notice your model is losing to a one-liner.

Poisson wins on RMSSE (squared-error metric), Tweedie wins on MAE/sMAPE. For inventory the
squared-error metric is the relevant one — it penalises the large misses that cause stockouts.

### Inventory policy (28-day simulation, 993 series with non-zero forecast)

| Policy | Holding cost | Lost margin | Order cost | **Total** | Avg inventory | Fill rate |
|---|---|---|---|---|---|---|
| Naive weekly order | 444 | 11,710 | 99,300 | **111,455** | 13,757 | 82.4% |
| Textbook EOQ, no ROP | 1,128 | 8,656 | 14,450 | **24,234** | 35,112 | 86.9% |
| **Optimized (s, S) + safety stock** | 1,517 | 2,890 | 15,025 | **19,432** | 47,191 | **95.8%** |

Total cost −82.6% vs the naive rule, fill rate 82.4% → 95.8%, at the cost of 3.4× the average
inventory. Read that trade-off carefully: cutting stockout losses by 75% costs roughly ₹56k of
extra inventory value — a 1.4-month payback on these assumptions. **Always present the
inventory line next to the cost line**, or the saving is meaningless.

ABC-XYZ: 225 AY series (42% of revenue) get the tight service targets; 225 CZ series (6.7% of
revenue) can run simple rules.

---

## 7. What is already built in this folder

```
retail-inventory-ml/
├── fetch_data.sh               downloads the real M5 files (public mirror)
├── src/
│   ├── 01_load_m5.py          REAL M5 -> long parquet (chunked, scope-selectable, laptop-safe)
│   ├── 01_simulate_data.py    synthetic M5-shaped data (fallback / unit-testing)
│   ├── 02_features.py         leakage-safe feature builder + 28-day holdout split
│   ├── 03_train_forecast.py   LightGBM vs baselines, RMSSE/WRMSSE/sMAPE/bias
│   └── 04_inventory_opt.py    SS/ROP/EOQ, ABC-XYZ, 28-day (s,S) simulation, figures
├── data/                      parquet outputs
├── figures/                   forecast_vs_actual.png, optimization.png, feature_importance.png
└── reports/                   leaderboard.csv, inventory_policy.csv, policy_comparison.csv, segments.csv
```

Run it end to end on the real data:

```bash
pip install -r requirements.txt
./fetch_data.sh                            # ~310 MB into data/raw/
python src/01_load_m5.py                   # CA_1 x FOODS_1,FOODS_3 -> 2.0M rows
python src/02_features.py                  # 29 leak-safe features, ~1.25 GB peak RAM
python src/03_train_forecast.py            # 7 models vs 2 baselines, ~80 s
python src/04_inventory_opt.py             # policies + 28-day simulation + figures
```

Peak RAM was measured at ~1.3 GB, so this runs on a 2 GB machine. Widen the scope with
`--stores CA_1 TX_1 WI_1` (needs ~8 GB) or `--all` (~16 GB).

### What M5 does *not* contain — and how we handled it

M5 has sales, price, calendar and events. It has **no inventory, no lead time, no cost**. Any
inventory project on M5 must declare these. Ours (in `01_load_m5.py`, printed on every run):

| Column | Assumption | How to replace it properly |
|---|---|---|
| `lead_time_days` | CA=5, TX=7, WI=4 days | Real supplier data, or **predict** it from the DataCo dataset |
| `unit_cost` | `sell_price × 0.65` (35% margin) | Company P&L |
| `unit_price` | `sell_price` | — |
| `promo` | price ≥5% below trailing 28-day avg | Real promo calendars |

Two things this means for your report: (a) **the rupee totals are illustrative**, the *relative*
comparison between policies is what carries the argument; (b) a sensitivity sweep over margin
and lead time is a cheap, high-credit addition — try `MARKDOWN_RATE` and `HOLD_RATE` in
`04_inventory_opt.py`.

The synthetic generator deliberately includes the hard parts of real retail data — intermittent
series, promo lifts, SNAP/payday effects, price variation, mixed lead times — so your pipeline
logic transfers directly.
