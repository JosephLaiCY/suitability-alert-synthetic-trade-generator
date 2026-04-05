# CLAUDE.md — Synthetic Trade Data & Suitability Alert Generator

## Project Overview

**Domain:** Wealth Management / Private Banking Compliance  
**Objective:** Generate a realistic synthetic dataset of private banking trades and suitability alerts for ML model training, focused on **Mis-selling and Conduct Risk** ("Bad Apple" advisor behaviours).

This is a **learning / portfolio project**. The user is implementing features independently and asks for guidance and review afterward — do not preemptively rewrite or extend code beyond what is asked.

---

## Architecture

### Pipeline (top-down)
```
config/config.json
  → generate_advisors()
  → generate_clients()
  → generate_products()
  → generate_trades()
  → apply_suitability_rules()   ← adds flag_* columns, is_alert, is_genuine
  → export to output/advisors.csv, output/clients.csv, output/products.csv, output/trade_suitability.csv
```

### Key source files
| File | Purpose |
|------|---------|
| `src/synthetic_trade/data_generation.py` | All entity generation logic |
| `src/synthetic_trade/suitability_rules.py` | Vectorised rule application, is_alert, is_genuine |
| `main.py` | CLI orchestrator — generates all entities, applies rules, exports to `output/` |
| `notebooks/eda.py` | marimo EDA notebook — reads from `output/`, visualises advisors, clients, products |
| `config/config.json` | Single source of truth for all parameters |

---

## Data Model

### Advisors
| Field | Type | Notes |
|-------|------|-------|
| `advisor_id` | string | |
| `conduct_flag` | bool | True = "Bad Apple" |
| `risk_typology` | categorical | Clean, Unsuitable Recs, Churning, Misrepresentation, Liquidity Mismatch |

### Clients
| Field | Type | Notes |
|-------|------|-------|
| `client_id` | string | |
| `advisor_id` | string | FK to advisors |
| `age` | int | Used for Vulnerable Client logic (>75) |
| `risk_tolerance` | int | 1 (Conservative) – 5 (Aggressive) |
| `aum_usd` | float | |
| `ke_equities` | bool | Knowledge & Experience flags |
| `ke_structured_prods` | bool | |
| `ke_derivatives` | bool | |
| `ke_fixed_income` | bool | |
| `ke_alternatives` | bool | |
| `liquidity_needs` | categorical | High / Medium / Low |
| `investment_horizon` | categorical | Short / Medium / Long |
| `investment_goal` | categorical | Capital Preservation / Income / Growth / Speculation |

### Products
| Field | Type | Notes |
|-------|------|-------|
| `product_id` | string | |
| `product_type` | categorical | Equity, Fixed Income, Structured, Derivative, Alternative |
| `is_structured` | bool | True for ELNs, FCNs, Accumulators |
| `actual_risk_rating` | int | 1–5, internal bank rating |
| `recommended_tenor_yrs` | float | |
| `commission_rate` | float | Higher for Structured/Complex products |
| `lockup_days` | int | |

### Trades (final output schema)
| Field | Type | Notes |
|-------|------|-------|
| `trade_id` | string | |
| `client_id` | string | |
| `advisor_id` | string | |
| `product_id` | string | |
| `trade_date_offset_days` | int | Days from start of 365-day window |
| `trade_amount` | float | |
| `solicitation_type` | categorical | Solicited / Unsolicited |
| `estimated_comm` | float | |
| `typology_applied` | string | Clean or bad-apple typology name |
| `actual_risk_rating` | int | Copied from product at trade time |
| `marketed_risk_rating` | int | May be lowered for Misrepresentation |
| `flag_ke_mismatch` | bool | Suitability rule breach flags |
| `flag_asset_concentration` | bool | |
| `flag_risk_mismatch` | bool | |
| `flag_liquidity_mismatch` | bool | |
| `is_alert` | bool | True if any rule flag is breached |
| `is_genuine` | bool | True if is_alert AND typology_applied != "Clean" |

---

## Bad Apple Typologies

| Typology | Behaviour |
|----------|-----------|
| **Unsuitable Recs** | High-risk/complex products sold to conservative or elderly clients |
| **Churning** | Cluster of 3–5 trades within 60-day window; high-commission structured products; solicitation_type = Solicited |
| **Misrepresentation** | marketed_risk_rating set lower than actual_risk_rating |
| **Liquidity Mismatch** | Products with high lockup_days sold to clients with High liquidity_needs |

---

## Suitability Rules (implemented)

| Rule | Logic |
|------|-------|
| K&E Mismatch | client.ke_{product_type} is False |
| Asset Concentration | trade_amount > 15% of client.aum_usd |
| Risk Mismatch | product.actual_risk_rating > client.risk_tolerance |
| Liquidity Mismatch | product.lockup_days > threshold for client.liquidity_needs |

**Descoped:** Goal Mismatch and Horizon Mismatch rules intentionally removed from scope.

---

## Labeling

- `is_genuine = True`: alert triggered AND trade originated from a Bad Apple typology
- `is_genuine = False`: alert triggered but advisor is Clean (justifiable mechanical mismatch)
- Label noise: 2% random flip of labels (configured in `config/config.json` → `labeling.label_noise_rate`)
- Target class balance: ~85% justifiable mismatches, ~15% genuine conduct issues

---

## Planned Work (in progress by user)

1. **ML model notebook** (`notebooks/ml_model.py`) — classify genuine vs non-genuine alerts using Logistic Regression + Gradient Boosting
2. **Suitability risk dashboard** (`notebooks/dashboard.py`) — marimo app with KPI cards, advisor leaderboard, typology breakdown, temporal patterns; interactive filters; plotly charts

### Completed
- Data generation pipeline runs via `main.py`; outputs written to `output/` directory
- Suitability rules applied and `is_alert` / `is_genuine` labels computed
- EDA notebook (`notebooks/eda.py`) built with interactive altair charts

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `numpy`, `pandas` | Data generation and manipulation |
| `marimo` | Notebook + dashboard framework |
| `plotly` | (planned) Interactive charts for dashboard |
| `scikit-learn` | (planned) ML model |
| `pyarrow` | (planned) Parquet export |

Run pipeline: `python main.py`  
Run EDA notebook: `marimo edit notebooks/eda.py`  
Run dashboard (once built): `marimo run notebooks/dashboard.py`
