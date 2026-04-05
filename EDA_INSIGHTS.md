# EDA Insights & Data Generation Decisions

A running log of exploratory findings from the synthetic dataset and the modelling decisions they motivated. Entries are dated and linked to the relevant source change.

---

## 2026-04-03 — Age, Risk Tolerance & Investment Goal Distributions

### Observations

**Age was uniformly distributed** across `[age_min, age_max]` (25–90).  
A uniform draw produces an unrealistic client book — in practice, private banking books skew toward middle-aged and older clients (peak around 45–55), with a fat tail of very elderly clients and a thin tail of younger ones.

**Risk tolerance and investment goal showed no age gradient.**  
Both were sampled independently of age, so a 25-year-old and an 85-year-old had statistically identical risk profiles. This breaks a core real-world relationship:
- Elderly clients tend to be more risk-averse (capital preservation, lower risk tolerance).
- Suitability rules around vulnerable clients (age > 75) become much less meaningful if the underlying profile distribution doesn't reflect that skew.

### Changes Made

| Component | Before | After |
|-----------|--------|-------|
| Age distribution | Uniform `[25, 90]` | Student's t, `df=4`, mean=50, std=12, clipped to `[25, 90]` |
| Risk tolerance age cap | Hard cap from age 65+ | Graduated cap from age 55+, with ε=5% exception probability |
| Investment goal | Age-blind draw | Age-weighted draw (shifts toward preservation/income above 40), with ε=5% exception |

#### Age: Student's t-distribution (`df=4`)

A t-distribution with low degrees of freedom gives fatter tails than a Gaussian — more very-young and very-old clients than a normal distribution, while still peaking around 50. `df=4` is a reasonable default; lower values (e.g. `df=2`) would make the tails even heavier.

```
df=4, mean=50, std=12  →  most clients 38–62, visible tail past 75
```

#### Risk tolerance: graduated age cap + epsilon

```python
age_risk_cap = np.where(ages > 80, 1,
               np.where(ages > 75, 2,
               np.where(ages > 70, 3,
               np.where(ages > 65, 4,
               np.where(ages > 55, 4, 5)))))
# ~5% of clients are exceptions and keep their sampled value
is_rt_exception = rng.random(n) < cfg.age_risk_epsilon
risk_tolerance = np.where(is_rt_exception, risk_tolerance, np.minimum(risk_tolerance, age_risk_cap))
```

The epsilon exception prevents a mechanical hard rule from eliminating all high-risk elderly clients — some clients are genuinely aggressive regardless of age, which is also a realistic source of suitability alerts.

#### Investment goal: age-weighted probabilities + epsilon

Probability mass shifts linearly from growth/speculation toward capital-preservation/income as age rises above 40, reaching full shift at age 80+:

```
age_factor = clip((age - 40) / 40, 0, 1)
P(capital preservation) += age_factor * 0.4
P(income)               += age_factor * 0.2
P(growth)               -= age_factor * 0.3
P(speculation)          -= age_factor * 0.3
```

~5% of clients ignore the age weighting entirely (epsilon draw from base distribution).

### Config knobs (`config.json → clients.age_epsilon`)

```json
"age_epsilon": {
  "risk_tolerance": 0.05,
  "investment_goal": 0.05
}
```

Tune independently. Lower epsilon = stricter age-driven conservatism. Higher epsilon = more age-profile diversity (useful if you want more label noise from unexpected client behaviour).

### Source files changed

- [`src/synthetic_trade/data_generation.py`](src/synthetic_trade/data_generation.py) — `GenerationConfig`, `load_generation_config`, `generate_clients`, `_sample_investment_goal_by_age`
- [`config/config.json`](config/config.json) — added `clients.age_epsilon`

---

---

## 2026-04-03 — Boundary spikes on age distribution (follow-up)

### Observation

After switching to the t-distribution, EDA showed unexpected spikes at exactly `age_min=25` and `age_max=90`. Both boundary ages had far more clients than neighbouring ages.

### Root cause

`np.clip` was used to enforce bounds. Any t-draw that fell outside `[25, 90]` got pinned to the nearest boundary, accumulating all out-of-range probability mass at the two edges.

### Fix

Replaced `np.clip` with **rejection sampling**: out-of-range draws are discarded and redrawn until all values fall within bounds naturally. This produces a proper truncated t-distribution with no boundary artefacts.

```python
ages = np.empty(cfg.num_clients, dtype=int)
unfilled = np.ones(cfg.num_clients, dtype=bool)
while unfilled.any():
    n = int(unfilled.sum())
    candidates = np.round(50 + rng.standard_t(df=4, size=n) * 12).astype(int)
    in_range = (candidates >= cfg.age_min) & (candidates <= cfg.age_max)
    ages[unfilled] = np.where(in_range, candidates, 0)
    unfilled[unfilled] = ~in_range
```

Converges in 1–2 iterations in practice since only a small tail fraction falls outside `[25, 90]`.

*Add new entries below as EDA continues.*

---

## 2026-04-05 — Product Feature Distributions: Tenor, Commission, Lockup

### Observation

All 50 products of the same type were **identical** on `recommended_tenor_yrs`, `commission_rate`, and `lockup_days`. Every Structured product had exactly `commission_rate=0.02`, `recommended_tenor_yrs=3.0`, and `lockup_days=365`. This was unrealistic and would cause spurious patterns in any ML model — the feature carries no within-type variance and can't discriminate individual products.

### Changes Made

Replaced flat lookup maps with **per-product draws from a clipped Normal distribution**. Means are set to the original lookup values; standard deviations are calibrated so roughly ±1σ spans a plausible real-world range for each product type.

#### `recommended_tenor_yrs`

| Product Type  | Mean (yrs) | Std  | Clip      | Rationale |
|---------------|-----------|------|-----------|-----------|
| Equity        | 1.0       | 0.50 | [0.25, 3] | Short to medium horizon; some multi-year equity mandates |
| Fixed Income  | 3.0       | 1.50 | [0.5, 7]  | Broad range: short bonds to medium-duration notes |
| Structured    | 3.0       | 0.75 | [1, 5]    | Typically 1–5 year ELNs/FCNs |
| Derivative    | 0.5       | 0.20 | [0.08, 1.5] | Short-dated options/forwards; weeks to months |
| Alternative   | 5.0       | 2.00 | [1, 10]   | Long-term funds; broad dispersion reflects fund-of-funds vs direct |

#### `commission_rate`

| Product Type  | Mean   | Std    | Clip            | Rationale |
|---------------|--------|--------|-----------------|-----------|
| Equity        | 0.50%  | 0.10%  | [0.10%, 1.00%]  | Standard equity brokerage |
| Fixed Income  | 0.75%  | 0.20%  | [0.20%, 1.50%]  | Slightly higher for bonds due to spread |
| Structured    | 2.00%  | 0.50%  | [0.50%, 4.00%]  | Wide spread on bespoke structured notes |
| Derivative    | 1.50%  | 0.40%  | [0.30%, 3.00%]  | Options premium varies by strike/tenor |
| Alternative   | 2.00%  | 0.50%  | [0.50%, 4.00%]  | Placement fee on alternative funds |

#### `lockup_days`

| Product Type  | Mean (days) | Std | Clip           | Rationale |
|---------------|------------|-----|----------------|-----------|
| Equity        | 3          | 4   | [0, 15]        | Settlement window; many are 0, tail to ~15 |
| Fixed Income  | 30         | 15  | [0, 90]        | Cooling-off / settlement ranges |
| Structured    | 365        | 90  | [180, 730]     | Semi-annual to 2-year lockups common |
| Derivative    | 2          | 3   | [0, 14]        | No lockup to short settlement |
| Alternative   | 730        | 180 | [180, 1825]    | 6-month to 5-year redemption gates |

For Equity and Derivative, the mean is near zero and the distribution is clipped at 0 — this produces a right-skewed integer distribution where most products have short or zero lockup, which is realistic.

### Impact on Suitability Rules

The Liquidity Mismatch rule selects products with `lockup_days >= 365` as misrepresentation candidates. With the new distributions:
- **Structured** (mean 365, std 90): ~50% of products qualify — intentional, as borderline products are more realistic triggers.
- **Alternative** (mean 730, std 180): effectively all products qualify.
- **Fixed Income** (mean 30, clip 90): none qualify — correctly excluded from this typology.

### Source files changed

- [`src/synthetic_trade/data_generation.py`](src/synthetic_trade/data_generation.py) — `generate_products`: replaced `commission_map`, `tenor_map`, `lockup_map` dicts with `tenor_params`, `commission_params`, `lockup_params` dicts of `(mean, std, lo, hi)` tuples; vectorised draw per product type using `rng.normal` + `np.clip`.
