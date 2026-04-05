from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class GenerationConfig:
    num_advisors: int
    num_clients: int
    num_trades: int
    bad_apple_rate: float
    typology_weights: Dict[str, float]
    churning_cluster_min_trades: int
    churning_cluster_max_trades: int
    churning_window_days: int
    age_min: int
    age_max: int
    aum_min: float
    aum_max: float
    trade_date_horizon_days: int
    risk_tolerance_distribution: Dict[int, float]
    liquidity_needs_distribution: Dict[str, float]
    investment_goal_distribution: Dict[str, float]
    ke_probability: Dict[str, float]
    age_risk_epsilon: float
    age_goal_epsilon: float
    seed: Optional[int]


def load_generation_config(raw_config: Dict) -> GenerationConfig:
    sizes = raw_config.get("sizes", {})
    conduct = raw_config.get("conduct", {})
    clients = raw_config.get("clients", {})
    trades_cfg = raw_config.get("trades", {})
    age_cfg = clients.get("age", {})
    aum_cfg = clients.get("aum_usd", {})
    client_generation = raw_config.get("client_generation", {})
    distributions = client_generation.get("distributions", {})
    ke_probability = client_generation.get("ke_probability", {})

    risk_tolerance_dist_raw = distributions.get(
        "risk_tolerance", {"1": 0.10, "2": 0.10, "3": 0.20, "4": 0.20, "5": 0.40}
    )
    risk_tolerance_distribution = {
        int(k): float(v) for k, v in risk_tolerance_dist_raw.items()
    }

    liquidity_needs_distribution = {
        str(k): float(v)
        for k, v in distributions.get(
            "liquidity_needs", {"High": 0.1, "Medium": 0.5, "Low": 0.4}
        ).items()
    }
    investment_goal_distribution = {
        str(k): float(v)
        for k, v in distributions.get(
            "investment_goal",
            {
                "Capital Preservation": 0.10,
                "Income": 0.40,
                "Growth": 0.40,
                "Speculation": 0.10,
            },
        ).items()
    }

    if not ke_probability:
        ke_probability = {
            "equities": 0.7,
            "structured_prods": 0.4,
            "derivatives": 0.2,
            "fixed_income": 0.8,
            "alternatives": 0.2,
        }

    seed_raw = raw_config.get("seed", None)
    seed = int(seed_raw) if seed_raw is not None else None

    age_epsilon_cfg = clients.get("age_epsilon", {})

    return GenerationConfig(
        num_advisors=int(sizes.get("num_advisors", 100)),
        num_clients=int(sizes.get("num_clients", 5000)),
        num_trades=int(sizes.get("num_trades", 100000)),
        bad_apple_rate=float(conduct.get("bad_apple_rate", 0.05)),
        typology_weights={k.title(): v for k, v in conduct.get("typology_weights", {}).items()},
        churning_cluster_min_trades=int(conduct.get("churning_cluster_min_trades", 3)),
        churning_cluster_max_trades=int(conduct.get("churning_cluster_max_trades", 5)),
        churning_window_days=int(conduct.get("churning_window_days", 30)),
        age_min=int(age_cfg.get("min", 25)),
        age_max=int(age_cfg.get("max", 90)),
        aum_min=float(aum_cfg.get("min", 1e5)),
        aum_max=float(aum_cfg.get("max", 1e7)),
        trade_date_horizon_days=int(trades_cfg.get("trade_date_horizon_days", 365)),
        risk_tolerance_distribution=risk_tolerance_distribution,
        liquidity_needs_distribution=liquidity_needs_distribution,
        investment_goal_distribution=investment_goal_distribution,
        ke_probability=ke_probability,
        age_risk_epsilon=float(age_epsilon_cfg.get("risk_tolerance", 0.05)),
        age_goal_epsilon=float(age_epsilon_cfg.get("investment_goal", 0.05)),
        seed=seed,
    )


def generate_advisors(cfg: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    advisor_ids = [f"ADV_{i:04d}" for i in range(cfg.num_advisors)]

    num_bad = max(1, int(cfg.num_advisors * cfg.bad_apple_rate))
    bad_indices = set(
        rng.choice(cfg.num_advisors, size=num_bad, replace=False).tolist()
    )

    typology_categories = list(cfg.typology_weights.keys())
    typology_probs = list(cfg.typology_weights.values())
    if typology_categories:
        typology_probs = np.array(typology_probs, dtype=float)
        typology_probs = typology_probs / typology_probs.sum()

    conduct_flags: List[bool] = []
    risk_typologies: List[str] = []

    for idx in range(cfg.num_advisors):
        is_bad = idx in bad_indices
        conduct_flags.append(is_bad)
        if is_bad and typology_categories:
            risk_typology = rng.choice(typology_categories, p=typology_probs)
        else:
            risk_typology = "Clean"
        risk_typologies.append(risk_typology)

    advisors = pd.DataFrame(
        {
            "advisor_id": advisor_ids,
            "conduct_flag": conduct_flags,
            "risk_typology": risk_typologies,
        }
    )
    return advisors


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    total = float(np.sum(probs))
    if total <= 0:
        raise ValueError("Probability vector must sum to a positive value.")
    return probs / total


def _sample_from_distribution(
    *,
    choices: np.ndarray,
    probabilities: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    probs = _normalize_probs(probabilities)
    return rng.choice(choices, size=size, p=probs)


def _sample_risk_tolerance(
    size: int, cfg: GenerationConfig, rng: np.random.Generator
) -> np.ndarray:
    choices = np.array([1, 2, 3, 4, 5], dtype=int)
    probs = np.array([cfg.risk_tolerance_distribution.get(i, 0.0) for i in choices])
    return _sample_from_distribution(
        choices=choices, probabilities=probs, size=size, rng=rng
    )


def _sample_liquidity_needs(
    size: int, cfg: GenerationConfig, rng: np.random.Generator
) -> np.ndarray:
    choices = np.array(["high", "medium", "low"])
    probs = np.array(
        [cfg.liquidity_needs_distribution.get(str(k), 0.0) for k in choices]
    )
    return _sample_from_distribution(
        choices=choices, probabilities=probs, size=size, rng=rng
    )


def _sample_investment_goal_by_age(
    ages: np.ndarray, cfg: GenerationConfig, rng: np.random.Generator
) -> np.ndarray:
    choices = np.array(["capital preservation", "income", "growth", "speculation"])
    base = np.array(
        [cfg.investment_goal_distribution.get(k, 0.0) for k in choices], dtype=float
    )
    base /= base.sum()

    is_exception = rng.random(len(ages)) < cfg.age_goal_epsilon

    results = np.empty(len(ages), dtype=object)
    for i, age in enumerate(ages):
        if is_exception[i]:
            results[i] = rng.choice(choices, p=base)
        else:
            age_factor = np.clip((age - 40) / 40, 0.0, 1.0)
            probs = base.copy()
            probs[0] += age_factor * 0.4  # capital preservation ↑
            probs[1] += age_factor * 0.2  # income ↑
            probs[2] -= age_factor * 0.3  # growth ↓
            probs[3] -= age_factor * 0.3  # speculation ↓
            probs = np.clip(probs, 0.0, None)
            probs /= probs.sum()
            results[i] = rng.choice(choices, p=probs)

    return results


def generate_clients(
    advisors: pd.DataFrame, cfg: GenerationConfig, rng: np.random.Generator
) -> pd.DataFrame:
    client_ids = [f"CLT_{i:06d}" for i in range(cfg.num_clients)]

    advisor_choices = advisors["advisor_id"].to_numpy()
    advisor_assignments = rng.choice(
        advisor_choices, size=cfg.num_clients, replace=True
    )

    # Fat-tailed age distribution (t, df=4) centred at 50.
    # Rejection-sample so out-of-range draws are replaced rather than clipped —
    # clipping would pile mass onto the boundary values (25 and 90).
    ages = np.empty(cfg.num_clients, dtype=int)
    unfilled = np.ones(cfg.num_clients, dtype=bool)
    while unfilled.any():
        n = int(unfilled.sum())
        candidates = np.round(50 + rng.standard_t(df=4, size=n) * 12).astype(int)
        in_range = (candidates >= cfg.age_min) & (candidates <= cfg.age_max)
        ages[unfilled] = np.where(in_range, candidates, 0)
        unfilled[unfilled] = ~in_range

    risk_tolerance = _sample_risk_tolerance(cfg.num_clients, cfg, rng)

    # Older clients are more conservative: cap risk_tolerance by age bracket.
    # Epsilon fraction of clients are exceptions and keep their sampled value.
    age_risk_cap = np.where(
        ages > 80,
        1,
        np.where(
            ages > 75,
            2,
            np.where(ages > 70, 3, np.where(ages > 65, 4, np.where(ages > 55, 4, 5))),
        ),
    )
    is_rt_exception = rng.random(cfg.num_clients) < cfg.age_risk_epsilon
    risk_tolerance = np.where(
        is_rt_exception, risk_tolerance, np.minimum(risk_tolerance, age_risk_cap)
    )

    # Log-normal-ish AUM distribution between aum_min and aum_max
    log_aum = rng.normal(loc=12, scale=1.0, size=cfg.num_clients)
    aum_raw = np.exp(log_aum)
    aum_scaled = cfg.aum_min + (
        (aum_raw - aum_raw.min())
        / (aum_raw.max() - aum_raw.min())
        * (cfg.aum_max - cfg.aum_min)
    )

    liquidity_needs = _sample_liquidity_needs(cfg.num_clients, cfg, rng)
    investment_goal = _sample_investment_goal_by_age(ages, cfg, rng)

    ke_cfg = cfg.ke_probability
    ke_equities = rng.random(cfg.num_clients) < ke_cfg.get("equities", 0.0)
    ke_structured = rng.random(cfg.num_clients) < ke_cfg.get("structured_prods", 0.0)
    ke_derivatives = rng.random(cfg.num_clients) < ke_cfg.get("derivatives", 0.0)
    ke_fixed_income = rng.random(cfg.num_clients) < ke_cfg.get("fixed_income", 0.0)
    ke_alternatives = rng.random(cfg.num_clients) < ke_cfg.get("alternatives", 0.0)

    clients = pd.DataFrame(
        {
            "client_id": client_ids,
            "advisor_id": advisor_assignments,
            "age": ages,
            "risk_tolerance": risk_tolerance,
            "aum_usd": aum_scaled,
            "ke_equities": ke_equities,
            "ke_structured_prods": ke_structured,
            "ke_derivatives": ke_derivatives,
            "ke_fixed_income": ke_fixed_income,
            "ke_alternatives": ke_alternatives,
            "liquidity_needs": liquidity_needs,
            "investment_goal": investment_goal,
        }
    )
    return clients


def generate_products(cfg: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    product_types = [
        "Equity",
        "Fixed Income",
        "Structured",
        "Derivative",
        "Alternative",
    ]

    risk_map = {
        "Equity": 3,
        "Fixed Income": 2,
        "Structured": 4,
        "Derivative": 5,
        "Alternative": 5,
    }

    # Normal distribution params: (mean, std, lo_clip, hi_clip)
    tenor_params = {
        "Equity":       (1.00, 0.50, 0.25,  3.0),
        "Fixed Income": (3.00, 1.50, 0.50,  7.0),
        "Structured":   (3.00, 0.75, 1.00,  5.0),
        "Derivative":   (0.50, 0.20, 0.08,  1.5),
        "Alternative":  (5.00, 2.00, 1.00, 10.0),
    }
    commission_params = {
        "Equity":       (0.0050, 0.0010, 0.0010, 0.0100),
        "Fixed Income": (0.0075, 0.0020, 0.0020, 0.0150),
        "Structured":   (0.0200, 0.0050, 0.0050, 0.0400),
        "Derivative":   (0.0150, 0.0040, 0.0030, 0.0300),
        "Alternative":  (0.0200, 0.0050, 0.0050, 0.0400),
    }
    # Lockup in days; floats drawn then rounded to int
    lockup_params = {
        "Equity":       (  3,   4,    0,   15),
        "Fixed Income": ( 30,  15,    0,   90),
        "Structured":   (365,  90,  180,  730),
        "Derivative":   (  2,   3,    0,   14),
        "Alternative":  (730, 180,  180, 1825),
    }

    num_products_per_type = 50
    rows: List[Dict] = []

    pid = 0
    for ptype in product_types:
        t_mean, t_std, t_lo, t_hi = tenor_params[ptype]
        c_mean, c_std, c_lo, c_hi = commission_params[ptype]
        l_mean, l_std, l_lo, l_hi = lockup_params[ptype]

        tenors = np.clip(
            rng.normal(t_mean, t_std, size=num_products_per_type), t_lo, t_hi
        )
        commissions = np.clip(
            rng.normal(c_mean, c_std, size=num_products_per_type), c_lo, c_hi
        )
        lockups = np.round(
            np.clip(rng.normal(l_mean, l_std, size=num_products_per_type), l_lo, l_hi)
        ).astype(int)

        for i in range(num_products_per_type):
            pid += 1
            product_id = f"PRD_{pid:05d}"
            actual_risk = risk_map[ptype]

            rows.append(
                {
                    "product_id": product_id,
                    "product_type": ptype,
                    "is_structured": ptype
                    in ("Structured", "Derivative", "Alternative"),
                    "actual_risk_rating": actual_risk,
                    "recommended_tenor_yrs": round(float(tenors[i]), 2),
                    "commission_rate": round(float(commissions[i]), 6),
                    "lockup_days": int(lockups[i]),
                }
            )

    product_df = pd.DataFrame(rows)

    # Introduce randomness to product risk
    product_risk_variation_vector = rng.choice(
        [0, 1, 2], size=len(product_df), p=[0.6, 0.3, 0.1]
    )

    product_df["actual_risk_rating"] = (
        product_df["actual_risk_rating"] - product_risk_variation_vector
    ).clip(1, 5)

    return product_df


def _pick_product_for_client_suitable(
    client_row: pd.Series,
    products: pd.DataFrame,
    rng: np.random.Generator,
    breach_prob: float = 0.15,
) -> pd.Series:
    """
    # Pick a product as a responsible advisor would—generally aligned with the client's profile,
    # thoughtfully considering the client's circumstances. If suitability hard rules are breached
    # (K&E, asset concentration, risk, horizon), it is done with plausible justification documented
    # (e.g., client-specific rationale, comprehensive disclosure, or exceptional needs).
    """
    # Map product types to K&E fields
    ke_fields = {
        "Equity": "ke_equities",
        "Fixed Income": "ke_fixed_income",
        "Structured": "ke_structured_prods",
        "Derivative": "ke_derivatives",
        "Alternative": "ke_alternatives",
    }

    # --- Step 1: Candidate products, initially require K&E ---
    eligible_types = []
    for ptype, ke_field in ke_fields.items():
        if client_row.get(ke_field, False):
            eligible_types.append(ptype)

    # With a small chance, allow products outside K&E (breach K&E)
    if not eligible_types or rng.random() < breach_prob:
        eligible_types = list(ke_fields.keys())

    subset = products[products["product_type"].isin(eligible_types)]
    if subset.empty:
        subset = products

    # --- Step 2: Filter for no risk/horizon/asset breach (but allow chance of breach) ---

    # Risk filter: product actual risk <= client risk tolerance, unless breaching
    risk_filtered = subset[
        subset["actual_risk_rating"] <= client_row.get("risk_tolerance", 3)
    ]
    if risk_filtered.empty or rng.random() < breach_prob:
        risk_filtered = subset  # allow breach

    # --- Step 3: Pick randomly among filtered candidates ---
    final_subset = risk_filtered if not risk_filtered.empty else subset

    return final_subset.sample(1, random_state=rng).iloc[0]


def _pick_product_for_typology(
    typology: str,
    client_row: pd.Series,
    products: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.Series, str]:
    """Returns (product_row, effective_typology).

    For Unsuitable Recs and Misrepresentation, if the selected product's
    actual_risk_rating does not exceed the client's risk_tolerance, the
    trade is not genuinely harmful and effective_typology is downgraded to
    "Clean".
    """
    if typology == "Unsuitable Recs":
        # High-risk / structured products for elderly or conservative clients
        subset = products[
            (products["actual_risk_rating"] >= 5)
            & (
                products["product_type"].isin(
                    ["Structured", "Derivative", "Alternative"]
                )
            )
        ]
        if subset.empty:
            subset = products[products["actual_risk_rating"] >= 5]
        if subset.empty:
            subset = products
        product_row = subset.sample(1, random_state=rng).iloc[0]
        actual_risk = int(product_row["actual_risk_rating"])
        client_tolerance = int(client_row.get("risk_tolerance", 3))
        effective_typology = "Clean" if actual_risk <= client_tolerance else typology
        return product_row, effective_typology

    if typology == "Churning":
        # Select products with high commission rates
        high_commission = products["commission_rate"].quantile(0.75)
        subset = products[
            (products["product_type"].isin(["Structured", "Alternative"]))
            & (products["commission_rate"] >= high_commission)
        ]
        if subset.empty:
            subset = products[
                products["product_type"].isin(["Structured", "Alternative"])
            ]
        if subset.empty:
            subset = products[products["commission_rate"] >= high_commission]
        if subset.empty:
            subset = products
        return subset.sample(1, random_state=rng).iloc[0], typology

    if typology == "Misrepresentation":
        subset = products[products["actual_risk_rating"] >= 4]
        if subset.empty:
            subset = products
        product_row = subset.sample(1, random_state=rng).iloc[0]
        actual_risk = int(product_row["actual_risk_rating"])
        client_tolerance = int(client_row.get("risk_tolerance", 3))
        effective_typology = "Clean" if actual_risk <= client_tolerance else typology
        return product_row, effective_typology

    if typology == "Liquidity Mismatch":
        subset = products[products["lockup_days"] >= 365]
        if subset.empty:
            subset = products
        product_row = subset.sample(1, random_state=rng).iloc[0]
        liquidity_threshold_map = {"high": 90, "medium": 365, "low": 730}
        threshold = liquidity_threshold_map.get(
            str(client_row.get("liquidity_needs", "medium")).lower(), 365
        )
        effective_typology = (
            "Clean" if int(product_row["lockup_days"]) <= threshold else typology
        )
        return product_row, effective_typology

    # Fallback to clean logic
    return _pick_product_for_client_suitable(client_row, products, rng), "Clean"


def _generate_churning_cluster(
    client_id: str,
    advisor_id: str,
    client_row: pd.Series,
    churning_candidates: pd.DataFrame,
    cfg: GenerationConfig,
    rng: np.random.Generator,
    trade_idx_start: int,
    max_cluster_size: int,
) -> List[Dict]:
    cluster_size = int(
        rng.integers(
            cfg.churning_cluster_min_trades,
            cfg.churning_cluster_max_trades + 1,
        )
    )
    cluster_size = min(cluster_size, max_cluster_size)

    horizon = max(1, cfg.trade_date_horizon_days)
    base_day = int(rng.integers(0, max(1, horizon - cfg.churning_window_days)))
    offsets = np.sort(
        rng.integers(
            base_day,
            min(horizon, base_day + cfg.churning_window_days),
            size=cluster_size,
        )
    )

    rows: List[Dict] = []
    for i, offset in enumerate(offsets):
        trade_id = f"TRD_{trade_idx_start + i:07d}"
        product_row = churning_candidates.sample(1, random_state=rng).iloc[0]
        frac = rng.uniform(0.05, 0.2)
        trade_amount = float(client_row["aum_usd"] * frac)
        estimated_comm = trade_amount * float(product_row["commission_rate"])
        actual_risk = int(product_row["actual_risk_rating"])
        rows.append(
            {
                "trade_id": trade_id,
                "client_id": client_id,
                "advisor_id": advisor_id,
                "product_id": product_row["product_id"],
                "trade_date_offset_days": int(offset),
                "trade_amount": trade_amount,
                "solicitation_type": "Solicited",
                "estimated_comm": estimated_comm,
                "typology_applied": "Churning",
                "actual_risk_rating": actual_risk,
                "marketed_risk_rating": actual_risk,
            }
        )
    return rows


def generate_trades(
    advisors: pd.DataFrame,
    clients: pd.DataFrame,
    products: pd.DataFrame,
    cfg: GenerationConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    # Precompute advisor conduct info
    advisor_info = advisors.set_index("advisor_id")

    client_ids = clients["client_id"].to_numpy()
    advisor_ids = clients["advisor_id"].to_numpy()

    # Precompute churning product candidates (high-commission structured/alternative)
    high_commission = products["commission_rate"].quantile(0.75)
    churning_candidates = products[
        (products["product_type"].isin(["Structured", "Alternative"]))
        & (products["commission_rate"] >= high_commission)
    ]
    if churning_candidates.empty:
        churning_candidates = products[
            products["product_type"].isin(["Structured", "Alternative"])
        ]
    if churning_candidates.empty:
        churning_candidates = products

    trade_rows: List[Dict] = []

    # Simple helper to get client row quickly
    clients_indexed = clients.set_index("client_id")

    # Generate trades, inserting clusters for Churning advisors
    while len(trade_rows) < cfg.num_trades:
        remaining = cfg.num_trades - len(trade_rows)

        # Choose a client uniformly
        c_idx = rng.integers(0, len(client_ids))
        client_id = client_ids[c_idx]
        advisor_id = advisor_ids[c_idx]
        client_row = clients_indexed.loc[client_id]
        advisor_row = advisor_info.loc[advisor_id]

        conduct_flag = bool(advisor_row["conduct_flag"])
        typology = advisor_row["risk_typology"]

        # Handle churning clusters
        if (
            conduct_flag
            and typology == "Churning"
            and remaining >= cfg.churning_cluster_min_trades
        ):
            cluster = _generate_churning_cluster(
                client_id=client_id,
                advisor_id=advisor_id,
                client_row=client_row,
                churning_candidates=churning_candidates,
                cfg=cfg,
                rng=rng,
                trade_idx_start=len(trade_rows),
                max_cluster_size=remaining,
            )
            trade_rows.extend(cluster)
            continue

        # Non-churning: pick product and typology
        if conduct_flag and typology != "Clean":
            product_row, typology_applied = _pick_product_for_typology(
                typology, client_row, products, rng
            )
        else:
            typology_applied = "Clean"
            product_row = _pick_product_for_client_suitable(client_row, products, rng)

        trade_idx = len(trade_rows)
        trade_id = f"TRD_{trade_idx:07d}"

        # Trade amount as a fraction of client AUM
        frac = rng.uniform(0.01, 0.25)
        trade_amount = float(client_row["aum_usd"] * frac)

        estimated_comm = trade_amount * float(product_row["commission_rate"])

        # marketed_risk_rating is a trade-level attribute: defaults to actual, lowered for Misrepresentation
        actual_risk = int(product_row["actual_risk_rating"])
        marketed_risk = actual_risk
        if typology_applied == "Misrepresentation":
            marketed_risk = max(1, actual_risk - int(rng.integers(1, 3)))

        trade_rows.append(
            {
                "trade_id": trade_id,
                "client_id": client_id,
                "advisor_id": advisor_id,
                "product_id": product_row["product_id"],
                "trade_date_offset_days": int(
                    rng.integers(0, max(1, cfg.trade_date_horizon_days))
                ),
                "trade_amount": trade_amount,
                "solicitation_type": "Unsolicited",
                "estimated_comm": estimated_comm,
                "typology_applied": typology_applied,
                "actual_risk_rating": actual_risk,
                "marketed_risk_rating": marketed_risk,
            }
        )

    trades = pd.DataFrame(trade_rows)

    # Vectorized: about half of "Clean" trades are solicited, ~90% of bad-apple trades are solicited
    p_solicited_clean = 0.5
    p_solicited_bad = 0.9

    mask_clean = trades["typology_applied"] == "Clean"
    rng_vals = rng.random(len(trades))

    trades["solicitation_type"] = np.where(
        mask_clean,
        np.where(rng_vals < p_solicited_clean, "Solicited", "Unsolicited"),
        np.where(rng_vals < p_solicited_bad, "Solicited", "Unsolicited"),
    )
    return trades
