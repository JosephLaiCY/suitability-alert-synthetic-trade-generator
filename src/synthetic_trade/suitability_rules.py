from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def apply_suitability_rules(
    trades: pd.DataFrame,
    clients: pd.DataFrame,
    products: pd.DataFrame,
    rules_cfg: Dict,
) -> pd.DataFrame:
    """
    Apply mechanical (hard) suitability rules and return a dataframe with:
    - per-rule flags
    - is_alert (any rule breached)
    """
    asset_concentration_threshold = float(
        rules_cfg.get("asset_concentration_threshold", 0.15)
    )
    liquidity_lockup_thresholds = rules_cfg.get(
        "liquidity_mismatch_lockup_threshold_days",
        {"high": 90, "medium": 365, "low": 730},
    )

    trades_enriched = trades.merge(
        clients[
            [
                "client_id",
                "aum_usd",
                "risk_tolerance",
                "liquidity_needs",
                "ke_equities",
                "ke_fixed_income",
                "ke_structured_prods",
                "ke_derivatives",
                "ke_alternatives",
            ]
        ],
        on="client_id",
        how="left",
        validate="many_to_one",
    ).merge(
        products[
            [
                "product_id",
                "product_type",
                "lockup_days",
            ]
        ],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    # K&E mapping by product_type (vectorized)
    pt = trades_enriched["product_type"]
    ke_allowed = pd.Series(True, index=trades_enriched.index, dtype=bool)

    mask = pt == "Equity"
    ke_allowed.loc[mask] = trades_enriched.loc[mask, "ke_equities"].astype(bool)

    mask = pt == "Fixed Income"
    ke_allowed.loc[mask] = trades_enriched.loc[mask, "ke_fixed_income"].astype(bool)

    mask = pt == "Structured"
    ke_allowed.loc[mask] = trades_enriched.loc[mask, "ke_structured_prods"].astype(bool)

    mask = pt == "Derivative"
    ke_allowed.loc[mask] = trades_enriched.loc[mask, "ke_derivatives"].astype(bool)

    mask = pt == "Alternative"
    ke_allowed.loc[mask] = trades_enriched.loc[mask, "ke_alternatives"].astype(bool)

    flag_ke_mismatch = ~ke_allowed

    # Asset concentration
    flag_asset_concentration = trades_enriched["trade_amount"] > (
        asset_concentration_threshold * trades_enriched["aum_usd"]
    )

    # Risk mismatch
    flag_risk_mismatch = (
        trades_enriched["actual_risk_rating"] > trades_enriched["risk_tolerance"]
    )

    # Liquidity mismatch: product lockup exceeds threshold for client's liquidity_needs
    lockup_threshold = trades_enriched["liquidity_needs"].map(
        liquidity_lockup_thresholds
    )
    flag_liquidity_mismatch = trades_enriched["lockup_days"] > lockup_threshold

    out = trades.copy()
    out["flag_ke_mismatch"] = flag_ke_mismatch.to_numpy()
    out["flag_asset_concentration"] = flag_asset_concentration.to_numpy()
    out["flag_risk_mismatch"] = flag_risk_mismatch.to_numpy()
    out["flag_liquidity_mismatch"] = flag_liquidity_mismatch.to_numpy()

    out["is_alert"] = out[
        [
            "flag_ke_mismatch",
            "flag_asset_concentration",
            "flag_risk_mismatch",
            "flag_liquidity_mismatch",
        ]
    ].any(axis=1)

    # is_genuine: alert was caused by a bad-apple advisor acting on their conduct typology
    out["is_genuine"] = out["is_alert"] & (out["typology_applied"] != "Clean")

    return out
