import pandas as pd
import numpy as np
import pytest

from src.synthetic_trade.suitability_rules import apply_suitability_rules


# ---------------------------------------------------------------------------
# Minimal fixture builders
# ---------------------------------------------------------------------------

def _make_client(
    client_id="CLT_000001",
    aum_usd=1_000_000,
    risk_tolerance=3,
    investment_horizon="Medium",
    liquidity_needs="medium",
    ke_equities=True,
    ke_fixed_income=True,
    ke_structured_prods=False,
    ke_derivatives=False,
    ke_alternatives=False,
):
    return pd.DataFrame([{
        "client_id": client_id,
        "aum_usd": aum_usd,
        "risk_tolerance": risk_tolerance,
        "investment_horizon": investment_horizon,
        "liquidity_needs": liquidity_needs,
        "ke_equities": ke_equities,
        "ke_fixed_income": ke_fixed_income,
        "ke_structured_prods": ke_structured_prods,
        "ke_derivatives": ke_derivatives,
        "ke_alternatives": ke_alternatives,
    }])


def _make_product(
    product_id="PRD_00001",
    product_type="Equity",
    actual_risk_rating=3,
    recommended_tenor_yrs=1.0,
    lockup_days=0,
):
    return pd.DataFrame([{
        "product_id": product_id,
        "product_type": product_type,
        "actual_risk_rating": actual_risk_rating,
        "recommended_tenor_yrs": recommended_tenor_yrs,
        "lockup_days": lockup_days,
    }])


def _make_trade(
    trade_id="TRD_0000001",
    client_id="CLT_000001",
    advisor_id="ADV_0001",
    product_id="PRD_00001",
    trade_amount=100_000,
    typology_applied="Clean",
    actual_risk_rating=3,
    marketed_risk_rating=3,
):
    return pd.DataFrame([{
        "trade_id": trade_id,
        "client_id": client_id,
        "advisor_id": advisor_id,
        "product_id": product_id,
        "trade_date_offset_days": 0,
        "trade_amount": trade_amount,
        "solicitation_type": "Unsolicited",
        "estimated_comm": 500.0,
        "typology_applied": typology_applied,
        "actual_risk_rating": actual_risk_rating,
        "marketed_risk_rating": marketed_risk_rating,
    }])


RULES_CFG = {
    "asset_concentration_threshold": 0.15,
    "investment_horizon_bounds_yrs": {"Short": 1, "Medium": 5, "Long": 10},
}


# ---------------------------------------------------------------------------
# flag_ke_mismatch
# ---------------------------------------------------------------------------

class TestKEMismatch:
    def test_no_flag_when_client_has_ke(self):
        client = _make_client(ke_equities=True)
        product = _make_product(product_type="Equity")
        trade = _make_trade()
        result = apply_suitability_rules(trade, client, product, RULES_CFG)
        assert result["flag_ke_mismatch"].iloc[0] is np.bool_(False)

    def test_flag_when_client_lacks_ke(self):
        client = _make_client(ke_equities=False)
        product = _make_product(product_type="Equity")
        trade = _make_trade()
        result = apply_suitability_rules(trade, client, product, RULES_CFG)
        assert result["flag_ke_mismatch"].iloc[0] is np.bool_(True)

    def test_structured_product_uses_ke_structured_prods(self):
        client = _make_client(ke_structured_prods=False)
        product = _make_product(product_type="Structured")
        trade = _make_trade()
        result = apply_suitability_rules(trade, client, product, RULES_CFG)
        assert result["flag_ke_mismatch"].iloc[0] is np.bool_(True)


# ---------------------------------------------------------------------------
# flag_asset_concentration
# ---------------------------------------------------------------------------

class TestAssetConcentration:
    def test_no_flag_below_threshold(self):
        # 10% of AUM — below 15% threshold
        client = _make_client(aum_usd=1_000_000)
        product = _make_product()
        trade = _make_trade(trade_amount=100_000)
        result = apply_suitability_rules(trade, client, product, RULES_CFG)
        assert result["flag_asset_concentration"].iloc[0] is np.bool_(False)

    def test_flag_above_threshold(self):
        # 20% of AUM — above 15% threshold
        client = _make_client(aum_usd=1_000_000)
        product = _make_product()
        trade = _make_trade(trade_amount=200_000)
        result = apply_suitability_rules(trade, client, product, RULES_CFG)
        assert result["flag_asset_concentration"].iloc[0] is np.bool_(True)

    def test_exactly_at_threshold_is_not_flagged(self):
        # strictly greater than, so exactly 15% should not flag
        client = _make_client(aum_usd=1_000_000)
        product = _make_product()
        trade = _make_trade(trade_amount=150_000)
        result = apply_suitability_rules(trade, client, product, RULES_CFG)
        assert result["flag_asset_concentration"].iloc[0] is np.bool_(False)


# ---------------------------------------------------------------------------
# flag_risk_mismatch
# ---------------------------------------------------------------------------

class TestRiskMismatch:
    def test_no_flag_when_product_risk_equals_tolerance(self):
        client = _make_client(risk_tolerance=3)
        product = _make_product(actual_risk_rating=3)
        trade = _make_trade(actual_risk_rating=3)
        result = apply_suitability_rules(trade, client, product, RULES_CFG)
        assert result["flag_risk_mismatch"].iloc[0] is np.bool_(False)

    def test_flag_when_product_risk_exceeds_tolerance(self):
        client = _make_client(risk_tolerance=2)
        product = _make_product(actual_risk_rating=5)
        trade = _make_trade(actual_risk_rating=5)
        result = apply_suitability_rules(trade, client, product, RULES_CFG)
        assert result["flag_risk_mismatch"].iloc[0] is np.bool_(True)


# ---------------------------------------------------------------------------
# is_alert and is_genuine
# ---------------------------------------------------------------------------

class TestAlertAndGenuine:
    def test_no_alert_when_no_flags(self):
        client = _make_client(risk_tolerance=5, ke_equities=True, aum_usd=1_000_000)
        product = _make_product(actual_risk_rating=3, recommended_tenor_yrs=1.0)
        trade = _make_trade(trade_amount=50_000, actual_risk_rating=3)
        result = apply_suitability_rules(trade, client, product, RULES_CFG)
        assert result["is_alert"].iloc[0] is np.bool_(False)
        assert result["is_genuine"].iloc[0] is np.bool_(False)

    def test_alert_but_not_genuine_for_clean_advisor(self):
        # Risk mismatch triggered, but typology is Clean → justifiable, not genuine
        client = _make_client(risk_tolerance=1)
        product = _make_product(actual_risk_rating=5)
        trade = _make_trade(actual_risk_rating=5, typology_applied="Clean")
        result = apply_suitability_rules(trade, client, product, RULES_CFG)
        assert result["is_alert"].iloc[0] is np.bool_(True)
        assert result["is_genuine"].iloc[0] is np.bool_(False)

    def test_genuine_alert_for_bad_apple_typology(self):
        # Risk mismatch triggered AND typology is a bad-apple conduct type
        client = _make_client(risk_tolerance=1)
        product = _make_product(actual_risk_rating=5)
        trade = _make_trade(actual_risk_rating=5, typology_applied="Unsuitable Recs")
        result = apply_suitability_rules(trade, client, product, RULES_CFG)
        assert result["is_alert"].iloc[0] is np.bool_(True)
        assert result["is_genuine"].iloc[0] is np.bool_(True)

    def test_is_alert_true_if_any_single_flag_set(self):
        # Only KE mismatch fires; all others clean
        client = _make_client(ke_equities=False, risk_tolerance=5, aum_usd=1_000_000)
        product = _make_product(
            product_type="Equity", actual_risk_rating=1, recommended_tenor_yrs=1.0
        )
        trade = _make_trade(trade_amount=10_000, actual_risk_rating=1)
        result = apply_suitability_rules(trade, client, product, RULES_CFG)
        assert result["is_alert"].iloc[0] is np.bool_(True)
        assert result["flag_ke_mismatch"].iloc[0] is np.bool_(True)
        assert result["flag_risk_mismatch"].iloc[0] is np.bool_(False)
        assert result["flag_asset_concentration"].iloc[0] is np.bool_(False)
