"""Tests for the hard gate.

The gate is the one component that must never be wrong in the permissive
direction, so the bulk of these cases are rejections. Each one names the single
limit it violates, and the last case checks that a proposal breaking several
rules reports all of them rather than the first.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from onenode.risk.gate import evaluate
from onenode.risk.limits import DEFAULT_LIMITS
from onenode.risk.models import (
    AccountSnapshot,
    MarketClock,
    OptionLeg,
    ProposedTrade,
    Side,
)

TODAY = date(2026, 8, 31)
EXPIRY_CODE = "260904"  # 4 days out


def leg(strike: int, right: str, side: Side, ratio: int = 1) -> OptionLeg:
    return OptionLeg.from_symbol(
        f"SPY{EXPIRY_CODE}{right}{strike * 1000:08d}", side=side, ratio_qty=ratio
    )


BULL_PUT_SPREAD = (leg(450, "P", Side.SELL), leg(445, "P", Side.BUY))


@pytest.fixture
def trade() -> ProposedTrade:
    """A well-behaved one-lot bull put spread: $400 risk against $100 credit."""
    return ProposedTrade(
        underlying="SPY",
        legs=BULL_PUT_SPREAD,
        contracts=1,
        net_cash=100.0,
        quote_age_seconds=5.0,
        worst_leg_spread_pct=2.0,
        rationale="IV rank elevated, price above the 20-day mean",
    )


@pytest.fixture
def account() -> AccountSnapshot:
    return AccountSnapshot(
        equity=100_000.0,
        last_equity=100_000.0,
        open_positions=0,
        committed_risk=0.0,
    )


@pytest.fixture
def clock() -> MarketClock:
    return MarketClock(is_open=True, minutes_to_close=180.0)


def check(trade, account, clock, limits=DEFAULT_LIMITS):
    return evaluate(trade, account, clock, today=TODAY, limits=limits)


class TestApproval:
    def test_a_sound_trade_is_approved(self, trade, account, clock):
        decision = check(trade, account, clock)
        assert decision.approved, decision.violations
        assert decision.violations == ()
        assert decision.worst_case_loss == pytest.approx(400.0)

    def test_approval_reports_the_risk_it_computed(self, trade, account, clock):
        decision = check(replace(trade, contracts=3, net_cash=300.0), account, clock)
        assert decision.approved
        assert decision.worst_case_loss == pytest.approx(1200.0)


class TestSessionGates:
    def test_closed_market_is_rejected(self, trade, account, clock):
        decision = check(trade, account, replace(clock, is_open=False))
        assert not decision.approved
        assert any("closed" in v for v in decision.violations)

    def test_last_half_hour_is_rejected(self, trade, account, clock):
        decision = check(trade, account, replace(clock, minutes_to_close=20.0))
        assert not decision.approved
        assert any("close to the bell" in v for v in decision.violations)

    def test_exactly_at_the_threshold_is_allowed(self, trade, account, clock):
        decision = check(trade, account, replace(clock, minutes_to_close=30.0))
        assert decision.approved, decision.violations


class TestDailyStop:
    def test_breaching_the_daily_stop_halts_trading(self, trade, account, clock):
        # Down 3.1% on the day.
        hurt = replace(account, equity=96_900.0)
        decision = check(trade, hurt, clock)
        assert not decision.approved
        assert any("daily stop" in v for v in decision.violations)

    def test_a_losing_but_survivable_day_still_trades(self, trade, account, clock):
        decision = check(trade, replace(account, equity=98_000.0), clock)
        assert decision.approved, decision.violations

    def test_the_stop_is_inclusive(self, trade, account, clock):
        # Exactly -3.00% must halt, not squeak through.
        decision = check(trade, replace(account, equity=97_000.0), clock)
        assert not decision.approved
        assert any("daily stop" in v for v in decision.violations)


class TestInstrumentGates:
    def test_underlying_outside_the_allowlist_is_rejected(self, trade, account, clock):
        decision = check(replace(trade, underlying="TSLA"), account, clock)
        assert not decision.approved
        assert any("allowlist" in v for v in decision.violations)

    def test_expiry_beyond_the_dte_window_is_rejected(self, trade, account, clock):
        far = (
            OptionLeg.from_symbol("SPY261218P00450000", side=Side.SELL),
            OptionLeg.from_symbol("SPY261218P00445000", side=Side.BUY),
        )
        decision = check(replace(trade, legs=far), account, clock)
        assert not decision.approved
        assert any("DTE" in v for v in decision.violations)

    def test_calendar_spread_is_rejected(self, trade, account, clock):
        mixed = (
            OptionLeg.from_symbol("SPY260904P00450000", side=Side.SELL),
            OptionLeg.from_symbol("SPY260902P00445000", side=Side.BUY),
        )
        decision = check(replace(trade, legs=mixed), account, clock)
        assert not decision.approved
        assert any("distinct expiries" in v for v in decision.violations)


class TestQuoteQuality:
    def test_stale_quote_is_rejected(self, trade, account, clock):
        decision = check(replace(trade, quote_age_seconds=600.0), account, clock)
        assert not decision.approved
        assert any("stale quote" in v for v in decision.violations)

    def test_wide_market_is_rejected(self, trade, account, clock):
        decision = check(replace(trade, worst_leg_spread_pct=25.0), account, clock)
        assert not decision.approved
        assert any("wide market" in v for v in decision.violations)


class TestRiskGates:
    def test_undefined_risk_is_rejected_outright(self, trade, account, clock):
        naked = (leg(460, "C", Side.SELL),)
        decision = check(replace(trade, legs=naked, net_cash=200.0), account, clock)
        assert not decision.approved
        assert decision.worst_case_loss is None
        assert any("undefined risk" in v for v in decision.violations)

    def test_oversized_trade_is_rejected(self, trade, account, clock):
        # 5 lots is $2,000 of risk against a $1,500 per-trade budget.
        decision = check(replace(trade, contracts=5, net_cash=500.0), account, clock)
        assert not decision.approved
        assert any("per-trade budget" in v for v in decision.violations)

    def test_portfolio_ceiling_blocks_an_otherwise_fine_trade(self, trade, account, clock):
        loaded = replace(account, committed_risk=5_800.0, open_positions=4)
        decision = check(trade, loaded, clock)
        assert not decision.approved
        assert any("portfolio risk" in v for v in decision.violations)

    def test_position_count_ceiling(self, trade, account, clock):
        crowded = replace(account, open_positions=5, committed_risk=1_000.0)
        decision = check(trade, crowded, clock)
        assert not decision.approved
        assert any("already open" in v for v in decision.violations)

    def test_risk_budget_follows_equity_down(self, trade, account, clock):
        # On a $20k account, 3 lots of $400 risk is over the 1.5% budget.
        small = replace(account, equity=20_000.0, last_equity=20_000.0)
        decision = check(replace(trade, contracts=3, net_cash=300.0), small, clock)
        assert not decision.approved
        assert any("per-trade budget" in v for v in decision.violations)


class TestTradeQuality:
    def test_premium_too_thin_is_rejected(self, trade, account, clock):
        # $10 of credit against $490 of risk: a 2% reward-to-risk ratio.
        decision = check(replace(trade, net_cash=10.0), account, clock)
        assert not decision.approved
        assert any("reward-to-risk" in v for v in decision.violations)


class TestReporting:
    def test_all_violations_are_reported_together(self, account, clock):
        awful = ProposedTrade(
            underlying="TSLA",
            legs=BULL_PUT_SPREAD,
            contracts=50,
            net_cash=100.0,
            quote_age_seconds=900.0,
            worst_leg_spread_pct=40.0,
        )
        decision = check(awful, replace(account, equity=96_000.0), clock)
        assert not decision.approved
        joined = " | ".join(decision.violations)
        for expected in [
            "daily stop",
            "allowlist",
            "contracts exceeds",
            "stale quote",
            "wide market",
            "per-trade budget",
        ]:
            assert expected in joined, f"missing {expected!r} in {joined}"

    def test_decision_renders_readably(self, trade, account, clock):
        assert "APPROVED" in str(check(trade, account, clock))
        blocked = check(replace(trade, underlying="TSLA"), account, clock)
        assert str(blocked).startswith("BLOCKED:")
