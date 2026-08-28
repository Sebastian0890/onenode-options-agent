"""Tests for candidate construction.

The rule these pin down: the agent may only ever consider spreads assembled
from contracts that are really in the chain and really quoted, priced at what a
marketable order would actually collect.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from onenode.broker.cli import ContractQuote
from onenode.risk.models import Right, Side
from onenode.strategy import build_credit_spreads, size_position

TODAY = date(2026, 8, 28)
QUOTED_AT = datetime(2026, 8, 28, 18, 42, tzinfo=UTC)


def quote(
    strike: int,
    right: str,
    bid: float,
    ask: float,
    delta: float,
    expiry: str = "260831",
) -> ContractQuote:
    return ContractQuote(
        symbol=f"SPY{expiry}{right}{strike * 1000:08d}",
        bid=bid,
        ask=ask,
        quoted_at=QUOTED_AT,
        delta=delta,
    )


# A plausible slice of the SPY put chain with spot near 769.
PUT_CHAIN = [
    quote(766, "P", 1.60, 1.64, -0.24),
    quote(764, "P", 1.20, 1.24, -0.17),
    quote(763, "P", 1.02, 1.06, -0.15),
    quote(762, "P", 0.88, 0.92, -0.13),
    quote(759, "P", 0.56, 0.60, -0.09),
    quote(755, "P", 0.30, 0.34, -0.05),
]


class TestPutCreditSpreads:
    def test_builds_a_spread_from_two_real_contracts(self):
        candidates = build_credit_spreads(PUT_CHAIN, underlying="SPY", today=TODAY)
        assert candidates
        keys = {c.key for c in candidates}
        assert "SPY260831P00764000/SPY260831P00759000" in keys

    def test_priced_at_the_fill_not_at_the_mid(self):
        """short bid minus long ask, which is what a marketable order collects."""
        candidates = build_credit_spreads(PUT_CHAIN, underlying="SPY", widths=(5.0,), today=TODAY)
        spread = next(c for c in candidates if c.short_leg.strike == 764)
        # 1.20 bid on the short, 0.60 ask on the long: $60, not the $62 mid-to-mid.
        assert spread.credit_per_contract == pytest.approx(60.0)

    def test_max_loss_is_width_less_credit(self):
        candidates = build_credit_spreads(PUT_CHAIN, underlying="SPY", widths=(5.0,), today=TODAY)
        spread = next(c for c in candidates if c.short_leg.strike == 764)
        assert spread.width == pytest.approx(5.0)
        assert spread.max_loss_per_contract == pytest.approx(440.0)
        assert spread.reward_to_risk == pytest.approx(60.0 / 440.0)

    def test_protection_is_below_the_short_strike_for_puts(self):
        candidates = build_credit_spreads(PUT_CHAIN, underlying="SPY", today=TODAY)
        for candidate in candidates:
            assert candidate.long_leg.strike < candidate.short_leg.strike
            assert candidate.short_leg.side is Side.SELL
            assert candidate.long_leg.side is Side.BUY

    def test_ranked_by_reward_to_risk(self):
        candidates = build_credit_spreads(PUT_CHAIN, underlying="SPY", today=TODAY)
        ratios = [c.reward_to_risk for c in candidates]
        assert ratios == sorted(ratios, reverse=True)


class TestFilters:
    def test_short_strike_must_sit_near_the_target_delta(self):
        candidates = build_credit_spreads(
            PUT_CHAIN, underlying="SPY", target_delta=0.17, delta_tolerance=0.01, today=TODAY
        )
        assert {c.short_leg.strike for c in candidates} == {764.0}

    def test_contracts_without_greeks_are_skipped(self):
        blind = [
            ContractQuote("SPY260831P00764000", 1.20, 1.24, QUOTED_AT, delta=None),
            quote(759, "P", 0.56, 0.60, -0.09),
        ]
        assert build_credit_spreads(blind, underlying="SPY", today=TODAY) == []

    def test_untradeable_contracts_are_skipped(self):
        no_bid = [
            quote(764, "P", 1.20, 1.24, -0.17),
            ContractQuote("SPY260831P00759000", 0.0, 0.60, QUOTED_AT, delta=-0.09),
        ]
        assert build_credit_spreads(no_bid, underlying="SPY", widths=(5.0,), today=TODAY) == []

    def test_wide_markets_are_skipped(self):
        wide = [
            quote(764, "P", 1.00, 2.00, -0.17),  # 66% spread
            quote(759, "P", 0.56, 0.60, -0.09),
        ]
        assert build_credit_spreads(wide, underlying="SPY", widths=(5.0,), today=TODAY) == []

    def test_a_spread_that_collects_nothing_is_not_a_trade(self):
        inverted = [
            quote(764, "P", 0.50, 0.54, -0.17),
            quote(759, "P", 0.56, 0.60, -0.09),  # ask above the short's bid
        ]
        assert build_credit_spreads(inverted, underlying="SPY", widths=(5.0,), today=TODAY) == []

    def test_premium_too_thin_against_the_width_is_rejected(self):
        thin = [
            quote(764, "P", 1.20, 1.24, -0.17),
            quote(759, "P", 1.14, 1.18, -0.09),  # $6 of credit against $494 of risk
        ]
        assert build_credit_spreads(thin, underlying="SPY", widths=(5.0,), today=TODAY) == []

    def test_expiries_beyond_the_window_are_skipped(self):
        far = [
            quote(764, "P", 1.20, 1.24, -0.17, expiry="261218"),
            quote(759, "P", 0.56, 0.60, -0.09, expiry="261218"),
        ]
        assert build_credit_spreads(far, underlying="SPY", widths=(5.0,), today=TODAY) == []

    def test_legs_never_cross_expiries(self):
        mixed = [
            quote(764, "P", 1.20, 1.24, -0.17, expiry="260831"),
            quote(759, "P", 0.56, 0.60, -0.09, expiry="260901"),
        ]
        assert build_credit_spreads(mixed, underlying="SPY", widths=(5.0,), today=TODAY) == []

    def test_other_underlyings_are_ignored(self):
        foreign = [
            ContractQuote("QQQ260831P00764000", 1.20, 1.24, QUOTED_AT, delta=-0.17),
            quote(759, "P", 0.56, 0.60, -0.09),
        ]
        assert build_credit_spreads(foreign, underlying="SPY", widths=(5.0,), today=TODAY) == []


class TestCallSpreads:
    CALL_CHAIN = [
        quote(772, "C", 1.20, 1.24, 0.17),
        quote(777, "C", 0.56, 0.60, 0.09),
    ]

    def test_protection_is_above_the_short_strike_for_calls(self):
        candidates = build_credit_spreads(
            self.CALL_CHAIN, underlying="SPY", right=Right.CALL, widths=(5.0,), today=TODAY
        )
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.long_leg.strike > candidate.short_leg.strike
        assert candidate.credit_per_contract == pytest.approx(60.0)


class TestHandoffToTheGate:
    def test_produces_a_trade_the_gate_can_evaluate(self):
        candidate = build_credit_spreads(PUT_CHAIN, underlying="SPY", widths=(5.0,), today=TODAY)[0]
        trade = candidate.to_proposed_trade(contracts=3, now=QUOTED_AT)
        assert trade.contracts == 3
        assert trade.net_cash == pytest.approx(candidate.credit_per_contract * 3)
        assert len(trade.legs) == 2
        assert trade.quote_age_seconds == pytest.approx(0.0)

    def test_the_gate_agrees_with_the_candidates_own_arithmetic(self):
        from onenode.risk.payoff import worst_case_loss

        candidate = build_credit_spreads(PUT_CHAIN, underlying="SPY", widths=(5.0,), today=TODAY)[0]
        trade = candidate.to_proposed_trade(contracts=2, now=QUOTED_AT)
        assert worst_case_loss(trade.legs, trade.contracts, trade.net_cash) == pytest.approx(
            candidate.max_loss_per_contract * 2
        )


class TestSizing:
    def test_largest_count_that_fits_the_budget(self):
        candidate = build_credit_spreads(PUT_CHAIN, underlying="SPY", widths=(5.0,), today=TODAY)[0]
        # $1,500 budget against $440 of risk per contract.
        assert size_position(candidate, 100_000, max_risk_pct=1.5, max_contracts=25) == 3

    def test_capped_by_the_per_order_limit(self):
        candidate = build_credit_spreads(PUT_CHAIN, underlying="SPY", widths=(5.0,), today=TODAY)[0]
        assert size_position(candidate, 10_000_000, max_risk_pct=1.5, max_contracts=25) == 25

    def test_returns_zero_rather_than_rounding_up_to_one(self):
        candidate = build_credit_spreads(PUT_CHAIN, underlying="SPY", widths=(5.0,), today=TODAY)[0]
        # A $5,000 account cannot afford even one contract of $440 risk at 1.5%.
        assert size_position(candidate, 5_000, max_risk_pct=1.5, max_contracts=25) == 0
