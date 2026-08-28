"""Tests for regrouping open positions into structures and measuring them.

The failure this guards against is subtle and expensive: adding up the legs of
a spread separately overstates risk enormously, and an agent that overstates
its own exposure quietly stops trading halfway through the week.
"""

from __future__ import annotations

from datetime import date

import pytest

from onenode.portfolio import committed_risk, group_option_positions
from onenode.risk.models import OptionLeg, Side
from onenode.risk.payoff import remaining_risk

EXPIRY = date(2026, 9, 4)


def position(symbol: str, qty: int, side: str, market_value: float) -> dict:
    return {
        "asset_class": "us_option",
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "market_value": str(market_value),
    }


SHORT_450P = "SPY260904P00450000"
LONG_445P = "SPY260904P00445000"


class TestRemainingRisk:
    """A short put spread that cost $60 to close can still lose $440."""

    legs = (
        OptionLeg.from_symbol(SHORT_450P, side=Side.SELL),
        OptionLeg.from_symbol(LONG_445P, side=Side.BUY),
    )

    def test_distance_to_the_worst_case(self):
        assert remaining_risk(self.legs, 1, market_value=-60.0) == pytest.approx(440.0)

    def test_a_position_near_max_loss_has_little_left_to_lose(self):
        assert remaining_risk(self.legs, 1, market_value=-480.0) == pytest.approx(20.0)

    def test_long_premium_can_lose_only_what_it_is_worth(self):
        long_call = (OptionLeg.from_symbol("SPY260904C00450000", side=Side.BUY),)
        assert remaining_risk(long_call, 1, market_value=200.0) == pytest.approx(200.0)

    def test_unbounded_structure_reports_none(self):
        naked = (OptionLeg.from_symbol("SPY260904C00460000", side=Side.SELL),)
        assert remaining_risk(naked, 1, market_value=-150.0) is None

    def test_never_returns_a_negative_number(self):
        # A structure already past its worst case cannot lose a negative amount.
        assert remaining_risk(self.legs, 1, market_value=-600.0) == 0.0


class TestGrouping:
    def test_two_legs_become_one_structure(self):
        groups = group_option_positions(
            [
                position(SHORT_450P, 1, "short", -105.0),
                position(LONG_445P, 1, "long", 45.0),
            ]
        )
        assert len(groups) == 1
        group = groups[0]
        assert group.underlying == "SPY"
        assert group.expiry == EXPIRY
        assert group.contracts == 1
        assert group.market_value == pytest.approx(-60.0)
        assert group.remaining_risk == pytest.approx(440.0)

    def test_contract_count_is_factored_out_of_the_ratios(self):
        groups = group_option_positions(
            [
                position(SHORT_450P, 3, "short", -900.0),
                position(LONG_445P, 3, "long", 300.0),
            ]
        )
        group = groups[0]
        assert group.contracts == 3
        assert {leg.ratio_qty for leg in group.legs} == {1}
        assert group.remaining_risk == pytest.approx(900.0)

    def test_legs_summed_separately_would_have_been_far_worse(self):
        """The point of grouping, stated as a test."""
        groups = group_option_positions(
            [
                position(SHORT_450P, 1, "short", -105.0),
                position(LONG_445P, 1, "long", 45.0),
            ]
        )
        grouped = committed_risk(groups)
        # The short leg alone, treated as naked, is unbounded downside; measured
        # as a structure the exposure is $440.
        assert grouped == pytest.approx(440.0)

    def test_different_expiries_are_separate_structures(self):
        groups = group_option_positions(
            [
                position(SHORT_450P, 1, "short", -105.0),
                position(LONG_445P, 1, "long", 45.0),
                position("SPY260911P00440000", 1, "short", -80.0),
                position("SPY260911P00435000", 1, "long", 30.0),
            ]
        )
        assert len(groups) == 2
        assert {g.expiry for g in groups} == {EXPIRY, date(2026, 9, 11)}

    def test_different_underlyings_are_separate_structures(self):
        groups = group_option_positions(
            [
                position(SHORT_450P, 1, "short", -105.0),
                position("QQQ260904P00400000", 1, "short", -90.0),
            ]
        )
        assert {g.underlying for g in groups} == {"SPY", "QQQ"}

    def test_equity_positions_are_ignored(self):
        groups = group_option_positions(
            [
                {
                    "asset_class": "us_equity",
                    "symbol": "SPY",
                    "qty": "100",
                    "side": "long",
                    "market_value": "45000",
                },
                position(SHORT_450P, 1, "short", -105.0),
            ]
        )
        assert len(groups) == 1
        assert len(groups[0].legs) == 1

    def test_zero_quantity_and_junk_symbols_are_skipped(self):
        groups = group_option_positions(
            [
                position(SHORT_450P, 0, "short", 0.0),
                position("NOT-A-SYMBOL", 1, "long", 10.0),
            ]
        )
        assert groups == []

    def test_no_positions_means_no_risk(self):
        assert committed_risk(group_option_positions([])) == 0.0


class TestCommittedRisk:
    def test_sums_across_structures(self):
        groups = group_option_positions(
            [
                position(SHORT_450P, 1, "short", -105.0),
                position(LONG_445P, 1, "long", 45.0),
                position("QQQ260904P00400000", 1, "short", -120.0),
                position("QQQ260904P00395000", 1, "long", 50.0),
            ]
        )
        # SPY structure: 440. QQQ structure: -70 - (-500) = 430.
        assert committed_risk(groups) == pytest.approx(870.0)

    def test_an_unbounded_position_poisons_the_total(self):
        groups = group_option_positions([position("SPY260904C00460000", 1, "short", -150.0)])
        assert committed_risk(groups) is None
