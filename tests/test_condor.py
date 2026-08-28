"""Tests for iron condors.

The claim these check is the reason condors are worth building at all: two
credits against roughly one width, because only one side can finish in the
money. If the arithmetic here were wrong the structure would look better than
it is, which is the most expensive kind of wrong.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from onenode.broker.cli import ContractQuote
from onenode.execution import leg_payload
from onenode.risk.models import Right, Side
from onenode.risk.payoff import worst_case_loss
from onenode.strategy import build_credit_spreads, build_iron_condors, size_position

TODAY = date(2026, 8, 28)
QUOTED_AT = datetime(2026, 8, 28, 18, 42, tzinfo=UTC)


def quote(strike, right, bid, ask, delta, expiry="260831"):
    return ContractQuote(
        symbol=f"SPY{expiry}{right}{strike * 1000:08d}",
        bid=bid,
        ask=ask,
        quoted_at=QUOTED_AT,
        delta=delta,
    )


PUT_CHAIN = [quote(764, "P", 1.20, 1.24, -0.17), quote(759, "P", 0.56, 0.60, -0.09)]
CALL_CHAIN = [quote(774, "C", 1.20, 1.24, 0.17), quote(779, "C", 0.56, 0.60, 0.09)]


def sides(puts=PUT_CHAIN, calls=CALL_CHAIN):
    put_spreads = build_credit_spreads(puts, underlying="SPY", widths=(5.0,), today=TODAY)
    call_spreads = build_credit_spreads(
        calls, underlying="SPY", right=Right.CALL, widths=(5.0,), today=TODAY
    )
    return put_spreads, call_spreads


@pytest.fixture
def condor():
    put_spreads, call_spreads = sides()
    return build_iron_condors(put_spreads, call_spreads, min_reward_to_risk=0.0)[0]


class TestArithmetic:
    def test_credit_is_the_sum_of_both_sides(self, condor):
        assert condor.credit_per_contract == pytest.approx(120.0)

    def test_risk_is_one_width_not_two(self, condor):
        """The whole point: only one wing can finish in the money."""
        assert condor.max_loss_per_contract == pytest.approx(380.0)

    def test_reward_to_risk_beats_either_side_alone(self, condor):
        put_spreads, _ = sides()
        assert condor.reward_to_risk == pytest.approx(120.0 / 380.0)
        assert condor.reward_to_risk > 2 * put_spreads[0].reward_to_risk * 0.9

    def test_risk_agrees_with_the_gates_own_engine(self, condor):
        """The quoted number and the enforced number come from one implementation."""
        trade = condor.to_proposed_trade(contracts=4, now=QUOTED_AT)
        assert worst_case_loss(trade.legs, 4, trade.net_cash) == pytest.approx(
            condor.max_loss_per_contract * 4
        )

    def test_reports_the_wing_most_likely_to_be_tested(self, condor):
        assert condor.short_delta == pytest.approx(0.17)


class TestStructure:
    def test_has_four_legs_two_short_two_long(self, condor):
        assert len(condor.legs) == 4
        shorts = [leg for leg in condor.legs if leg.side is Side.SELL]
        longs = [leg for leg in condor.legs if leg.side is Side.BUY]
        assert len(shorts) == 2
        assert len(longs) == 2

    def test_shorts_straddle_the_price_and_longs_sit_outside(self, condor):
        strikes = {leg.symbol: leg.strike for leg in condor.legs}
        assert min(strikes.values()) == 759  # long put, furthest out
        assert max(strikes.values()) == 779  # long call, furthest out

    def test_all_legs_share_one_expiry(self, condor):
        assert len({leg.expiry for leg in condor.legs}) == 1

    def test_the_order_carries_all_four_legs(self, condor):
        """Alpaca accepts up to four legs, which a condor exactly fills."""
        payload = leg_payload(condor)
        assert len(payload) == 4
        assert all(leg["position_intent"].endswith("_to_open") for leg in payload)

    def test_key_names_both_sides(self, condor):
        assert "+" in condor.key
        assert condor.key.count("/") == 2


class TestPairingRules:
    def test_an_inverted_condor_is_rejected(self):
        """Short call below the short put means both wings can lose at once."""
        low_calls = [quote(760, "C", 1.20, 1.24, 0.17), quote(765, "C", 0.56, 0.60, 0.09)]
        put_spreads, call_spreads = sides(calls=low_calls)
        assert build_iron_condors(put_spreads, call_spreads, min_reward_to_risk=0.0) == []

    def test_sides_from_different_expiries_are_not_paired(self):
        other = [
            quote(774, "C", 1.20, 1.24, 0.17, expiry="260901"),
            quote(779, "C", 0.56, 0.60, 0.09, expiry="260901"),
        ]
        put_spreads, call_spreads = sides(calls=other)
        assert build_iron_condors(put_spreads, call_spreads, min_reward_to_risk=0.0) == []

    def test_the_floor_is_stricter_than_for_a_single_vertical(self):
        """A condor that does not clearly beat its own put side is not worth four legs."""
        put_spreads, call_spreads = sides()
        assert build_iron_condors(put_spreads, call_spreads, min_reward_to_risk=0.20)
        assert build_iron_condors(put_spreads, call_spreads, min_reward_to_risk=0.90) == []

    def test_only_the_best_few_per_expiry_are_returned(self):
        """Pairing every put with every call buries the model in near-identical noise."""
        puts = [
            quote(764, "P", 1.20, 1.24, -0.17),
            quote(763, "P", 1.10, 1.14, -0.16),
            quote(762, "P", 1.00, 1.04, -0.15),
            quote(759, "P", 0.56, 0.60, -0.09),
            quote(758, "P", 0.50, 0.54, -0.08),
            quote(757, "P", 0.46, 0.50, -0.07),
        ]
        calls = [
            quote(774, "C", 1.20, 1.24, 0.17),
            quote(775, "C", 1.10, 1.14, 0.16),
            quote(776, "C", 1.00, 1.04, 0.15),
            quote(779, "C", 0.56, 0.60, 0.09),
            quote(780, "C", 0.50, 0.54, 0.08),
            quote(781, "C", 0.46, 0.50, 0.07),
        ]
        put_spreads, call_spreads = sides(puts, calls)
        assert len(put_spreads) * len(call_spreads) > 3
        condors = build_iron_condors(put_spreads, call_spreads, min_reward_to_risk=0.0)
        assert len(condors) == 3

    def test_ranked_by_reward_to_risk(self):
        put_spreads, call_spreads = sides()
        condors = build_iron_condors(put_spreads, call_spreads, min_reward_to_risk=0.0)
        ratios = [c.reward_to_risk for c in condors]
        assert ratios == sorted(ratios, reverse=True)


class TestSizing:
    def test_sized_against_the_same_budget_as_a_vertical(self, condor):
        # $1,500 of budget against $380 of risk per contract.
        assert size_position(condor, 100_000, max_risk_pct=1.5, max_contracts=25) == 3

    def test_more_credit_for_the_same_risk_budget(self, condor):
        put_spreads, _ = sides()
        vertical = put_spreads[0]
        condor_size = size_position(condor, 100_000, 1.5, 25)
        vertical_size = size_position(vertical, 100_000, 1.5, 25)
        assert condor.credit_per_contract * condor_size > (
            vertical.credit_per_contract * vertical_size
        )
