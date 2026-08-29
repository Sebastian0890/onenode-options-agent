"""Tests for the pricing arithmetic on a candidate.

The filter these pin down is the one about execution cost, because that is the
only figure on this list a snapshot can settle honestly. The expectancy numbers
beside it are informational, and the tests say so: they assert the *ordering*
those numbers give, never that a particular sign means a good trade. A test
that asserted "edge above zero means profitable" would be encoding the mistake
the class docstring exists to warn about.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from onenode.broker.cli import ContractQuote
from onenode.risk.models import Right
from onenode.strategy import build_credit_spreads

TODAY = date(2026, 8, 28)
QUOTED_AT = datetime(2026, 8, 28, 18, 42, tzinfo=UTC)


def quote(strike: int, bid: float, ask: float, delta: float) -> ContractQuote:
    return ContractQuote(
        symbol=f"SPY260831P{strike * 1000:08d}",
        bid=bid,
        ask=ask,
        quoted_at=QUOTED_AT,
        delta=delta,
    )


def spreads(chain, **kwargs):
    return build_credit_spreads(chain, underlying="SPY", right=Right.PUT, today=TODAY, **kwargs)


# The calibration chain: SPY near 769, penny-wide quotes.
TIGHT = [
    quote(764, 1.20, 1.24, -0.17),
    quote(759, 0.56, 0.60, -0.09),
]

# A one-dollar spread. Every leg here passes the per-leg spread filter - each
# bid-ask is under 10% of its own mid - and the structure is still ruinous,
# because the cost is charged against the *net* credit and the net credit of a
# narrow spread is small. SPY has dollar strikes and daily expiries, so this is
# the shape the agent reaches for most often, not an exotic case.
NARROW = [
    quote(764, 1.16, 1.28, -0.17),
    quote(763, 0.92, 1.01, -0.15),
]


class TestExecutionCost:
    def test_measures_what_crossing_the_spread_costs(self):
        candidate = spreads(TIGHT)[0]
        # mid 1.22 - mid 0.58 = 0.64 theoretical; bid 1.20 - ask 0.60 = 0.60 real.
        assert candidate.mid_credit_per_contract == pytest.approx(64.0)
        assert candidate.credit_per_contract == pytest.approx(60.0)
        assert candidate.execution_cost_per_contract == pytest.approx(4.0)
        assert candidate.execution_drag == pytest.approx(0.0625)

    def test_a_liquid_spread_survives_the_filter(self):
        assert spreads(TIGHT, max_execution_drag=0.30)

    def test_a_spread_that_eats_its_own_premium_is_dropped(self):
        candidate = spreads(NARROW, max_execution_drag=1.0)[0]
        assert candidate.mid_credit_per_contract == pytest.approx(25.5)
        assert candidate.credit_per_contract == pytest.approx(15.0)
        assert candidate.execution_drag > 0.40
        assert spreads(NARROW, max_execution_drag=0.30) == []

    def test_the_per_leg_spread_filter_does_not_already_catch_this(self):
        """The reason this filter is not redundant.

        Both legs above are quoted inside the 10%-of-mid ceiling the quote
        filter enforces, so that filter passes them. The cost is charged
        against the net credit, which for a one-dollar spread is a fraction of
        either leg - so two individually respectable quotes combine into a
        structure that gives away 41% of its premium on the way in.
        """
        for leg in NARROW:
            assert leg.spread_pct_of_mid < 10.0
        assert spreads(NARROW, max_spread_pct=10.0, max_execution_drag=1.0)
        assert spreads(NARROW, max_spread_pct=10.0, max_execution_drag=0.30) == []

    def test_the_filter_is_inert_at_its_permissive_end(self):
        """A ceiling of 1.0 must not reject; a knob has to do nothing at its
        open end, or its default is doing something unexamined."""
        assert spreads(NARROW, max_execution_drag=1.0)


class TestExpectancyIsComparative:
    def test_break_even_is_the_inverse_of_reward_to_risk(self):
        candidate = spreads(TIGHT)[0]
        assert candidate.break_even_win_rate == 1.0 / (1.0 + candidate.reward_to_risk)

    def test_implied_win_rate_comes_from_the_short_delta(self):
        candidate = spreads(TIGHT)[0]
        assert candidate.implied_win_rate == pytest.approx(0.83)

    def test_a_better_credit_for_the_same_odds_scores_higher(self):
        """The ordering is the part that means something."""
        generous = spreads([quote(764, 1.60, 1.64, -0.17), quote(759, 0.56, 0.60, -0.09)])[0]
        stingy = spreads(TIGHT)[0]
        assert generous.short_delta == stingy.short_delta
        assert generous.edge > stingy.edge

    def test_the_calibration_spread_prices_at_roughly_its_execution_cost(self):
        """This is the number that killed the first version of the filter.

        An ordinary, liquid, correctly-priced SPY spread scores about -5% on
        the binary break-even measure - not because the market is mispricing
        it, but because charging a full loss for a partial one costs about what
        the bid-ask does. Gating on this being positive would have refused
        every trade for the whole week.
        """
        candidate = spreads(TIGHT)[0]
        assert candidate.edge < 0
        assert abs(candidate.edge) < 0.10


class TestCondorProbabilities:
    def test_both_wings_count_toward_being_tested(self):
        from onenode.strategy import build_iron_condors

        calls = [
            ContractQuote(
                symbol=f"SPY260831C{strike * 1000:08d}",
                bid=bid,
                ask=ask,
                quoted_at=QUOTED_AT,
                delta=delta,
            )
            for strike, bid, ask, delta in [(774, 1.20, 1.24, 0.17), (779, 0.56, 0.60, 0.09)]
        ]
        call_spreads = build_credit_spreads(calls, underlying="SPY", right=Right.CALL, today=TODAY)
        condors = build_iron_condors(spreads(TIGHT), call_spreads, min_reward_to_risk=0.0)
        assert condors
        condor = condors[0]
        # A single vertical is tested 17% of the time; the pair, 34%. That is
        # the cost the extra credit is paying for, and averaging the two wings
        # instead of adding them would hide it.
        assert condor.tested_probability == pytest.approx(0.34)
        assert condor.implied_win_rate == pytest.approx(0.66)
        assert condor.tested_probability > condor.put_side.tested_probability
