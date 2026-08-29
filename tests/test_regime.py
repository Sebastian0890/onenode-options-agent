"""Tests for the regime filter.

The rule under test is narrow on purpose: do not sell the side the market is
currently moving against. So the tests check the classification, the block, and
the two ways this can go wrong quietly - a matrix built from too little history
passing itself off as knowledge, and a data outage silently halting the agent
instead of silently un-narrowing it.

The transition matrix is checked for arithmetic, not for predictive power. It
does not claim any, and a test asserting otherwise would be asserting the thing
the module docstring says is untrue.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from onenode import regime
from onenode.regime import BEAR, BULL, NEUTRAL, DailyBar, Regime, TransitionMatrix, from_bars
from onenode.risk.models import Right


def series(*returns_pct: float, start: float = 100.0) -> list[DailyBar]:
    """Daily bars whose closes compound the given per-session returns."""
    bars = [DailyBar(day=date(2026, 1, 1), close=start)]
    close = start
    for index, step in enumerate(returns_pct, start=1):
        close *= 1 + step / 100.0
        bars.append(DailyBar(day=date(2026, 1, 1) + timedelta(days=index), close=close))
    return bars


FLAT = series(*([0.0] * 60))
RISING = series(*([0.5] * 60))
FALLING = series(*([-0.5] * 60))


class TestClassification:
    @pytest.mark.parametrize(
        ("window_return", "state"),
        [(7.0, BULL), (5.0, BULL), (4.9, NEUTRAL), (0.0, NEUTRAL), (-4.9, NEUTRAL), (-5.0, BEAR)],
    )
    def test_thresholds_are_inclusive_at_the_edge(self, window_return, state):
        assert regime.classify(window_return) == state

    def test_a_steady_climb_is_a_bull_regime(self):
        assert from_bars("SPY", RISING).state == BULL

    def test_a_steady_slide_is_a_bear_regime(self):
        assert from_bars("SPY", FALLING).state == BEAR

    def test_a_flat_tape_is_neutral(self):
        assert from_bars("SPY", FLAT).state == NEUTRAL

    def test_the_window_is_trailing_not_cumulative(self):
        """A market that fell hard and then recovered is not still a bear.

        Measuring from the start of the series instead of twenty sessions back
        would keep the agent out of puts for months after a drawdown ended.
        """
        crash_then_recovery = series(*([-1.0] * 20 + [1.0] * 30))
        assert from_bars("SPY", crash_then_recovery).state != BEAR


class TestBlocking:
    def test_puts_are_refused_in_a_bear_regime(self):
        market = from_bars("SPY", FALLING)
        assert market.blocks(Right.PUT)
        assert not market.blocks(Right.CALL)

    def test_calls_are_refused_in_a_bull_regime(self):
        market = from_bars("SPY", RISING)
        assert market.blocks(Right.CALL)
        assert not market.blocks(Right.PUT)

    def test_a_neutral_regime_refuses_nothing(self):
        market = from_bars("SPY", FLAT)
        assert not market.blocks(Right.PUT)
        assert not market.blocks(Right.CALL)

    def test_missing_data_refuses_nothing_and_says_so(self):
        """Fail open, deliberately.

        This filter narrows a system that is already safe without it. An outage
        that halted trading would turn a data problem into a strategy decision
        nobody made.
        """
        market = Regime.unavailable("SPY")
        assert not market.known
        assert not market.blocks(Right.PUT)
        assert not market.blocks(Right.CALL)
        assert "unknown" in market.describe()

    def test_too_little_history_is_unavailable_rather_than_a_guess(self):
        """Fewer bars than the window is not a neutral market. It is no market."""
        assert not from_bars("SPY", series(*([0.1] * 5))).known

    def test_a_state_read_back_as_a_plain_string_still_blocks(self):
        """The state may arrive from somewhere other than classify() - a
        journal replay, a fixture. Identity comparison would pass every test
        above and fail on exactly that."""
        market = Regime(
            underlying="SPY",
            state="bear",
            window_return_pct=-9.0,
            matrix=TransitionMatrix.from_states([]),
            sessions=0,
        )
        assert market.blocks(Right.PUT)


class TestTransitionMatrix:
    def test_counts_what_followed_what(self):
        matrix = TransitionMatrix.from_states([BULL, BULL, NEUTRAL, NEUTRAL, BEAR])
        assert matrix.counts[BULL][BULL] == 1
        assert matrix.counts[BULL][NEUTRAL] == 1
        assert matrix.counts[NEUTRAL][NEUTRAL] == 1
        assert matrix.counts[NEUTRAL][BEAR] == 1
        assert matrix.observations(BEAR) == 0

    def test_rows_are_probabilities(self):
        matrix = TransitionMatrix.from_states([BULL, NEUTRAL, BULL, BULL])
        assert sum(matrix.row(BULL).values()) == pytest.approx(1.0)

    def test_an_unobserved_row_is_uniform_not_empty(self):
        """Zeros would leak into the k-step multiplication as certainty about
        a state never once seen."""
        matrix = TransitionMatrix.from_states([NEUTRAL, NEUTRAL])
        assert matrix.row(BEAR) == pytest.approx({BULL: 1 / 3, NEUTRAL: 1 / 3, BEAR: 1 / 3})

    def test_the_horizon_compounds(self):
        """Two steps must not equal one step; otherwise the DTE is decorative."""
        matrix = TransitionMatrix.from_states([BULL, BULL, NEUTRAL, NEUTRAL, NEUTRAL, BEAR])
        assert matrix.after(BULL, 1) != pytest.approx(matrix.after(BULL, 2))

    def test_a_zero_horizon_is_where_it_already_is(self):
        matrix = TransitionMatrix.from_states([BULL, NEUTRAL])
        assert matrix.after(BULL, 0)[BULL] == pytest.approx(1.0)

    def test_the_distribution_stays_a_distribution(self):
        matrix = TransitionMatrix.from_states([BULL, NEUTRAL, BEAR, NEUTRAL, BULL])
        for horizon in (1, 5, 20):
            assert sum(matrix.after(NEUTRAL, horizon).values()) == pytest.approx(1.0)


class TestHonesty:
    def test_the_description_carries_its_own_sample_size(self):
        """A row backed by four observations and one backed by four hundred
        read identically once normalised. The count is what stops the first
        from being quoted like the second."""
        market = from_bars("SPY", FLAT)
        assert f"{market.matrix.observations(market.state)} transitions" in market.describe()
        assert f"in {market.sessions} sessions" in market.describe()

    def test_the_description_states_the_measured_move(self):
        market = from_bars("SPY", RISING)
        assert "over 20 sessions" in market.describe()
        assert market.window_return_pct > 5.0
