"""Tests for the exit rules.

Closing is where the money is, and these rules run unattended during the
trading day - so each one is pinned to an arithmetic case rather than left to
judgement at the moment it fires.
"""

from __future__ import annotations

from datetime import date

from onenode.management import DEFAULT_EXIT_RULES, ExitRules, evaluate_exit, positions_to_close
from onenode.portfolio import group_option_positions

TODAY = date(2026, 8, 31)


def spread(expiry: str = "260904", unrealized_pl: float = 0.0, credit: float = 200.0):
    """One short put spread, 2 contracts, opened for `credit` dollars."""
    positions = [
        {
            "asset_class": "us_option",
            "symbol": f"SPY{expiry}P00764000",
            "qty": "2",
            "side": "short",
            "market_value": "-150",
            "cost_basis": str(-credit - 100),
            "unrealized_pl": str(unrealized_pl),
        },
        {
            "asset_class": "us_option",
            "symbol": f"SPY{expiry}P00759000",
            "qty": "2",
            "side": "long",
            "market_value": "60",
            "cost_basis": "100",
            "unrealized_pl": "0",
        },
    ]
    return group_option_positions(positions)[0]


class TestProfitTarget:
    def test_closes_once_half_the_credit_is_banked(self):
        group = spread(unrealized_pl=110.0, credit=200.0)
        decision = evaluate_exit(group, today=TODAY, minutes_to_close=200)
        assert decision.should_close
        assert "max profit" in decision.reason

    def test_holds_below_the_target(self):
        group = spread(unrealized_pl=60.0, credit=200.0)
        assert not evaluate_exit(group, today=TODAY, minutes_to_close=200)

    def test_target_is_configurable(self):
        group = spread(unrealized_pl=60.0, credit=200.0)
        rules = ExitRules(profit_target=0.25)
        assert evaluate_exit(group, today=TODAY, minutes_to_close=200, rules=rules)


class TestStopLoss:
    def test_closes_at_twice_the_credit_lost(self):
        group = spread(unrealized_pl=-410.0, credit=200.0)
        decision = evaluate_exit(group, today=TODAY, minutes_to_close=200)
        assert decision.should_close
        assert "2x" in decision.reason

    def test_holds_a_loss_that_has_not_reached_the_stop(self):
        group = spread(unrealized_pl=-250.0, credit=200.0)
        assert not evaluate_exit(group, today=TODAY, minutes_to_close=200)


class TestExpiry:
    def test_flattens_on_expiry_day_inside_the_window(self):
        group = spread(expiry="260831", unrealized_pl=10.0)
        decision = evaluate_exit(group, today=TODAY, minutes_to_close=60)
        assert decision.should_close
        assert "assignment risk" in decision.reason

    def test_a_profitable_position_is_still_closed_on_expiry_day(self):
        """Being green does not make an expiring short spread safe to carry."""
        group = spread(expiry="260831", unrealized_pl=190.0)
        assert evaluate_exit(group, today=TODAY, minutes_to_close=30)

    def test_holds_earlier_in_the_expiry_session(self):
        group = spread(expiry="260831", unrealized_pl=10.0)
        assert not evaluate_exit(group, today=TODAY, minutes_to_close=300)

    def test_anything_past_expiry_is_closed_on_sight(self):
        group = spread(expiry="260828", unrealized_pl=10.0)
        decision = evaluate_exit(group, today=TODAY, minutes_to_close=300)
        assert decision.should_close
        assert "expired" in decision.reason

    def test_expiry_beats_every_other_rule(self):
        group = spread(expiry="260831", unrealized_pl=-9999.0)
        decision = evaluate_exit(group, today=TODAY, minutes_to_close=30)
        assert "assignment risk" in decision.reason


class TestNonCreditStructures:
    def test_a_debit_structure_is_left_alone(self):
        """Nothing here knows what a long position's target should be."""
        positions = [
            {
                "asset_class": "us_option",
                "symbol": "SPY260904P00764000",
                "qty": "1",
                "side": "long",
                "market_value": "300",
                "cost_basis": "200",
                "unrealized_pl": "100",
            }
        ]
        group = group_option_positions(positions)[0]
        assert group.credit_received == 0.0
        assert not evaluate_exit(group, today=TODAY, minutes_to_close=200)


class TestBatch:
    def test_returns_every_group_that_qualifies_with_its_reason(self):
        groups = [
            spread(expiry="260904", unrealized_pl=150.0, credit=200.0),
            spread(expiry="260904", unrealized_pl=10.0, credit=200.0),
        ]
        closing = positions_to_close(groups, today=TODAY, minutes_to_close=200)
        assert len(closing) == 1
        assert closing[0][1]

    def test_defaults_are_the_documented_ones(self):
        assert DEFAULT_EXIT_RULES.profit_target == 0.50
        assert DEFAULT_EXIT_RULES.stop_multiple == 2.0
