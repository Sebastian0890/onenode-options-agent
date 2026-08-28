"""Tests for order construction.

The shapes here were confirmed against the broker's own --dry-run on
2026-08-28, which accepted the body these functions produce.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest

from onenode.broker.cli import ContractQuote
from onenode.execution import (
    build_order_args,
    closing_leg_payload,
    closing_limit_price,
    leg_payload,
    limit_price_for_credit,
)
from onenode.portfolio import group_option_positions
from onenode.strategy import build_credit_spreads

QUOTED_AT = datetime(2026, 8, 28, 18, 42, tzinfo=UTC)
TODAY = date(2026, 8, 28)


def quote(strike: int, bid: float, ask: float, delta: float) -> ContractQuote:
    return ContractQuote(
        symbol=f"SPY260831P{strike * 1000:08d}", bid=bid, ask=ask, quoted_at=QUOTED_AT, delta=delta
    )


CHAIN = [quote(764, 1.20, 1.24, -0.17), quote(759, 0.56, 0.60, -0.09)]


@pytest.fixture
def candidate():
    return build_credit_spreads(CHAIN, underlying="SPY", widths=(5.0,), today=TODAY)[0]


class TestOpeningLegs:
    def test_short_leg_sells_to_open_and_long_leg_buys_to_open(self, candidate):
        legs = leg_payload(candidate)
        assert legs[0] == {
            "symbol": "SPY260831P00764000",
            "side": "sell",
            "ratio_qty": "1",
            "position_intent": "sell_to_open",
        }
        assert legs[1]["side"] == "buy"
        assert legs[1]["position_intent"] == "buy_to_open"

    def test_intent_is_always_explicit(self, candidate):
        """Left implicit, a broker may net a new spread against an open one."""
        for leg in leg_payload(candidate):
            assert leg["position_intent"].endswith("_to_open")


class TestClosingLegs:
    def test_every_side_is_reversed(self, candidate):
        legs = closing_leg_payload((candidate.short_leg, candidate.long_leg))
        assert legs[0]["side"] == "buy"
        assert legs[0]["position_intent"] == "buy_to_close"
        assert legs[1]["side"] == "sell"
        assert legs[1]["position_intent"] == "sell_to_close"

    def test_closes_what_the_broker_reports_not_what_opened_it(self):
        groups = group_option_positions(
            [
                {
                    "asset_class": "us_option",
                    "symbol": "SPY260831P00764000",
                    "qty": "2",
                    "side": "short",
                    "market_value": "-90",
                    "cost_basis": "-240",
                    "unrealized_pl": "150",
                },
                {
                    "asset_class": "us_option",
                    "symbol": "SPY260831P00759000",
                    "qty": "2",
                    "side": "long",
                    "market_value": "40",
                    "cost_basis": "120",
                    "unrealized_pl": "-80",
                },
            ]
        )
        legs = closing_leg_payload(groups[0].legs)
        assert {leg["side"] for leg in legs} == {"buy", "sell"}
        assert all(leg["position_intent"].endswith("_to_close") for leg in legs)


class TestPricing:
    def test_credit_limit_is_shaded_below_the_quoted_credit(self, candidate):
        # $60 per contract is $0.60 per share; shaded by two cents.
        assert limit_price_for_credit(candidate) == pytest.approx(0.58)

    def test_credit_limit_never_goes_to_zero(self, candidate):
        assert limit_price_for_credit(candidate, slippage=99.0) == pytest.approx(0.01)

    def test_closing_price_is_a_positive_debit_with_a_cushion(self):
        groups = group_option_positions(
            [
                {
                    "asset_class": "us_option",
                    "symbol": "SPY260831P00764000",
                    "qty": "2",
                    "side": "short",
                    "market_value": "-90",
                    "cost_basis": "-240",
                    "unrealized_pl": "150",
                },
                {
                    "asset_class": "us_option",
                    "symbol": "SPY260831P00759000",
                    "qty": "2",
                    "side": "long",
                    "market_value": "40",
                    "cost_basis": "120",
                    "unrealized_pl": "-80",
                },
            ]
        )
        # Net -50 across 2 contracts is $0.25 per share, plus the $0.05 cushion.
        assert closing_limit_price(groups[0]) == pytest.approx(0.30)


class TestOrderArgs:
    def test_multi_leg_limit_order_shape(self, candidate):
        args = build_order_args(leg_payload(candidate), 18, 0.16)
        pairs = dict(zip(args, args[1:], strict=False))
        assert args[:2] == ["order", "submit"]
        assert pairs["--order-class"] == "mleg"
        assert pairs["--qty"] == "18"
        assert pairs["--type"] == "limit"
        assert pairs["--limit-price"] == "0.16"
        assert pairs["--time-in-force"] == "day"

    def test_legs_are_compact_json(self, candidate):
        args = build_order_args(leg_payload(candidate), 1, 0.5)
        payload = json.loads(args[args.index("--legs") + 1])
        assert len(payload) == 2
        assert payload[0]["symbol"] == "SPY260831P00764000"

    def test_time_in_force_is_always_day(self, candidate):
        """The agent should never wake up owning something it ordered yesterday."""
        args = build_order_args(leg_payload(candidate), 1, 0.5)
        assert args[args.index("--time-in-force") + 1] == "day"

    def test_dry_run_is_opt_in(self, candidate):
        assert "--dry-run" not in build_order_args(leg_payload(candidate), 1, 0.5)
        assert "--dry-run" in build_order_args(leg_payload(candidate), 1, 0.5, dry_run=True)

    def test_client_order_id_is_passed_through(self, candidate):
        args = build_order_args(leg_payload(candidate), 1, 0.5, client_order_id="onenode-abc")
        assert args[args.index("--client-order-id") + 1] == "onenode-abc"

    def test_limit_price_is_always_two_decimals(self, candidate):
        args = build_order_args(leg_payload(candidate), 1, 0.5)
        assert args[args.index("--limit-price") + 1] == "0.50"
