"""Tests for the run loop, against a fake broker.

What matters here is not that a trade gets placed but the *order* of the safety
steps: that nothing reaches the broker without passing the reviewer and the
gate, that exits still happen when new trades are halted, and that no model is
called once the daily stop has fired.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from onenode import agent as agent_module
from onenode.agent import run_once
from onenode.agents.proposer import ProposerDecision
from onenode.agents.risk_officer import RiskVerdict
from onenode.broker.cli import AlpacaCLIError
from onenode.journal import Journal
from onenode.risk.models import MarketClock

TODAY = date(2026, 8, 31)
QUOTED_AT = datetime(2026, 8, 31, 15, 0, tzinfo=UTC)

CHAIN_PAYLOAD = {
    "snapshots": {
        "SPY260904P00764000": {
            "greeks": {"delta": -0.17},
            "latestQuote": {"bp": 1.20, "ap": 1.24, "t": QUOTED_AT.isoformat()},
        },
        "SPY260904P00759000": {
            "greeks": {"delta": -0.09},
            "latestQuote": {"bp": 0.56, "ap": 0.60, "t": QUOTED_AT.isoformat()},
        },
    }
}


class FakeCLI:
    """Stands in for the Alpaca CLI, and records every order it is handed."""

    def __init__(
        self,
        *,
        equity=100_000.0,
        last_equity=100_000.0,
        is_open=True,
        minutes_to_close=180.0,
        positions=None,
        problems=None,
    ):
        self._equity = equity
        self._last_equity = last_equity
        self._is_open = is_open
        self._minutes = minutes_to_close
        self._positions = positions or []
        self._problems = problems or []
        self.orders: list[list[str]] = []

    def daily_bars(self, symbol, sessions=260):
        """A flat year. No regime, so the regime filter blocks nothing and the
        rest of these tests keep testing what they were written to test."""
        from onenode.regime import DailyBar

        return [DailyBar(day=date(2026, 1, 1), close=100.0) for _ in range(sessions)]

    def preflight(self):
        return list(self._problems)

    def clock(self):
        return MarketClock(is_open=self._is_open, minutes_to_close=self._minutes)

    def account(self):
        return {"equity": str(self._equity), "last_equity": str(self._last_equity)}

    def positions(self):
        return list(self._positions)

    def latest_price(self, symbol):
        return 769.0

    def option_chain(self, underlying, **kwargs):
        from onenode.broker.cli import ContractQuote

        if underlying != "SPY":
            return {}
        return {
            symbol: ContractQuote.from_snapshot(symbol, snap)
            for symbol, snap in CHAIN_PAYLOAD["snapshots"].items()
        }

    def _run(self, *args):
        self.orders.append(list(args))
        return {"id": f"order-{len(self.orders)}", "status": "accepted"}


@pytest.fixture
def journal(tmp_path):
    return Journal(tmp_path / "trades.jsonl", run_id="test")


@pytest.fixture
def accepting(monkeypatch):
    """Proposer picks the first candidate; risk officer approves."""

    def _propose(candidates, **kwargs):
        return ProposerDecision(
            action="trade", candidate_key=candidates[0].key, rationale="looks fine"
        )

    monkeypatch.setattr(agent_module, "propose_trade", _propose)
    monkeypatch.setattr(
        agent_module,
        "review_trade",
        lambda *a, **k: RiskVerdict(approve=True, reason="no objection", reviewer="fake"),
    )


def run(cli, journal, **kwargs):
    return run_once(cli, journal, today=TODAY, underlyings=("SPY",), **kwargs)


class TestGuardsBeforeTrading:
    def test_a_closed_market_stops_everything(self, journal):
        cli = FakeCLI(is_open=False)
        result = run(cli, journal)
        assert result.status == "market_closed"
        assert cli.orders == []

    def test_a_failed_preflight_is_an_error(self, journal):
        cli = FakeCLI(problems=["options trading level is 1"])
        result = run(cli, journal)
        assert result.status == "error"
        assert cli.orders == []

    def test_the_last_half_hour_places_nothing(self, journal):
        cli = FakeCLI(minutes_to_close=15)
        result = run(cli, journal)
        assert result.status == "too_late"
        assert cli.orders == []


class TestDailyStop:
    def test_halts_new_positions(self, journal):
        cli = FakeCLI(equity=96_000.0)
        result = run(cli, journal)
        assert result.status == "halted"
        assert cli.orders == []

    def test_no_model_is_called_once_halted(self, journal, monkeypatch):
        """Tokens spent on a proposal that cannot be placed are tokens wasted."""

        def _explode(*args, **kwargs):
            raise AssertionError("the proposer must not run after the daily stop")

        monkeypatch.setattr(agent_module, "propose_trade", _explode)
        result = run(FakeCLI(equity=96_000.0), journal)
        assert result.status == "halted"


class TestExits:
    EXPIRING = [
        {
            "asset_class": "us_option",
            "symbol": "SPY260831P00764000",
            "qty": "2",
            "side": "short",
            "market_value": "-40",
            "cost_basis": "-300",
            "unrealized_pl": "260",
        },
        {
            "asset_class": "us_option",
            "symbol": "SPY260831P00759000",
            "qty": "2",
            "side": "long",
            "market_value": "10",
            "cost_basis": "100",
            "unrealized_pl": "-90",
        },
    ]

    def test_expiring_positions_are_closed(self, journal, accepting):
        cli = FakeCLI(positions=self.EXPIRING, minutes_to_close=60)
        result = run(cli, journal)
        assert result.positions_closed == 1
        assert any("_to_close" in " ".join(order) for order in cli.orders)

    def test_exits_still_run_when_new_trades_are_halted(self, journal):
        """Closing a position is how you stop losing money - never gate it."""
        cli = FakeCLI(equity=96_000.0, positions=self.EXPIRING, minutes_to_close=60)
        result = run(cli, journal)
        assert result.status == "halted"
        assert result.positions_closed == 1
        assert result.orders_placed == 0


class TestProposalPath:
    def test_standing_aside_places_nothing(self, journal, monkeypatch):
        monkeypatch.setattr(
            agent_module,
            "propose_trade",
            lambda *a, **k: ProposerDecision(action="stand_aside", rationale="not today"),
        )
        cli = FakeCLI()
        result = run(cli, journal)
        assert result.status == "stood_aside"
        assert cli.orders == []

    def test_a_veto_places_nothing(self, journal, monkeypatch, accepting):
        monkeypatch.setattr(
            agent_module,
            "review_trade",
            lambda *a, **k: RiskVerdict(approve=False, reason="too close to spot", reviewer="fake"),
        )
        cli = FakeCLI()
        result = run(cli, journal)
        assert result.status == "vetoed"
        assert cli.orders == []

    def test_a_broken_proposer_places_nothing(self, journal, monkeypatch):
        def _explode(*args, **kwargs):
            raise RuntimeError("no API key")

        monkeypatch.setattr(agent_module, "propose_trade", _explode)
        cli = FakeCLI()
        result = run(cli, journal)
        assert result.status == "proposer_failed"
        assert cli.orders == []

    def test_a_key_that_is_not_on_the_menu_places_nothing(self, journal, monkeypatch):
        monkeypatch.setattr(
            agent_module,
            "propose_trade",
            lambda *a, **k: ProposerDecision(
                action="trade", candidate_key="MADE/UP", rationale="invented"
            ),
        )
        cli = FakeCLI()
        result = run(cli, journal)
        assert result.status == "stood_aside"
        assert cli.orders == []


class TestHappyPath:
    def test_an_approved_trade_reaches_the_broker(self, journal, accepting):
        cli = FakeCLI()
        result = run(cli, journal)
        assert result.status == "traded"
        assert result.orders_placed == 1
        order = cli.orders[0]
        assert "mleg" in order
        assert any(arg.startswith("onenode-") for arg in order)

    def test_the_run_is_journalled_end_to_end(self, journal, accepting):
        run(FakeCLI(), journal)
        events = [entry["event"] for entry in journal.entries()]
        assert events[0] == "run_started"
        assert "gate_approved" in events
        assert "order_placed" in events

    def test_dry_run_still_journals_but_marks_the_order(self, journal, accepting):
        cli = FakeCLI()
        result = run(cli, journal, dry_run=True)
        assert result.status == "traded"
        assert "--dry-run" in cli.orders[0]


class TestDuplicateStructures:
    """Two identical spreads merge into one position group.

    That makes the max-positions ceiling blind to them: the agent could buy the
    same strike pair every fifteen minutes and concentrate the whole risk budget
    into one structure without the position count ever moving.
    """

    OPEN_ALREADY = [
        {
            "asset_class": "us_option",
            "symbol": "SPY260904P00764000",
            "qty": "1",
            "side": "short",
            "market_value": "-100",
            "cost_basis": "-120",
            "unrealized_pl": "20",
        },
        {
            "asset_class": "us_option",
            "symbol": "SPY260904P00759000",
            "qty": "1",
            "side": "long",
            "market_value": "45",
            "cost_basis": "60",
            "unrealized_pl": "-15",
        },
    ]

    def test_a_structure_already_open_is_not_offered_again(self, journal, accepting):
        cli = FakeCLI(positions=self.OPEN_ALREADY)
        result = run(cli, journal)
        assert result.orders_placed == 0
        assert result.status == "no_candidates"

    def test_the_filtering_is_recorded(self, journal, accepting):
        run(FakeCLI(positions=self.OPEN_ALREADY), journal)
        events = [e["event"] for e in journal.entries()]
        assert "duplicates_filtered" in events

    def test_a_different_structure_still_gets_through(self, journal, accepting):
        other = [dict(p) for p in self.OPEN_ALREADY]
        other[0]["symbol"] = "SPY260904P00755000"
        other[1]["symbol"] = "SPY260904P00750000"
        cli = FakeCLI(positions=other)
        result = run(cli, journal)
        assert result.orders_placed == 1


class TestUnmeasurableRisk:
    def test_an_unbounded_open_position_blocks_new_trades(self, journal, accepting):
        """Not knowing the exposure is treated as infinite exposure."""
        naked = [
            {
                "asset_class": "us_option",
                "symbol": "SPY260904C00790000",
                "qty": "1",
                "side": "short",
                "market_value": "-120",
                "cost_basis": "-150",
                "unrealized_pl": "30",
            }
        ]
        cli = FakeCLI(positions=naked)
        result = run(cli, journal)
        assert result.status == "blocked"
        assert result.orders_placed == 0


class TestRegimeFilter:
    """The chain in this file is puts only, so a bear regime empties the menu
    entirely - which makes it easy to see whether the filter fired at all."""

    @staticmethod
    def _cli_in(trend_pct_per_session):
        class Trending(FakeCLI):
            def daily_bars(self, symbol, sessions=260):
                from onenode.regime import DailyBar

                bars, close = [], 100.0
                for _index in range(sessions):
                    close *= 1 + trend_pct_per_session / 100.0
                    bars.append(DailyBar(day=date(2026, 1, 1), close=close))
                return bars

        return Trending()

    def test_puts_are_not_sold_into_a_falling_market(self, journal, accepting):
        cli = self._cli_in(-0.5)
        result = run(cli, journal)
        assert result.orders_placed == 0
        assert cli.orders == []
        events = [e["event"] for e in journal.entries()]
        assert "regime_blocked" in events

    def test_a_flat_market_leaves_the_menu_alone(self, journal, accepting):
        cli = self._cli_in(0.0)
        result = run(cli, journal)
        assert result.orders_placed == 1
        assert "regime_blocked" not in [e["event"] for e in journal.entries()]

    def test_a_broker_that_cannot_serve_history_does_not_halt_trading(self, journal, accepting):
        """A data outage must cost the refinement, not the session."""

        class NoHistory(FakeCLI):
            def daily_bars(self, symbol, sessions=260):
                raise AlpacaCLIError("bars endpoint unavailable")

        cli = NoHistory()
        result = run(cli, journal)
        assert result.orders_placed == 1
        events = [e["event"] for e in journal.entries()]
        assert "regime_unavailable" in events

    def test_the_regime_is_journalled_every_run(self, journal, accepting):
        cli = self._cli_in(0.0)
        run(cli, journal)
        recorded = [e for e in journal.entries() if e["event"] == "regime"]
        assert recorded
        assert "neutral regime" in recorded[0]["reason"]
