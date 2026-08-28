"""Tests for the CLI boundary.

The snapshot fixture below is real output captured from a live SPY chain on
2026-08-28, not something invented to match the parser. That is the point: the
field names were originally guessed from documentation that does not publish
them, and this pins what the API actually sends.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from onenode.broker.cli import ContractQuote, _parse_timestamp, chain_args

# Verbatim from `alpaca data option chain --underlying-symbol SPY`, trimmed to
# the keys the agent reads. SPY was trading at 769.12.
LIVE_SNAPSHOT = {
    "dailyBar": {"c": 1.9, "h": 2.4, "l": 1.1, "n": 8123, "o": 2.1, "v": 91234},
    "greeks": {
        "delta": -0.4711,
        "gamma": 0.0421,
        "rho": -0.0188,
        "theta": -0.3087,
        "vega": 0.2934,
    },
    "latestQuote": {
        "ap": 1.91,
        "as": 84,
        "ax": "N",
        "bp": 1.89,
        "bs": 120,
        "bx": "?",
        "c": "A",
        "t": "2026-08-28T18:42:06.567648708Z",
    },
    "latestTrade": {"c": "a", "p": 1.9, "s": 3, "t": "2026-08-28T18:41:58.1Z", "x": "M"},
    "minuteBar": {"c": 1.9, "h": 1.92, "l": 1.88, "n": 41, "o": 1.9, "v": 512},
    "prevDailyBar": {"c": 2.4, "h": 2.8, "l": 1.9, "n": 7010, "o": 2.5, "v": 88010},
}

QUOTED_AT = datetime(2026, 8, 28, 18, 42, 6, 567648, tzinfo=UTC)


class TestChainArgs:
    def test_underlying_is_a_flag_not_a_positional(self):
        """The CLI rejects positional arguments. This bug cost a live run once."""
        args = chain_args("SPY")
        assert "--underlying-symbol" in args
        assert args.index("--underlying-symbol") + 1 == args.index("SPY")
        assert args[:3] == ["data", "option", "chain"]

    def test_underlying_is_upper_cased(self):
        assert "SPY" in chain_args("spy")

    def test_optional_filters_are_omitted_when_unset(self):
        args = chain_args("SPY")
        for flag in ["--expiration-date", "--type", "--strike-price-gte", "--feed"]:
            assert flag not in args

    def test_all_filters_are_passed_through(self):
        args = chain_args(
            "QQQ",
            expiration_date="2026-08-31",
            option_type="put",
            strike_gte=745,
            strike_lte=775.5,
            feed="indicative",
            limit=50,
        )
        pairs = dict(zip(args, args[1:], strict=False))
        assert pairs["--underlying-symbol"] == "QQQ"
        assert pairs["--expiration-date"] == "2026-08-31"
        assert pairs["--type"] == "put"
        assert pairs["--strike-price-gte"] == "745"
        assert pairs["--strike-price-lte"] == "775.5"
        assert pairs["--feed"] == "indicative"
        assert pairs["--limit"] == "50"

    def test_strikes_are_not_rendered_in_scientific_notation(self):
        args = chain_args("SPY", strike_gte=1000000)
        assert "1000000" in args
        assert not any("e+" in a for a in args)


class TestTimestampParsing:
    def test_nanosecond_precision_is_truncated_not_rejected(self):
        assert _parse_timestamp("2026-08-28T18:42:06.567648708Z") == QUOTED_AT

    def test_offset_form_is_understood(self):
        parsed = _parse_timestamp("2026-08-28T14:42:06.567648-04:00")
        assert parsed == QUOTED_AT

    def test_naive_timestamps_are_assumed_utc(self):
        assert _parse_timestamp("2026-08-28T18:42:06").tzinfo is UTC


class TestContractQuote:
    def test_parses_the_live_snapshot(self):
        quote = ContractQuote.from_snapshot("SPY260831P00769000", LIVE_SNAPSHOT)
        assert quote is not None
        assert quote.bid == pytest.approx(1.89)
        assert quote.ask == pytest.approx(1.91)
        assert quote.delta == pytest.approx(-0.4711)
        assert quote.theta == pytest.approx(-0.3087)
        assert quote.quoted_at == QUOTED_AT

    def test_implied_volatility_is_absent_on_this_feed(self):
        """Documented reality, not an aspiration: the chain carries no IV field."""
        quote = ContractQuote.from_snapshot("SPY260831P00769000", LIVE_SNAPSHOT)
        assert quote.implied_volatility is None

    def test_spread_as_a_percentage_of_mid(self):
        quote = ContractQuote.from_snapshot("SPY260831P00769000", LIVE_SNAPSHOT)
        # 0.02 wide on a 1.90 mid is a bit over 1%.
        assert quote.spread_pct_of_mid == pytest.approx(1.0526, abs=1e-3)

    def test_age_is_measured_against_the_quote_timestamp(self):
        quote = ContractQuote.from_snapshot("SPY260831P00769000", LIVE_SNAPSHOT)
        later = datetime(2026, 8, 28, 18, 47, 6, 567648, tzinfo=UTC)
        assert quote.age_seconds(later) == pytest.approx(300.0)

    def test_age_never_goes_negative_on_clock_skew(self):
        quote = ContractQuote.from_snapshot("SPY260831P00769000", LIVE_SNAPSHOT)
        earlier = datetime(2026, 8, 28, 18, 40, 0, tzinfo=UTC)
        assert quote.age_seconds(earlier) == 0.0

    def test_a_contract_with_no_bid_is_not_tradeable(self):
        """Far-OTM strikes come back quoted 0.00 x 0.01 and must be dropped."""
        snapshot = {
            "greeks": {"delta": 0, "gamma": 0, "theta": 0, "vega": 0},
            "latestQuote": {"ap": 0.01, "bp": 0, "t": "2026-08-28T18:42:06.567648708Z"},
        }
        quote = ContractQuote.from_snapshot("SPY260831P00420000", snapshot)
        assert quote is not None
        assert not quote.is_tradeable

    def test_zero_mid_reports_a_hundred_percent_spread_rather_than_dividing_by_zero(self):
        snapshot = {"latestQuote": {"ap": 0, "bp": 0, "t": "2026-08-28T18:42:06Z"}}
        quote = ContractQuote.from_snapshot("SPY260831P00420000", snapshot)
        assert quote.spread_pct_of_mid == 100.0

    @pytest.mark.parametrize(
        "snapshot",
        [
            {},
            {"latestQuote": None},
            {"latestQuote": {"ap": 1.0}},  # no bid
            {"latestQuote": {"ap": 1.0, "bp": 0.9}},  # no timestamp
        ],
    )
    def test_unusable_snapshots_return_none(self, snapshot):
        assert ContractQuote.from_snapshot("SPY260831P00769000", snapshot) is None

    def test_missing_greeks_are_none_not_zero(self):
        """A missing delta must not read as an at-the-money delta of 0."""
        snapshot = {"latestQuote": {"ap": 1.91, "bp": 1.89, "t": "2026-08-28T18:42:06Z"}}
        quote = ContractQuote.from_snapshot("SPY260831P00769000", snapshot)
        assert quote.delta is None
