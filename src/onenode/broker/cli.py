"""A typed wrapper around the ``alpaca`` command-line tool.

The CLI emits JSON on stdout, which makes it a clean subprocess boundary: this
module turns that JSON into the frozen models the risk layer understands and
refuses anything it cannot parse, rather than passing half-understood dicts
deeper into the agent.

Alpaca returns every numeric field as a string. All of them go through
:func:`_as_float` so a missing or malformed value fails loudly here instead of
becoming a silent zero inside a risk calculation.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from ..regime import DailyBar
from ..risk.models import MarketClock

DEFAULT_TIMEOUT_SECONDS = 45


class AlpacaCLIError(RuntimeError):
    """The CLI exited non-zero, timed out, or produced output we cannot read."""


def _parse_timestamp(value: str) -> datetime:
    """Parse an Alpaca RFC-3339 timestamp, tolerating nanosecond precision."""
    text = value.strip().replace("Z", "+00:00")
    # datetime.fromisoformat accepts at most microseconds; Alpaca sometimes
    # sends nine fractional digits.
    text = re.sub(r"\.(\d{6})\d+", r".\1", text)
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _as_float(payload: dict[str, Any], *keys: str) -> float:
    """Read the first present key and coerce it to float, or raise."""
    for key in keys:
        if key in payload and payload[key] is not None:
            try:
                return float(payload[key])
            except (TypeError, ValueError) as exc:
                raise AlpacaCLIError(f"field {key!r} is not numeric: {payload[key]!r}") from exc
    raise AlpacaCLIError(f"none of {keys} present in payload with keys {sorted(payload)}")


def _opt_float(payload: dict[str, Any], *keys: str) -> float | None:
    try:
        return _as_float(payload, *keys)
    except AlpacaCLIError:
        return None


@dataclass(frozen=True)
class ContractQuote:
    """One option contract as the agent sees it: price, spread, age, greeks.

    ``age_seconds`` and ``spread_pct_of_mid`` exist because the free data plan
    serves an indicative feed rather than full OPRA. They are what the hard
    gate uses to refuse trading on a quote it should not trust.
    """

    symbol: str
    bid: float
    ask: float
    quoted_at: datetime
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    implied_volatility: float | None = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct_of_mid(self) -> float:
        """Bid-ask spread as a percentage of mid; 100.0 when mid is zero."""
        mid = self.mid
        if mid <= 0:
            return 100.0
        return 100.0 * self.spread / mid

    def age_seconds(self, now: datetime | None = None) -> float:
        now = now or datetime.now(UTC)
        return max((now - self.quoted_at).total_seconds(), 0.0)

    @property
    def is_tradeable(self) -> bool:
        """A quote with no bid cannot be sold into and no ask cannot be bought."""
        return self.bid > 0 and self.ask > 0 and self.ask >= self.bid

    @classmethod
    def from_snapshot(cls, symbol: str, snapshot: dict[str, Any]) -> ContractQuote | None:
        """Build from one entry of a ``data option chain`` response.

        Returns ``None`` when the snapshot carries no usable quote, which is
        common for illiquid strikes. Key spellings are tried in both camelCase
        and snake_case because the field naming is not part of the CLI's
        published schema.
        """
        quote = snapshot.get("latestQuote") or snapshot.get("latest_quote")
        if not isinstance(quote, dict):
            return None

        bid = _opt_float(quote, "bp", "bid_price", "bidPrice")
        ask = _opt_float(quote, "ap", "ask_price", "askPrice")
        if bid is None or ask is None:
            return None

        raw_ts = quote.get("t") or quote.get("timestamp")
        if not isinstance(raw_ts, str):
            return None

        greeks = snapshot.get("greeks") or {}
        if not isinstance(greeks, dict):
            greeks = {}

        return cls(
            symbol=symbol,
            bid=bid,
            ask=ask,
            quoted_at=_parse_timestamp(raw_ts),
            delta=_opt_float(greeks, "delta"),
            gamma=_opt_float(greeks, "gamma"),
            theta=_opt_float(greeks, "theta"),
            vega=_opt_float(greeks, "vega"),
            # Verified against the live chain on 2026-08-28: the snapshot carries
            # dailyBar, greeks, latestQuote, latestTrade, minuteBar and
            # prevDailyBar, and no implied-volatility field at all. Kept optional
            # rather than removed in case a paid feed supplies it; strike
            # selection runs on delta, which is populated near the money.
            implied_volatility=_opt_float(
                snapshot, "impliedVolatility", "implied_volatility", "iv"
            ),
        )


def _format_number(value: float) -> str:
    """Render a number for the command line without scientific notation.

    ``f"{x:g}"`` looks right until the value reaches a million, at which point
    it emits ``1e+06`` and the CLI rejects it. Strikes never get that large,
    but a formatter that quietly breaks on large input is not one to keep.
    """
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def chain_args(
    underlying: str,
    *,
    expiration_date: str | None = None,
    option_type: str | None = None,
    strike_gte: float | None = None,
    strike_lte: float | None = None,
    feed: str | None = None,
    limit: int = 200,
) -> list[str]:
    """Build the argument list for ``alpaca data option chain``.

    Split out as a pure function because of a bug worth not repeating: the
    underlying was first passed positionally, which the CLI rejects. Every
    command in this tool takes flags and never positional arguments, and that
    convention is now pinned by a test rather than by memory.
    """
    args = ["data", "option", "chain", "--underlying-symbol", underlying.upper()]
    if expiration_date:
        args += ["--expiration-date", expiration_date]
    if option_type:
        args += ["--type", option_type]
    if strike_gte is not None:
        args += ["--strike-price-gte", _format_number(strike_gte)]
    if strike_lte is not None:
        args += ["--strike-price-lte", _format_number(strike_lte)]
    if feed:
        args += ["--feed", feed]
    args += ["--limit", str(limit)]
    return args


def _default_binary() -> str:
    """Locate the CLI: explicit override, vendored copy, then PATH."""
    override = os.environ.get("ALPACA_CLI_PATH")
    if override:
        return override

    name = "alpaca.exe" if sys.platform == "win32" else "alpaca"
    vendored = Path(__file__).resolve().parents[3] / "bin" / name
    if vendored.exists():
        return str(vendored)

    found = shutil.which("alpaca")
    if found:
        return found

    raise AlpacaCLIError(
        "alpaca CLI not found. Set ALPACA_CLI_PATH, put the binary in ./bin, "
        "or install it from https://github.com/alpacahq/cli/releases"
    )


class AlpacaCLI:
    """Thin, typed access to the Alpaca CLI.

    Holds no credentials of its own: the CLI reads them from the profile
    created by ``alpaca profile login --api-key``, so keys never pass through
    this process or appear in an argument list a process table could show.
    """

    def __init__(
        self,
        binary: str | None = None,
        profile: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.binary = binary or _default_binary()
        self.profile = profile
        self.timeout = timeout

    def _run(self, *args: str) -> Any:
        command = [self.binary, *args, "--quiet"]
        if self.profile:
            command += ["--profile", self.profile]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AlpacaCLIError(
                f"`alpaca {' '.join(args)}` timed out after {self.timeout}s"
            ) from exc

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise AlpacaCLIError(
                f"`alpaca {' '.join(args)}` exited {completed.returncode}: {detail}"
            )

        stdout = completed.stdout.strip()
        if not stdout:
            raise AlpacaCLIError(f"`alpaca {' '.join(args)}` produced no output")

        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise AlpacaCLIError(
                f"`alpaca {' '.join(args)}` did not return JSON: {stdout[:200]}"
            ) from exc

    # --- Session ---------------------------------------------------------

    def clock(self, now: datetime | None = None) -> MarketClock:
        """Market state from the broker, so holidays need no calendar of ours."""
        payload = self._run("clock")
        is_open = bool(payload.get("is_open"))
        next_close = _parse_timestamp(payload["next_close"])
        reference = now or _parse_timestamp(payload["timestamp"])

        minutes = (next_close - reference).total_seconds() / 60.0
        return MarketClock(is_open=is_open, minutes_to_close=minutes if is_open else 0.0)

    # --- Account ---------------------------------------------------------

    def account(self) -> dict[str, Any]:
        return self._run("account", "get")

    def preflight(self) -> list[str]:
        """Check the account can actually do what the strategy requires.

        Returns a list of problems; empty means good to trade. Called once at
        the start of every run so a misconfigured account fails at startup
        rather than at order submission.
        """
        problems: list[str] = []
        account = self.account()

        if account.get("status") != "ACTIVE":
            problems.append(f"account status is {account.get('status')!r}, expected ACTIVE")
        if account.get("trading_blocked"):
            problems.append("trading is blocked on this account")
        if account.get("account_blocked"):
            problems.append("account is blocked")
        if account.get("trade_suspended_by_user"):
            problems.append("trading is suspended by user setting")

        level = str(account.get("options_trading_level", "0"))
        if level != "3":
            problems.append(
                f"options trading level is {level}, but multi-leg defined-risk "
                "structures need level 3"
            )

        return problems

    def positions(self) -> list[dict[str, Any]]:
        payload = self._run("position", "list")
        return payload if isinstance(payload, list) else []

    # --- Market data -----------------------------------------------------

    def latest_price(self, symbol: str) -> float:
        """Last traded price of an underlying, used to centre the strike band."""
        payload = self._run("data", "latest-trade", "--symbol", symbol)
        trade = payload.get("trade")
        if not isinstance(trade, dict):
            raise AlpacaCLIError(f"no trade in latest-trade response for {symbol}")
        return _as_float(trade, "p", "price")

    def daily_bars(self, symbol: str, sessions: int = 260) -> list[DailyBar]:
        """Daily closes, oldest first, for classifying the regime.

        ``sessions`` counts trading days, not calendar days, so the window
        asked for is widened by the ratio between them - roughly 252 sessions
        to 365 days - plus a margin for holidays. Asking for too many is free;
        asking for too few silently shortens the history the matrix is built
        from, which is the failure that would not announce itself.
        """
        start = (date.today() - timedelta(days=int(sessions * 1.5) + 30)).isoformat()
        payload = self._run(
            "data",
            "bars",
            "--symbol",
            symbol,
            "--timeframe",
            "1Day",
            "--start",
            start,
            "--limit",
            str(sessions + 60),
            "--sort",
            "asc",
        )
        rows = payload.get("bars")
        if not isinstance(rows, list):
            raise AlpacaCLIError(f"no bars in response for {symbol}")

        bars: list[DailyBar] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            stamp = row.get("t")
            if not isinstance(stamp, str):
                continue
            bars.append(
                DailyBar(day=_parse_timestamp(stamp).date(), close=_as_float(row, "c", "close"))
            )
        return bars

    def option_chain(
        self,
        underlying: str,
        *,
        expiration_date: str | None = None,
        option_type: str | None = None,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
        feed: str | None = None,
        limit: int = 200,
    ) -> dict[str, ContractQuote]:
        """Fetch a chain and keep only the contracts that can actually be traded.

        Strikes with no bid or no ask are dropped here rather than handed to
        the model, so it never spends reasoning on a contract it could not fill.
        """
        args = chain_args(
            underlying,
            expiration_date=expiration_date,
            option_type=option_type,
            strike_gte=strike_gte,
            strike_lte=strike_lte,
            feed=feed,
            limit=limit,
        )
        payload = self._run(*args)
        snapshots = payload.get("snapshots") or {}

        quotes: dict[str, ContractQuote] = {}
        for symbol, snapshot in snapshots.items():
            if not isinstance(snapshot, dict):
                continue
            quote = ContractQuote.from_snapshot(symbol, snapshot)
            if quote is not None and quote.is_tradeable:
                quotes[symbol] = quote
        return quotes
