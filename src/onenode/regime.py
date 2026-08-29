"""Which side of the market it is currently dangerous to sell.

A put credit spread is a bet that the market will not fall much. Sold into a
market that is already falling, it is the same bet at worse odds, and the delta
that made it look safe was computed before the move. Every account that has been
destroyed selling premium was destroyed this way: not by one bad trade, but by
taking the same side repeatedly while the tape ran against it.

The states are the ones a Markov regime model uses, and they are deliberately
crude: cumulative return over a trailing twenty sessions, above +5% is bull,
below -5% is bear, anything else neutral. Crude is the point. A threshold with
three decimal places is a threshold fitted to the past.

**What this does and does not claim.** The transition matrix here counts what
followed what in the sessions it was given. It is a base rate, not a forecast,
and two things keep it from being one:

* Consecutive windows overlap by nineteen of their twenty sessions, so the
  state sequence is heavily autocorrelated. The matrix mostly measures how long
  regimes persist. That is a real property and a useful one, but the counts
  behind it are nothing like independent observations, and treating a cell of
  it as a probability with error bars would be wrong.
* It is fitted on the same history it reports on. Nothing here is held out.

So the matrix is carried as *context* - reported to the Proposer and written to
the journal, with its observation counts attached so a cell backed by four
sessions cannot pass for one backed by four hundred. The rule that actually
gates a trade needs none of it:

    In a bear regime the agent does not sell puts. In a bull regime it does not
    sell calls.

That rule has no fitted parameter to be wrong about. It refuses the side the
market is currently moving against, which is the mistake worth being protected
from, and it leaves the other side available so the agent does not simply stop.

When the data is unavailable the filter does nothing and says so. It is a
restriction layered on a system that is already safe without it - the hard gate,
the daily stop and the sizing budget do not depend on it - so an outage should
cost the refinement, not the session.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .risk.models import Right

BULL = "bull"
NEUTRAL = "neutral"
BEAR = "bear"
UNKNOWN = "unknown"

STATES: tuple[str, ...] = (BULL, NEUTRAL, BEAR)

WINDOW_SESSIONS = 20
THRESHOLD_PCT = 5.0


@dataclass(frozen=True)
class DailyBar:
    """One session's close. Nothing else here needs the rest of the bar."""

    day: date
    close: float


def classify(window_return_pct: float, threshold_pct: float = THRESHOLD_PCT) -> str:
    """Which regime a trailing return puts the market in."""
    if window_return_pct >= threshold_pct:
        return BULL
    if window_return_pct <= -threshold_pct:
        return BEAR
    return NEUTRAL


def window_returns(closes: list[float], window: int = WINDOW_SESSIONS) -> list[float]:
    """Cumulative percentage return over each trailing window, oldest first."""
    if len(closes) <= window:
        return []
    out: list[float] = []
    for index in range(window, len(closes)):
        start = closes[index - window]
        if start <= 0:
            continue
        out.append(100.0 * (closes[index] - start) / start)
    return out


def state_series(
    closes: list[float],
    window: int = WINDOW_SESSIONS,
    threshold_pct: float = THRESHOLD_PCT,
) -> list[str]:
    """The regime on each session for which a full trailing window exists."""
    return [classify(value, threshold_pct) for value in window_returns(closes, window)]


@dataclass(frozen=True)
class TransitionMatrix:
    """What followed what, counted rather than estimated."""

    counts: dict[str, dict[str, int]]

    @classmethod
    def from_states(cls, states: list[str]) -> TransitionMatrix:
        counts = {origin: dict.fromkeys(STATES, 0) for origin in STATES}
        for origin, destination in zip(states, states[1:], strict=False):
            if origin in counts and destination in counts[origin]:
                counts[origin][destination] += 1
        return cls(counts=counts)

    def observations(self, state: str) -> int:
        """How many transitions the row for ``state`` is built from.

        Carried everywhere the probabilities go. A row with four observations
        and a row with four hundred look identical once they are normalised,
        and only one of them is worth anything.
        """
        return sum(self.counts.get(state, {}).values())

    def row(self, state: str) -> dict[str, float]:
        """One step ahead from ``state``.

        An unobserved row returns the uniform distribution rather than zeros:
        with no evidence, "equally likely" is the honest answer, and it also
        keeps the matrix multiplication below well-formed.
        """
        total = self.observations(state)
        if total == 0:
            return dict.fromkeys(STATES, 1.0 / len(STATES))
        return {destination: self.counts[state][destination] / total for destination in STATES}

    def after(self, state: str, sessions: int) -> dict[str, float]:
        """Distribution over regimes ``sessions`` steps ahead.

        The k-step distribution is the start row times the matrix k times. The
        agent trades 0-7 DTE, so the horizon that matters is the one the
        candidate actually has, not a fixed guess.
        """
        distribution = dict.fromkeys(STATES, 0.0)
        distribution[state] = 1.0
        for _ in range(max(0, sessions)):
            stepped = dict.fromkeys(STATES, 0.0)
            for origin, mass in distribution.items():
                if mass == 0.0:
                    continue
                for destination, probability in self.row(origin).items():
                    stepped[destination] += mass * probability
            distribution = stepped
        return distribution


@dataclass(frozen=True)
class Regime:
    """The current regime for one underlying, and the history behind it."""

    underlying: str
    state: str
    window_return_pct: float
    matrix: TransitionMatrix
    sessions: int
    """Sessions of history the matrix was built from."""

    @classmethod
    def unavailable(cls, underlying: str) -> Regime:
        return cls(
            underlying=underlying,
            state=UNKNOWN,
            window_return_pct=0.0,
            matrix=TransitionMatrix.from_states([]),
            sessions=0,
        )

    @property
    def known(self) -> bool:
        return self.state in STATES

    def blocks(self, right: Right) -> bool:
        """Whether this regime forbids selling that side.

        Puts are refused in a bear regime and calls in a bull one. An unknown
        regime forbids nothing: this filter narrows a system that is already
        safe without it, so losing the data should cost the narrowing rather
        than the session.
        """
        if not self.known:
            return False
        return (right is Right.PUT and self.state == BEAR) or (
            right is Right.CALL and self.state == BULL
        )

    def outlook(self, sessions: int) -> dict[str, float]:
        """Where the regime has historically gone over that many sessions."""
        if not self.known:
            return dict.fromkeys(STATES, 1.0 / len(STATES))
        return self.matrix.after(self.state, sessions)

    def describe(self, sessions: int = 5) -> str:
        """One line for the Proposer's prompt and for the journal."""
        if not self.known:
            return f"{self.underlying}: regime unknown (no history available)"
        ahead = self.outlook(sessions)
        observations = self.matrix.observations(self.state)
        return (
            f"{self.underlying}: {self.state} regime "
            f"({self.window_return_pct:+.1f}% over {WINDOW_SESSIONS} sessions) | "
            f"after {sessions} sessions historically: "
            f"bull {ahead[BULL]:.0%} / neutral {ahead[NEUTRAL]:.0%} / bear {ahead[BEAR]:.0%} "
            f"({observations} transitions from this state in {self.sessions} sessions)"
        )


def from_bars(
    underlying: str,
    bars: list[DailyBar],
    *,
    window: int = WINDOW_SESSIONS,
    threshold_pct: float = THRESHOLD_PCT,
) -> Regime:
    """Build the regime picture from a run of daily closes, oldest first."""
    closes = [bar.close for bar in bars]
    returns = window_returns(closes, window)
    if not returns:
        return Regime.unavailable(underlying)

    states = [classify(value, threshold_pct) for value in returns]
    return Regime(
        underlying=underlying,
        state=states[-1],
        window_return_pct=returns[-1],
        matrix=TransitionMatrix.from_states(states),
        sessions=len(states),
    )
