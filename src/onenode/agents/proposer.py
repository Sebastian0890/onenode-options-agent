"""The Proposer: chooses one trade from the menu, or chooses none.

It is given a list of spreads the code already verified are real, quoted and
tradeable, and it answers with one key from that list. It cannot name a
contract, invent a price, or set a size - those are decided by arithmetic
elsewhere. What it contributes is the judgement that does not reduce to a
filter: whether today is a day to sell premium at all, and which of thirty
near-identical spreads best fits the tape.

Standing aside is a first-class answer and the prompt says so. An agent that
must trade every time it is asked will trade badly.

Which model answers is decided in ``llm.py`` from whichever keys are present.
The decision carries the answering model's name, so the journal records who
actually chose the trade rather than who was supposed to.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ..strategy import MAX_MENU, IronCondorCandidate, SpreadCandidate
from . import llm

Candidate = SpreadCandidate | IronCondorCandidate

MAX_CANDIDATES = MAX_MENU
ROLE_ENV = "ONENODE_PROPOSER_PROVIDER"

SYSTEM = """You select credit spreads for an autonomous options agent trading \
a $100,000 paper account during a one-week competition.

The strategy is short premium with defined risk: sell an out-of-the-money \
vertical credit spread on a liquid index ETF and let time decay work. The edge \
is win rate, not size of win. Over four trading days, finishing modestly green \
beats a large gain that risked the account.

You are given candidates that are already verified as real, quoted and \
tradeable, priced at what a marketable order would actually collect. You choose \
one by its exact key, or you stand aside.

Stand aside when the tape argues against selling premium - a sharp directional \
move against the short side, or credits so thin the risk is not paid for. \
Standing aside is a good answer, not a failure. There will be another window in \
fifteen minutes.

You do not set position size and you do not place orders. A deterministic risk \
gate sizes the trade and can reject your choice outright. Do not try to \
anticipate or argue with it - choose the trade you actually think is best.

Answer with JSON only, no prose and no code fences:
{"action": "trade" or "stand_aside", "candidate_key": "exact key from the list, \
or empty string", "rationale": "two or three sentences"}"""


class ProposerDecision(BaseModel):
    """What the Proposer decided, and why."""

    action: Literal["trade", "stand_aside"]
    candidate_key: str = Field(
        default="",
        description="Exact key of the chosen candidate; empty when standing aside.",
    )
    rationale: str = Field(
        default="",
        description="Two or three sentences: why this trade, or why nothing today.",
    )
    proposer: str = Field(default="", description="Backend that produced this decision.")
    family: str = Field(default="", description="Model lineage, for reviewer independence.")


def build_prompt(
    candidates: list[Candidate],
    *,
    underlying: str,
    spot: float,
    day_pnl_pct: float,
    equity: float,
    open_positions: int,
    committed_risk: float,
    minutes_to_close: float,
    regime: str = "",
) -> str:
    """Assemble the market picture and the menu into one message."""
    menu = "\n".join(f"  {candidate.describe()}" for candidate in candidates[:MAX_CANDIDATES])
    return f"""Market
  {underlying} spot: {spot:,.2f}
  minutes to close: {minutes_to_close:.0f}
  {regime or "regime not assessed"}

Account
  equity: ${equity:,.2f}
  day P&L: {day_pnl_pct:+.2f}%
  open structures: {open_positions}
  risk already committed: ${committed_risk:,.2f}

Candidates ({len(candidates)} found, showing {min(len(candidates), MAX_CANDIDATES)})
{menu}

Choose one candidate key from the list above, or stand aside."""


def propose_trade(
    candidates: list[Candidate],
    *,
    underlying: str,
    spot: float,
    day_pnl_pct: float,
    equity: float,
    open_positions: int,
    committed_risk: float,
    minutes_to_close: float,
    regime: str = "",
    ask=llm.ask,
) -> ProposerDecision:
    """Ask the Proposer for one trade, or for a pass.

    A key that is not on the menu is treated as standing aside rather than as
    something to correct. If the model cannot pick from a list of twenty-five
    strings, the right response is to skip this window, not to guess at what it
    meant and place a trade on the guess. A reply that does not fit the expected
    shape is read the same way, for the same reason.
    """
    if not candidates:
        return ProposerDecision(
            action="stand_aside",
            rationale="No candidate spread passed the structural filters.",
        )

    reply = ask(
        system=SYSTEM,
        user=build_prompt(
            candidates,
            underlying=underlying,
            spot=spot,
            day_pnl_pct=day_pnl_pct,
            equity=equity,
            open_positions=open_positions,
            committed_risk=committed_risk,
            minutes_to_close=minutes_to_close,
            regime=regime,
        ),
        role_env=ROLE_ENV,
        max_tokens=1024,
    )

    try:
        decision = ProposerDecision.model_validate(
            {**reply.payload, "proposer": reply.label, "family": reply.family}
        )
    except ValidationError as exc:
        return ProposerDecision(
            action="stand_aside",
            rationale=(
                f"Proposer reply did not fit the expected shape ({exc.error_count()} errors)."
            ),
            proposer=reply.label,
            family=reply.family,
        )

    if decision.action == "trade":
        offered = {candidate.key for candidate in candidates[:MAX_CANDIDATES]}
        if decision.candidate_key not in offered:
            return ProposerDecision(
                action="stand_aside",
                rationale=(
                    f"Proposer returned {decision.candidate_key!r}, which was not on the "
                    f"menu. Standing aside rather than guessing at the intent."
                ),
                proposer=reply.label,
                family=reply.family,
            )
    return decision


def find_candidate(candidates: list[Candidate], key: str) -> Candidate | None:
    return next((candidate for candidate in candidates if candidate.key == key), None)
