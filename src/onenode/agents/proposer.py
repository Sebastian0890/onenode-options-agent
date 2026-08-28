"""The Proposer: chooses one trade from the menu, or chooses none.

It is given a list of spreads the code already verified are real, quoted and
tradeable, and it answers with one key from that list. It cannot name a
contract, invent a price, or set a size - those are decided by arithmetic
elsewhere. What it contributes is the judgement that does not reduce to a
filter: whether today is a day to sell premium at all, and which of thirty
near-identical spreads best fits the tape.

Standing aside is a first-class answer and the prompt says so. An agent that
must trade every time it is asked will trade badly.
"""

from __future__ import annotations

import os
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from ..strategy import IronCondorCandidate, SpreadCandidate

Candidate = SpreadCandidate | IronCondorCandidate

MODEL = "claude-opus-5"
MAX_CANDIDATES = 25

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
anticipate or argue with it - choose the trade you actually think is best."""


class ProposerDecision(BaseModel):
    """What the Proposer decided, and why."""

    action: Literal["trade", "stand_aside"]
    candidate_key: str = Field(
        default="",
        description="Exact key of the chosen candidate; empty when standing aside.",
    )
    rationale: str = Field(
        description="Two or three sentences: why this trade, or why nothing today."
    )


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
) -> str:
    """Assemble the market picture and the menu into one message."""
    menu = "\n".join(f"  {candidate.describe()}" for candidate in candidates[:MAX_CANDIDATES])
    return f"""Market
  {underlying} spot: {spot:,.2f}
  minutes to close: {minutes_to_close:.0f}

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
    client: anthropic.Anthropic | None = None,
) -> ProposerDecision:
    """Ask the Proposer for one trade, or for a pass.

    A key that is not on the menu is treated as standing aside rather than as
    something to correct. If the model cannot pick from a list of twenty-five
    strings, the right response is to skip this window, not to guess at what it
    meant and place a trade on the guess.
    """
    if not candidates:
        return ProposerDecision(
            action="stand_aside",
            rationale="No candidate spread passed the structural filters.",
        )

    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set; the proposer cannot run")
        client = anthropic.Anthropic()

    response = client.messages.parse(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": build_prompt(
                    candidates,
                    underlying=underlying,
                    spot=spot,
                    day_pnl_pct=day_pnl_pct,
                    equity=equity,
                    open_positions=open_positions,
                    committed_risk=committed_risk,
                    minutes_to_close=minutes_to_close,
                ),
            }
        ],
        output_format=ProposerDecision,
    )

    decision = response.parsed_output
    if decision.action == "trade":
        offered = {candidate.key for candidate in candidates[:MAX_CANDIDATES]}
        if decision.candidate_key not in offered:
            return ProposerDecision(
                action="stand_aside",
                rationale=(
                    f"Proposer returned {decision.candidate_key!r}, which was not on the "
                    f"menu. Standing aside rather than guessing at the intent."
                ),
            )
    return decision


def find_candidate(candidates: list[Candidate], key: str) -> Candidate | None:
    return next((candidate for candidate in candidates if candidate.key == key), None)
