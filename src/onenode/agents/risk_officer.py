"""The Risk Officer: an independent reviewer whose only power is to say no.

Two things make this more than theatre. It runs on a different model family
from the Proposer, so it does not inherit the same blind spots. And it never
sees the Proposer's rationale - only the trade and the account - so it forms
its own view instead of grading an argument it has already been persuaded by.

Its prompt is adversarial by construction: find the reason not to, and refuse
when uncertain. That asymmetry is deliberate. A reviewer that splits the
difference approves everything eventually.

The default backend is Featherless (open-weights models, OpenAI-compatible
HTTP). Without that key it falls back to Claude with an isolated context, which
keeps the independence of context but loses the independence of model family -
so the fallback is reported, never silent.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from pydantic import BaseModel, Field

from ..risk.models import ProposedTrade
from ..strategy import SpreadCandidate

FEATHERLESS_URL = "https://api.featherless.ai/v1/chat/completions"
FEATHERLESS_MODEL = os.environ.get("FEATHERLESS_MODEL", "mistralai/Mistral-Small-24B-Instruct-2501")
FALLBACK_MODEL = "claude-opus-5"

SYSTEM = """You are the risk officer for an autonomous options trading agent \
on a $100,000 paper account. A trade has been proposed. Your job is to find the \
reason it should not be placed.

You are not asked whether the trade is reasonable. You are asked what is wrong \
with it. Look for: the short strike sitting where the underlying could \
plausibly reach before expiry, credit too thin for the risk taken, an account \
already carrying more exposure than a new position should be added to, timing \
too near expiry for the position to be managed.

If you are uncertain, reject. A window missed costs nothing; a bad position \
held over a weekend costs real money. There will be another candidate in \
fifteen minutes.

Approve only when you genuinely cannot find a substantive objection.

Answer with JSON only: {"approve": true or false, "reason": "one or two sentences"}"""


class RiskVerdict(BaseModel):
    """The reviewer's answer, plus which backend actually produced it."""

    approve: bool
    reason: str
    reviewer: str = Field(default="", description="Backend that produced this verdict.")
    degraded: bool = Field(
        default=False,
        description="True when the reviewer shares a model family with the proposer.",
    )


def build_prompt(
    candidate: SpreadCandidate,
    trade: ProposedTrade,
    *,
    spot: float,
    equity: float,
    day_pnl_pct: float,
    open_positions: int,
    committed_risk: float,
    minutes_to_close: float,
    worst_case_loss: float,
) -> str:
    """Describe the trade and the account - and nothing about why it was chosen."""
    distance = abs(spot - candidate.short_leg.strike)
    return f"""Proposed trade
  {candidate.right.value} credit spread on {candidate.underlying}
  short {candidate.short_leg.strike:g} / long {candidate.long_leg.strike:g}, \
width ${candidate.width:g}
  expiry {candidate.expiry}
  contracts: {trade.contracts}
  credit collected: ${trade.net_cash:,.2f}
  worst case loss: ${worst_case_loss:,.2f}
  short strike delta: {candidate.short_delta:.3f}
  distance from spot to short strike: {distance:,.2f} ({100 * distance / spot:.2f}%)

Market
  {candidate.underlying} spot: {spot:,.2f}
  minutes to close: {minutes_to_close:.0f}

Account
  equity: ${equity:,.2f}
  day P&L: {day_pnl_pct:+.2f}%
  open structures: {open_positions}
  risk already committed: ${committed_risk:,.2f}

Should this trade be placed?"""


def _parse_verdict(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a reply that may be wrapped in prose or fences."""
    cleaned = text.strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in reviewer reply: {cleaned[:200]}")
    return json.loads(cleaned[start : end + 1])


def _review_via_featherless(prompt: str, api_key: str, timeout: float) -> RiskVerdict:
    response = httpx.post(
        FEATHERLESS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": FEATHERLESS_MODEL,
            "max_tokens": 500,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    payload = _parse_verdict(text)
    return RiskVerdict(
        approve=bool(payload.get("approve", False)),
        reason=str(payload.get("reason", "")).strip() or "No reason given.",
        reviewer=f"featherless:{FEATHERLESS_MODEL}",
    )


def _review_via_claude(prompt: str) -> RiskVerdict:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=FALLBACK_MODEL,
        max_tokens=2048,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_format=RiskVerdict,
    )
    verdict = response.parsed_output
    return RiskVerdict(
        approve=verdict.approve,
        reason=verdict.reason,
        reviewer=f"anthropic:{FALLBACK_MODEL}",
        degraded=True,
    )


def review_trade(
    candidate: SpreadCandidate,
    trade: ProposedTrade,
    *,
    spot: float,
    equity: float,
    day_pnl_pct: float,
    open_positions: int,
    committed_risk: float,
    minutes_to_close: float,
    worst_case_loss: float,
    timeout: float = 40.0,
) -> RiskVerdict:
    """Get an independent verdict on a proposed trade.

    Any failure of the reviewer is a veto, not a pass. If the second opinion
    cannot be obtained, the trade does not happen - an unreviewed trade is
    exactly what this layer exists to prevent.
    """
    prompt = build_prompt(
        candidate,
        trade,
        spot=spot,
        equity=equity,
        day_pnl_pct=day_pnl_pct,
        open_positions=open_positions,
        committed_risk=committed_risk,
        minutes_to_close=minutes_to_close,
        worst_case_loss=worst_case_loss,
    )

    featherless_key = os.environ.get("FEATHERLESS_API_KEY", "").strip()
    if featherless_key:
        try:
            return _review_via_featherless(prompt, featherless_key, timeout)
        except Exception as exc:  # noqa: BLE001 - any failure must become a veto
            if not os.environ.get("ANTHROPIC_API_KEY"):
                return RiskVerdict(
                    approve=False,
                    reason=f"Reviewer unreachable and no fallback configured: {exc}",
                    reviewer="unavailable",
                )
            try:
                return _review_via_claude(prompt)
            except Exception as fallback_exc:  # noqa: BLE001
                return RiskVerdict(
                    approve=False,
                    reason=f"Both reviewers failed: {exc} / {fallback_exc}",
                    reviewer="unavailable",
                )

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _review_via_claude(prompt)
        except Exception as exc:  # noqa: BLE001
            return RiskVerdict(
                approve=False,
                reason=f"Reviewer failed: {exc}",
                reviewer="unavailable",
            )

    return RiskVerdict(
        approve=False,
        reason="No reviewer configured. A trade without a second opinion is not placed.",
        reviewer="unavailable",
    )
