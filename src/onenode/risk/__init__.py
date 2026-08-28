"""Risk layer: the deterministic core that decides what the agent is allowed to do.

Nothing in this package calls a language model or the network. Everything here
is a pure function of its inputs, which is what makes the agent's risk policy
something you can test rather than something you have to trust.
"""

from .gate import evaluate
from .limits import DEFAULT_LIMITS, RiskLimits
from .models import (
    AccountSnapshot,
    GateDecision,
    MarketClock,
    OptionLeg,
    ProposedTrade,
    Right,
    Side,
)
from .payoff import max_gain, worst_case_loss

__all__ = [
    "DEFAULT_LIMITS",
    "AccountSnapshot",
    "GateDecision",
    "MarketClock",
    "OptionLeg",
    "ProposedTrade",
    "Right",
    "RiskLimits",
    "Side",
    "evaluate",
    "max_gain",
    "worst_case_loss",
]
