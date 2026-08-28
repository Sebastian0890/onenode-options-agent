"""Broker access, exclusively through Alpaca's CLI.

Every call to Alpaca in this agent goes through ``alpaca`` the command-line
tool rather than an SDK. That is a deliberate fit to the shape of the problem:
the agent runs as a scheduled job that wakes up, looks around, acts once and
exits, which is exactly the long-running-agent-session case the CLI was built
for. It also means every action the agent takes is a command a human can paste
into a terminal and reproduce.
"""

from .cli import AlpacaCLI, AlpacaCLIError, ContractQuote

__all__ = ["AlpacaCLI", "AlpacaCLIError", "ContractQuote"]
