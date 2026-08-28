"""One-time setup: log the Alpaca CLI in and verify the account can trade.

Reads credentials from .env rather than taking them as arguments, so keys never
land in shell history. Run it once locally after creating the hackathon paper
account; CI does the same thing from GitHub Secrets.

    python scripts/setup_profile.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from onenode.broker.cli import AlpacaCLI, AlpacaCLIError, _default_binary  # noqa: E402


def load_env(path: Path) -> dict[str, str]:
    """Minimal .env reader; no dependency, no surprises about quoting."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def main() -> int:
    env = load_env(REPO_ROOT / ".env")
    key = env.get("ALPACA_API_KEY") or os.environ.get("ALPACA_API_KEY", "")
    secret = env.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY", "")

    if not key or not secret:
        print("ALPACA_API_KEY / ALPACA_SECRET_KEY are empty.")
        print(f"Fill them in {REPO_ROOT / '.env'} and run this again.")
        return 1

    if not key.startswith("PK"):
        print(f"Refusing to continue: key starts with {key[:2]!r}, not 'PK'.")
        print("Paper trading keys begin with PK. This looks like a live key.")
        return 1

    binary = _default_binary()
    print(f"Using CLI at {binary}")

    # Credentials go via argv rather than the shell, so they are not written to
    # any history file. They are still briefly visible to a local process list.
    result = subprocess.run(
        [binary, "profile", "login", "--api-key", "--paper", "--key", key, "--secret", secret],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print("Login failed:")
        print((result.stderr or result.stdout).strip())
        return 1
    print("Logged in.")

    cli = AlpacaCLI(binary=binary)

    try:
        account = cli.account()
    except AlpacaCLIError as exc:
        print(f"Could not read the account: {exc}")
        return 1

    print()
    print(f"  account id     {account.get('id')}")
    print(f"  status         {account.get('status')}")
    print(f"  equity         ${float(account.get('equity', 0)):,.2f}")
    print(f"  options level  {account.get('options_trading_level')}")
    print()

    problems = cli.preflight()

    equity = float(account.get("equity", 0))
    if abs(equity - 100_000) > 1:
        problems.append(
            f"equity is ${equity:,.2f}, but the hackathon requires a $100,000 starting balance"
        )

    configured_id = env.get("ALPACA_ACCOUNT_ID", "")
    if not configured_id:
        print(f"Note: put ALPACA_ACCOUNT_ID={account.get('id')} in .env - the")
        print("submission form needs it so the judges can evaluate your P&L.")
        print()
    elif configured_id != account.get("id"):
        problems.append(
            f"ALPACA_ACCOUNT_ID in .env is {configured_id}, "
            f"but the logged-in account is {account.get('id')}"
        )

    if problems:
        print("NOT READY:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("Ready to trade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
