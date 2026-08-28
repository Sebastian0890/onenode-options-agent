"""The trade journal: an append-only record of what the agent did and why.

Every run writes here, including the runs where nothing happened. That is the
point. An agent that only logs its trades looks decisive in hindsight; one that
also logs the trades it rejected, and the reason, can be argued with.

The file is JSONL so it appends safely and diffs cleanly, and it is committed
back to the repository after each run, which makes the whole decision history
public and timestamped by something other than ourselves.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

DEFAULT_JOURNAL = Path("journal/trades.jsonl")


def _encode(value: Any) -> Any:
    """Make agent objects JSON-safe without teaching every one of them to serialise."""
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _encode(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_encode(item) for item in value]
    if isinstance(value, float | int | str | bool) or value is None:
        return value
    return str(value)


class Journal:
    """Append-only event log for one agent run after another."""

    def __init__(self, path: Path | str = DEFAULT_JOURNAL, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _ends_without_newline(self) -> bool:
        """True if a previous write was cut off mid-line.

        A scheduled job can be cancelled or run out of disk between writing a
        line and writing its newline. Appending straight onto that stub would
        splice the next entry onto the broken one and lose both, so the damage
        has to be confined to the line that was actually truncated.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return False
        with self.path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) != b"\n"

    def record(self, event: str, **data: Any) -> dict[str, Any]:
        """Append one event. Returns what was written, for logging to stdout."""
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "event": event,
            **{key: _encode(value) for key, value in data.items()},
        }
        repair = self._ends_without_newline()
        with self.path.open("a", encoding="utf-8") as handle:
            if repair:
                handle.write("\n")
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def entries(self) -> list[dict[str, Any]]:
        """Read the whole journal, skipping any line that got truncated."""
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows


def render_markdown(entries: list[dict[str, Any]], limit: int = 60) -> str:
    """A human-readable digest of the journal, for the repo and for the demo.

    Ordered newest first, because the question anyone actually arrives with is
    "what did it just do", not "how did the week begin".
    """
    lines = [
        "# Trade journal",
        "",
        "Auto-generated from `journal/trades.jsonl` after each agent run.",
        "Paper trading only - no real capital at any point.",
        "",
    ]

    orders = [e for e in entries if e.get("event") == "order_placed"]
    blocked = [e for e in entries if e.get("event") == "gate_blocked"]
    vetoed = [e for e in entries if e.get("event") == "risk_officer_veto"]
    runs = {e.get("run_id") for e in entries if e.get("run_id")}

    lines += [
        "| | |",
        "|---|---|",
        f"| Runs | {len(runs)} |",
        f"| Orders placed | {len(orders)} |",
        f"| Blocked by the hard gate | {len(blocked)} |",
        f"| Vetoed by the risk officer | {len(vetoed)} |",
        "",
        "## Recent activity",
        "",
    ]

    # Failures belong here as much as trades do. A run that stopped because a
    # credential was missing is the most informative thing that can happen, and
    # leaving it out would make the agent look like it simply had nothing to say.
    # Keep in sync with INTERESTING in index.html.
    interesting = {
        "order_placed",
        "order_failed",
        "gate_blocked",
        "gate_approved",
        "risk_officer_veto",
        "proposal",
        "position_closed",
        "close_failed",
        "halted",
        "preflight_failed",
        "no_new_positions",
        "proposer_failed",
        "proposal_unusable",
        "unsizeable",
        "data_failed",
        "error",
    }
    recent = [e for e in entries if e.get("event") in interesting][-limit:]

    if not recent:
        lines.append("_Nothing to report yet._")
        return "\n".join(lines) + "\n"

    for entry in reversed(recent):
        stamp = str(entry.get("ts", ""))[:19].replace("T", " ")
        event = entry.get("event", "?")
        lines.append(f"**{stamp}Z — `{event}`**")

        for key in ("candidate", "contracts", "credit", "risk", "order_id", "reason"):
            if key in entry and entry[key] not in (None, "", []):
                lines.append(f"- {key}: {entry[key]}")

        for key in ("violations", "rationale", "verdict"):
            value = entry.get(key)
            if isinstance(value, list) and value:
                for item in value:
                    lines.append(f"- {key}: {item}")
            elif value:
                lines.append(f"- {key}: {value}")

        lines.append("")

    return "\n".join(lines) + "\n"


def write_markdown(entries: list[dict[str, Any]], path: Path | str = "journal/JOURNAL.md") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(entries), encoding="utf-8")
    return target
