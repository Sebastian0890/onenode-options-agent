"""Tests for the journal and the two things that read it.

The sync test at the bottom exists because the dashboard and the markdown
renderer each keep their own list of which events are worth showing, in
different languages. Two hand-maintained lists of the same thing drift, and the
failure mode is silent: an event stops appearing and nobody notices, because
"nothing shown" looks exactly like "nothing happened".
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from onenode.journal import Journal, render_markdown, write_markdown

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestWriting:
    def test_appends_one_line_per_event(self, tmp_path):
        journal = Journal(tmp_path / "t.jsonl", run_id="abc")
        journal.record("run_started")
        journal.record("order_placed", candidate="SPY/A", contracts=3)
        assert len(journal.entries()) == 2

    def test_every_entry_carries_a_timestamp_and_run_id(self, tmp_path):
        journal = Journal(tmp_path / "t.jsonl", run_id="abc")
        entry = journal.record("proposal", rationale="because")
        assert entry["run_id"] == "abc"
        assert entry["ts"].endswith("+00:00")

    def test_a_truncated_line_does_not_break_reading(self, tmp_path):
        path = tmp_path / "t.jsonl"
        journal = Journal(path, run_id="abc")
        journal.record("run_started")
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"event": "half-writ')
        journal.record("order_placed")
        assert [e["event"] for e in journal.entries()] == ["run_started", "order_placed"]

    def test_missing_file_reads_as_empty(self, tmp_path):
        assert Journal(tmp_path / "nope.jsonl").entries() == []

    def test_awkward_values_are_encoded_rather_than_dropped(self, tmp_path):
        from datetime import date

        from onenode.risk.models import Side

        journal = Journal(tmp_path / "t.jsonl")
        entry = journal.record("x", when=date(2026, 9, 4), side=Side.SELL, legs=["a", "b"])
        assert entry["when"] == "2026-09-04"
        assert entry["side"] == "sell"
        assert entry["legs"] == ["a", "b"]
        json.dumps(entry)  # must round-trip


class TestRendering:
    def test_counts_the_things_that_matter(self):
        entries = [
            {"event": "run_started", "run_id": "a"},
            {"event": "order_placed", "run_id": "a", "candidate": "SPY/X"},
            {"event": "gate_blocked", "run_id": "b", "violations": ["too big"]},
            {"event": "risk_officer_veto", "run_id": "b", "reason": "too close"},
        ]
        out = render_markdown(entries)
        assert "| Runs | 2 |" in out
        assert "| Orders placed | 1 |" in out
        assert "| Blocked by the hard gate | 1 |" in out
        assert "| Vetoed by the risk officer | 1 |" in out

    def test_a_failed_run_is_shown_not_hidden(self):
        """The most informative thing that can happen must not render as silence."""
        out = render_markdown([{"event": "proposer_failed", "reason": "no API key"}])
        assert "Nothing to report yet" not in out
        assert "no API key" in out

    def test_an_empty_journal_says_so(self):
        assert "Nothing to report yet" in render_markdown([])

    def test_newest_first(self):
        entries = [
            {"event": "order_placed", "ts": "2026-09-01T10:00:00", "candidate": "OLD"},
            {"event": "order_placed", "ts": "2026-09-02T10:00:00", "candidate": "NEW"},
        ]
        out = render_markdown(entries)
        assert out.index("NEW") < out.index("OLD")

    def test_violations_are_listed_individually(self):
        out = render_markdown([{"event": "gate_blocked", "violations": ["one", "two"]}])
        assert "one" in out
        assert "two" in out

    def test_written_to_disk(self, tmp_path):
        path = write_markdown([{"event": "order_placed"}], tmp_path / "j" / "JOURNAL.md")
        assert path.exists()
        assert "Trade journal" in path.read_text(encoding="utf-8")


class TestDashboardStaysInSync:
    def test_the_dashboard_shows_exactly_what_the_markdown_shows(self):
        """Two hand-maintained lists of the same thing drift. This catches it."""
        html = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        match = re.search(r"const INTERESTING = new Set\(\[(.*?)\]\)", html, re.S)
        assert match, "could not find INTERESTING in index.html"
        from_html = set(re.findall(r'"([a-z_]+)"', match.group(1)))

        source = (REPO_ROOT / "src" / "onenode" / "journal.py").read_text(encoding="utf-8")
        block = re.search(r"interesting = \{(.*?)\}", source, re.S)
        assert block, "could not find `interesting` in journal.py"
        from_python = set(re.findall(r'"([a-z_]+)"', block.group(1)))

        assert from_html == from_python, (
            f"only in the dashboard: {sorted(from_html - from_python)}; "
            f"only in the renderer: {sorted(from_python - from_html)}"
        )

    def test_failure_events_are_in_the_list(self):
        """Named explicitly, so removing one is a deliberate act with a red test."""
        source = (REPO_ROOT / "src" / "onenode" / "journal.py").read_text(encoding="utf-8")
        block = re.search(r"interesting = \{(.*?)\}", source, re.S).group(1)
        for event in ("proposer_failed", "order_failed", "preflight_failed", "halted"):
            assert f'"{event}"' in block
