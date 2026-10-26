"""Tests for the local-modification agent session state (snapshot cap, undo)."""

from pathlib import Path

import pandas as pd

from local_modifications import MAX_SNAPSHOTS, Session, build_tools


def _session(tmp_path: Path) -> Session:
    original = pd.DataFrame({"a": range(10)})
    return Session(input_path=tmp_path / "x.csv", original=original)


def _tool(session: Session, name: str):
    return {t.name: t for t in build_tools(session)}[name]


class TestSessionSnapshots:
    def test_snapshots_are_capped(self, tmp_path):
        """Committed steps beyond MAX_SNAPSHOTS drop the oldest snapshot."""
        session = _session(tmp_path)
        apply_t = _tool(session, "apply_transformation")
        for i in range(MAX_SNAPSHOTS + 2):
            apply_t.invoke({"description": f"step {i}", "code": "df['a'] = df['a'] + 1"})

        assert len(session.steps) == MAX_SNAPSHOTS + 2
        assert len(session.snapshots) == MAX_SNAPSHOTS

    def test_undo_still_works_after_cap(self, tmp_path):
        """Undo rolls back the latest step even when old snapshots were dropped."""
        session = _session(tmp_path)
        apply_t = _tool(session, "apply_transformation")
        for i in range(MAX_SNAPSHOTS + 2):
            apply_t.invoke({"description": f"step {i}", "code": "df['a'] = df['a'] + 1"})

        before = session.working["a"].tolist()
        _tool(session, "undo_last_transformation").invoke({})

        assert len(session.steps) == MAX_SNAPSHOTS + 1
        assert session.working["a"].tolist() == [v - 1 for v in before]

    def test_failed_transformation_commits_nothing(self, tmp_path):
        """Code that raises is not committed and adds no snapshot."""
        session = _session(tmp_path)
        apply_t = _tool(session, "apply_transformation")
        result = apply_t.invoke({"description": "bad", "code": "df['a'] = 1 / 0"})

        assert "ERROR" in result
        assert len(session.steps) == 0
        assert len(session.snapshots) == 0
