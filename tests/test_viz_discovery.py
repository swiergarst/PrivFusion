"""Tests for artifact discovery (filename patterns + run/round grouping)."""

from pathlib import Path

from viz import discovery


def _touch(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_groups_new_format_files_into_runs_and_rounds(tmp_path: Path) -> None:
    _touch(tmp_path / "dataset_modifications" / "harmonization_instructions_round_0_123.5.csv")
    _touch(tmp_path / "dataset_modifications" / "harmonization_instructions_round_1_123.5.csv")
    _touch(tmp_path / "dataset_checkpoints" / "ds-one_modified_round0_123.5.csv")
    _touch(tmp_path / "logs" / "fed_harmonization_logs_round0_id_123.5.log")
    _touch(tmp_path / "logs" / "local_modif_logs_round0_id_123.5.log")
    _touch(tmp_path / "logs" / "local_results_ds-one_round_0_id_123.5.json")

    runs, skipped = discovery.discover(tmp_path)

    assert list(runs) == [123.5]
    assert set(runs[123.5]) == {0, 1}
    r0 = runs[123.5][0]
    assert r0.instructions_path is not None
    assert r0.fed_log_path is not None
    assert r0.local_log_path is not None
    assert list(r0.checkpoints) == ["ds-one"]
    assert list(r0.local_results_paths) == ["ds-one"]
    assert runs[123.5][1].instructions_path is not None
    assert skipped == []


def test_separates_distinct_runs(tmp_path: Path) -> None:
    _touch(tmp_path / "logs" / "fed_harmonization_logs_round0_id_1.5.log")
    _touch(tmp_path / "logs" / "fed_harmonization_logs_round0_id_2.5.log")

    runs, skipped = discovery.discover(tmp_path)

    assert list(runs) == [1.5, 2.5]
    assert skipped == []


def test_matched_but_empty_file_is_not_skipped(tmp_path: Path) -> None:
    _touch(tmp_path / "logs" / "fed_harmonization_logs_round0_id_9.9.log", content="")

    runs, skipped = discovery.discover(tmp_path)

    assert runs[9.9][0].fed_log_path is not None
    assert skipped == []


def test_skips_unrecognized_files_with_reasons(tmp_path: Path) -> None:
    _touch(tmp_path / "dataset_modifications" / "dataset_modifications.csv")
    _touch(tmp_path / "logs" / "harm_logs_1791892981.0365536_round_0.log")
    _touch(tmp_path / "logs" / "local_modif_logs_1791903949.721643.log")
    _touch(tmp_path / "logs" / ".~lock.fed.log#")
    _touch(tmp_path / "dataset_checkpoints" / "something_weird.txt")

    runs, skipped = discovery.discover(tmp_path)

    assert runs == {}
    reasons = {s.path.name: s.reason for s in skipped}
    assert reasons["dataset_modifications.csv"] == "unrecognized filename pattern"
    assert "legacy" in reasons["harm_logs_1791892981.0365536_round_0.log"]
    assert "legacy" in reasons["local_modif_logs_1791903949.721643.log"]
    assert reasons[".~lock.fed.log#"] == "office lock file"
    assert reasons["something_weird.txt"] == "unrecognized filename pattern"
