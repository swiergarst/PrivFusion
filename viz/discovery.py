"""Discover run artifacts on disk and group them into runs/rounds.

Only the *new* naming scheme written by the current ``main.py`` is
understood. Anything else found in the artifact directories (legacy log
names, ad-hoc exports, lock files) is collected into a ``skipped`` list so
nothing is silently ignored.
"""

from __future__ import annotations

import re
from pathlib import Path

from viz.model import RoundArtifacts, SkippedFile

# run_id is a time.time() float, e.g. "1792398151.674631".
_ID = r"(?P<id>\d+\.\d+)"
_ROUND = r"(?P<round>\d+)"
_NAME = r"(?P<name>.+)"

# (compiled pattern, artifact kind). Order matters only for readability.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"^harmonization_instructions_round_{_ROUND}_{_ID}\.csv$"), "instructions"),
    (re.compile(rf"^{_NAME}_modified_round{_ROUND}_{_ID}\.csv$"), "checkpoint"),
    (re.compile(rf"^fed_harmonization_logs_round{_ROUND}_id_{_ID}\.log$"), "fed_log"),
    (re.compile(rf"^local_modif_logs_round{_ROUND}_id_{_ID}\.log$"), "local_log"),
    (re.compile(rf"^local_results_{_NAME}_round_{_ROUND}_id_{_ID}\.json$"), "local_results"),
]

# Heuristic reasons for files we recognize-but-reject (legacy scheme).
_LEGACY_HINTS: list[tuple[str, str]] = [
    ("harm_logs_", "legacy harmonization log name (harm_logs_<id>_round_<r>.log)"),
    ("local_modif_logs_", "legacy local log name (local_modif_logs_<id>.log)"),
]


def _match(name: str) -> tuple[str, dict[str, str]] | None:
    for pattern, kind in _PATTERNS:
        m = pattern.match(name)
        if m:
            return kind, m.groupdict()
    return None


def _legacy_reason(name: str) -> str | None:
    for prefix, reason in _LEGACY_HINTS:
        if name.startswith(prefix):
            return reason
    return None


def _skipped_reason(path: Path) -> str:
    if path.name.startswith(".~lock."):
        return "office lock file"
    legacy = _legacy_reason(path.name)
    if legacy:
        return legacy
    if path.stat().st_size == 0:
        return "empty file"
    return "unrecognized filename pattern"


def discover(
    root: Path,
    dataset_modifications_dir: str = "dataset_modifications",
    dataset_checkpoints_dir: str = "dataset_checkpoints",
    logs_dir: str = "logs",
) -> tuple[dict[float, dict[int, RoundArtifacts]], list[SkippedFile]]:
    """Scan the artifact directories under ``root``.

    Returns ``(runs, skipped)`` where ``runs`` maps run_id -> {round ->
    RoundArtifacts} and ``skipped`` lists files that did not match.
    """
    runs: dict[float, dict[int, RoundArtifacts]] = {}
    skipped: list[SkippedFile] = []

    def bucket(run_id: float, rnd: int) -> RoundArtifacts:
        return runs.setdefault(run_id, {}).setdefault(rnd, RoundArtifacts(round=rnd))

    def scan(directory: Path) -> None:
        if not directory.is_dir():
            return
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            matched = _match(path.name)
            if matched is None:
                skipped.append(SkippedFile(path=path, reason=_skipped_reason(path)))
                continue
            kind, groups = matched
            run_id = float(groups["id"])
            rnd = int(groups["round"])
            ra = bucket(run_id, rnd)
            if kind == "instructions":
                ra.instructions_path = path
            elif kind == "fed_log":
                ra.fed_log_path = path
            elif kind == "local_log":
                ra.local_log_path = path
            elif kind == "checkpoint":
                ra.checkpoints[groups["name"]] = path
            elif kind == "local_results":
                ra.local_results_paths[groups["name"]] = path

    scan(root / dataset_modifications_dir)
    scan(root / dataset_checkpoints_dir)
    scan(root / logs_dir)
    return runs, skipped
