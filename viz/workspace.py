"""Entry point: ``load_workspace()`` plus a small CLI (``python -m viz.workspace``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from viz import dataset_io, discovery, log_parsers
from viz.diff import compute_diff
from viz.model import LocalResults, Run, Workspace


def _parse_local_results(path: Path) -> LocalResults:
    data = json.loads(path.read_text(encoding="utf-8"))
    return LocalResults(
        path=path,
        success=bool(data.get("success", False)),
        summary=data.get("summary", ""),
        steps=data.get("steps", []) or [],
        replay_ok=data.get("replay_ok"),
        replay_message=data.get("replay_message", ""),
        error=data.get("error"),
    )


def _local_log_candidates(
    base_datasets: dict[str, Path], run: Run, rnd: int, dataset_names: set[str]
) -> dict[str, list[str]]:
    """Column lists of each dataset's *input* file for round ``rnd``.

    Round 0 inputs are the base datasets; later rounds take the previous
    round's checkpoint (falling back to the base dataset when missing).
    Used to attribute local-log sessions to datasets.
    """
    candidates: dict[str, list[str]] = {}
    for name in dataset_names:
        if rnd == 0:
            path = base_datasets.get(name)
        else:
            prev = run.rounds.get(rnd - 1)
            path = (prev.checkpoints.get(name) if prev is not None else None) or base_datasets.get(name)
        if path is None:
            continue

        candidates[name] = dataset_io.read_columns(path)

    return candidates


def load_workspace(root: str | Path | None = None) -> Workspace:
    """Discover and parse all run artifacts under ``root``.

    Instructions, logs and local results are parsed eagerly (they are small);
    checkpoints stay as paths and are loaded lazily by
    :func:`viz.dataset_io.load_checkpoint`.
    """
    root = Path(root) if root is not None else Path.cwd()
    config: dict = {}
    cfg_path = root / "config.yaml"
    if cfg_path.is_file():
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    runs_raw, skipped = discovery.discover(root)

    base_datasets: dict[str, Path] = {}
    for entry in config.get("dataset_config", []) or []:
        path = root / entry["path"]
        if path.is_file():
            base_datasets[entry["name"]] = path
        else:
            skipped.append(discovery.SkippedFile(path=path, reason="base dataset listed in config.yaml but missing"))

    ws = Workspace(root=root, config=config, base_datasets=base_datasets, skipped=skipped)
    for run_id in sorted(runs_raw):
        run = Run(run_id=run_id)
        for rnd in sorted(runs_raw[run_id]):
            ra = runs_raw[run_id][rnd]
            if ra.instructions_path is not None:
                ra.instructions = dataset_io.load_instructions(ra.instructions_path)
            if ra.fed_log_path is not None:
                ra.fed_log = log_parsers.parse_fed_log(ra.fed_log_path)
            if ra.local_log_path is not None:
                dataset_names = set(ra.checkpoints) | set(ra.local_results_paths) | set(base_datasets)
                candidates = _local_log_candidates(base_datasets, run, rnd, dataset_names)
                ra.local_log = log_parsers.parse_local_log(ra.local_log_path, candidates)
            for name, path in ra.local_results_paths.items():
                try:
                    ra.local_results[name] = _parse_local_results(path)
                except (json.JSONDecodeError, OSError) as exc:
                    print(f"warning: could not parse {path}: {exc}", file=sys.stderr)
            run.rounds[rnd] = ra
        ws.runs[run_id] = run
    return ws


def _input_path_for(ws: Workspace, run: Run, rnd: int, dataset: str) -> Path | None:
    if rnd == 0:
        return ws.base_datasets.get(dataset)
    prev = run.rounds.get(rnd - 1)
    return prev.checkpoints.get(dataset) if prev is not None else None


def _print_workspace(ws: Workspace, with_diff: bool = False) -> None:
    print(f"workspace root: {ws.root}")
    for name, path in ws.base_datasets.items():
        print(f"  base dataset: {name} -> {path}")
    if ws.skipped:
        print(f"  skipped {len(ws.skipped)} file(s):")
        for s in ws.skipped:
            print(f"    - {s.path.name}: {s.reason}")
    for run_id in ws.run_ids():
        run = ws.runs[run_id]
        print(f"  run {run_id} ({len(run.rounds)} round(s): {', '.join(map(str, run.round_list))})")
        for rnd in run.round_list:
            ra = run.rounds[rnd]
            parts: list[str] = []
            if ra.instructions is not None:
                parts.append(
                    f"instructions: {len(ra.instructions.normalized)} normalized / {len(ra.instructions.dropped)} dropped"
                )
            if ra.fed_log is not None:
                parts.append(f"fed_log: {len(ra.fed_log.calls)} calls")
            if ra.local_log is not None:
                sessions = ", ".join(
                    f"#{s.index} {s.dataset or '?'}{' (heuristic)' if not s.attribution_exact else ''}"
                    for s in ra.local_log.sessions
                )
                parts.append(f"local_log: {len(ra.local_log.sessions)} session(s) [{sessions}]")
            print(f"    round {rnd}: " + "  |  ".join(parts) if parts else f"    round {rnd}: (no artifacts)")
            for name, lr in ra.local_results.items():
                replay = "PASS" if lr.replay_ok else ("FAIL" if lr.replay_ok is False else "?")
                print(f"      local_results {name}: success={lr.success} steps={len(lr.steps)} replay={replay}")
            for name, path in ra.checkpoints.items():
                try:
                    rows = dataset_io.read_row_count(path)
                except Exception:
                    rows = -1
                print(f"      checkpoint {name}: {rows} rows ({path.name})")
            if with_diff:
                _print_diffs(ws, run, rnd)
    if not ws.runs:
        print("  (no runs found)")


def _print_diffs(ws: Workspace, run: Run, rnd: int) -> None:
    ra = run.rounds[rnd]
    rename_map = ra.instructions.rename_map if ra.instructions is not None else {}
    for name, out_path in ra.checkpoints.items():
        in_path = _input_path_for(ws, run, rnd, name)
        if in_path is None:
            print(f"      diff {name}: (input file not found)")
            continue
        d = compute_diff(name, rnd, in_path, out_path, rename_map=rename_map)
        if d.row_order_preserved is True:
            order = "ok"
        elif d.row_order_preserved is False:
            order = "REORDERED"
        else:
            order = "?"
        print(
            f"      diff {name}: rows {d.rows_before} -> {d.rows_after}"
            f" (removed {d.rows_removed}, added {d.rows_added}, order {order})"
            f" | renamed {d.n_renamed}, added {d.n_added}, removed {d.n_removed}"
            f" | value-changed cols {d.n_value_changed}, total changed cells {d.total_changed_cells}"
            f" | mode={d.cell_comparison} no_op={d.no_op}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect PrivFusion run artifacts.")
    parser.add_argument("--root", default=".", help="workspace root (default: current directory)")
    parser.add_argument("--diff", action="store_true", help="also compute dataset diffs (loads full checkpoints; slow)")
    args = parser.parse_args(argv)
    _print_workspace(load_workspace(args.root), with_diff=args.diff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
