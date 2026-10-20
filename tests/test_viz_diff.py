"""Tests for checkpoint diffing (row alignment, schema, values).

Fixtures mirror the real pipeline: files are written with the DataFrame
index included (an empty leading header cell), and the agent's output frame
carries *every column of the loaded input* (the leading index column
included — it is a data column as far as the agent is concerned).
``_round_out`` mimics one CSVAgent round: load the input, apply a mutation,
write with the index.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from viz.diff import compute_diff


def _write_base(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    """Write a base-style file (single index layer, like datasets/*.csv)."""
    p = tmp_path / name
    df.to_csv(p, index=True)
    return p


def _round_out(tmp_path: Path, in_path: Path, name: str, mutate=None) -> Path:
    """Write a checkpoint the way the agent does: load, mutate, to_csv(index=True)."""
    df = pd.read_csv(in_path)
    if mutate is not None:
        df = mutate(df)
    p = tmp_path / name
    df.to_csv(p, index=True)
    return p


def test_diff_aligned_with_row_removal_and_schema_changes(tmp_path: Path) -> None:
    base = pd.DataFrame(
        {
            "iso_code": ["AFG", "ALB", "IDN"],
            "Total Cases": [1.0, 2.0, 3.0],
            "junk": ["x", "y", "z"],
        }
    )
    p_in = _write_base(tmp_path, "base.csv", base)

    def mutate(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df.iso_code != "ALB"].reset_index(drop=True)
        df = df.rename(columns={"Total Cases": "total_cases"})
        df["total_cases"] = df["total_cases"].astype(int)
        df.loc[df.iso_code == "IDN", "total_cases"] = 5
        df["new_col"] = ["n1", "n2"]
        return df.drop(columns="junk")

    p_out = _round_out(tmp_path, p_in, "mod.csv", mutate)

    d = compute_diff("ds", 0, p_in, p_out, rename_map={("ds", "Total Cases"): "total_cases"})

    assert d.rows_before == 3
    assert d.rows_after == 2
    assert d.index_key_used
    assert d.cell_comparison == "aligned"
    assert d.rows_removed == 1
    assert d.rows_added == 0
    assert d.row_order_preserved is True

    kinds = {c.name_after or c.name_before: c.kind for c in d.columns}
    assert kinds["iso_code"] == "unchanged"
    assert kinds["total_cases"] == "renamed"
    assert kinds["new_col"] == "added"
    assert kinds["junk"] == "removed"

    tc = next(c for c in d.columns if c.name_after == "total_cases")
    assert tc.name_before == "Total Cases"
    assert tc.changed_cells == 1
    assert tc.sample_changes[0]["row_key"] == 2  # base row 2 (IDN)
    assert tc.sample_changes[0]["before"] == "3.0"
    assert tc.sample_changes[0]["after"] == "5"

    iso = next(c for c in d.columns if c.name_after == "iso_code")
    assert iso.changed_cells == 0
    assert not d.no_op


def test_diff_no_op_when_agent_changed_nothing(tmp_path: Path) -> None:
    base = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    p_in = _write_base(tmp_path, "in.csv", base)
    p_out = _round_out(tmp_path, p_in, "out.csv")  # agent did nothing

    d = compute_diff("ds", 0, p_in, p_out)

    # key is a fresh sequential index (no rows removed) -> identity mode
    assert d.cell_comparison == "identity"
    assert d.no_op
    assert d.rows_removed == 0
    assert d.rows_added == 0
    assert d.row_order_preserved is True
    assert all(c.kind == "unchanged" for c in d.columns)


def test_diff_nan_vs_nan_is_not_a_change(tmp_path: Path) -> None:
    base = pd.DataFrame({"n": [1.0, np.nan, 3.0]})
    p_in = _write_base(tmp_path, "in.csv", base)
    p_out = _round_out(tmp_path, p_in, "out.csv")  # agent did nothing

    d = compute_diff("ds", 0, p_in, p_out)

    assert d.cell_comparison == "identity"
    assert d.no_op
    n = next(c for c in d.columns if c.name_after == "n")
    assert n.changed_cells == 0
    assert n.nulls_before == 1
    assert n.nulls_after == 1


def test_diff_positional_when_no_index_layers(tmp_path: Path) -> None:
    p_in = tmp_path / "in.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(p_in, index=False)
    p_out = tmp_path / "out.csv"
    pd.DataFrame({"a": [1, 9], "b": ["x", "y"]}).to_csv(p_out, index=False)

    d = compute_diff("ds", 1, p_in, p_out)

    assert d.cell_comparison == "positional"
    assert not d.index_key_used
    a = next(c for c in d.columns if c.name_after == "a")
    assert a.changed_cells == 1
    assert a.sample_changes[0]["row_key"] == 1


def test_diff_skips_cells_when_unaligned_and_sizes_differ(tmp_path: Path) -> None:
    p_in = tmp_path / "in.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(p_in, index=False)
    p_out = tmp_path / "out.csv"
    pd.DataFrame({"a": [1], "b": [7]}).to_csv(p_out, index=False)

    d = compute_diff("ds", 1, p_in, p_out)

    assert d.cell_comparison == "skipped"
    assert d.rows_before == 3
    assert d.rows_after == 1
    a = next(c for c in d.columns if c.name_after == "a")
    assert a.changed_cells is None
    assert not d.no_op


def test_diff_reports_dtype_change_without_value_change(tmp_path: Path) -> None:
    base = pd.DataFrame({"n": [1, 2]})
    p_in = _write_base(tmp_path, "in.csv", base)
    p_out = _round_out(tmp_path, p_in, "out.csv", mutate=lambda df: df.assign(n=df["n"].astype(float)))

    d = compute_diff("ds", 0, p_in, p_out)

    assert d.cell_comparison == "identity"
    n = next(c for c in d.columns if c.name_after == "n")
    assert n.dtype_changed
    assert n.changed_cells == 0
    assert not n.values_changed
    assert not d.no_op  # a dtype change counts as a change


def test_diff_detects_row_reordering(tmp_path: Path) -> None:
    base = pd.DataFrame({"a": [10, 20, 30]})
    p_in = _write_base(tmp_path, "in.csv", base)
    p_out = _round_out(tmp_path, p_in, "out.csv", mutate=lambda df: df.iloc[[2, 0, 1]].reset_index(drop=True))

    d = compute_diff("ds", 0, p_in, p_out)

    assert d.cell_comparison == "aligned"
    assert d.row_order_preserved is False
    assert d.rows_removed == 0
    assert d.rows_added == 0
    a = next(c for c in d.columns if c.name_after == "a")
    assert a.changed_cells == 0  # values follow their rows


def test_diff_broken_key_falls_back_positionally(tmp_path: Path) -> None:
    base = pd.DataFrame({"a": [1, 2]})
    p_in = _write_base(tmp_path, "in.csv", base)
    # agent overwrote the carried position column -> key no longer valid
    p_out = _round_out(tmp_path, p_in, "out.csv", mutate=lambda df: df.assign(**{"Unnamed: 0": [5, 6], "a": [1, 9]}))

    d = compute_diff("ds", 0, p_in, p_out)

    assert d.cell_comparison == "positional"
    assert not d.index_key_used
    a = next(c for c in d.columns if c.name_after == "a")
    assert a.changed_cells == 1


def test_diff_fingerprint_alignment_when_agent_reindexes(tmp_path: Path) -> None:
    base = pd.DataFrame({"key": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
    p_in = _write_base(tmp_path, "in.csv", base)

    def mutate(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df.key != 3].reset_index(drop=True)
        # agent reassigned the position column (destroys row identity)
        df["Unnamed: 0"] = np.arange(len(df))
        return df

    p_out = _round_out(tmp_path, p_in, "out.csv", mutate)

    d = compute_diff("ds", 0, p_in, p_out)

    assert d.cell_comparison == "fingerprint"
    assert d.rows_removed == 1
    assert d.rows_added == 0
    assert d.row_order_preserved is True
    v = next(c for c in d.columns if c.name_after == "v")
    assert v.changed_cells == 0  # surviving rows are untouched
