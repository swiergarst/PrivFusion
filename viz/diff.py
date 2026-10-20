"""Before/after diffing of dataset checkpoints across rounds.

For one (run, round, dataset): input = base dataset (round 0) or the
previous round's checkpoint; output = this round's checkpoint. The diff
reports schema changes (added/removed/renamed columns, dtype changes) and,
on the row-aligned common part, per-column value changes. NaN-vs-NaN is
*not* a change.

Row identity. Both input and output are written with the DataFrame index
included, so the input's rows carry their file position in the leading
column, and the output carries a copy of that column as its second column
(the agent's frame kept it; each ``to_csv`` prepends one more fresh index
layer). When the carried key is intact (gaps where rows were removed), rows
are aligned on it. Agents sometimes *reassign* the position column to a
fresh sequential index (destroying the identity); then, if row counts
match, rows are aligned by position ("identity"), and if they differ, rows
are aligned as an order-preserving subsequence using row fingerprints
built from columns the agent left untouched ("fingerprint"). When no
alignment is possible the diff degrades to a positional comparison (equal
sizes) or a schema-only comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from viz.dataset_io import leading_is_fresh_index, load_checkpoint

_FP_SEP = "\x1f"  # unit separator: joins fingerprint column values per row


def _fmt(value: object, limit: int = 60) -> str:
    if value is None:
        return "<null>"
    try:
        if pd.isna(value):
            return "<null>"
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class ColumnDiff:
    """How one output column relates to the input columns."""

    name_before: str | None
    name_after: str | None
    kind: str  # unchanged | renamed | added | removed
    dtype_before: str | None = None
    dtype_after: str | None = None
    nulls_before: int | None = None  # over the compared rows
    nulls_after: int | None = None
    changed_cells: int | None = None  # over the compared rows; None when not comparable
    n_compared: int | None = None
    sample_changes: list[dict] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.kind == "renamed":
            return f"{self.name_before} -> {self.name_after}"
        return self.name_after or self.name_before or "?"

    @property
    def dtype_changed(self) -> bool:
        return self.dtype_before is not None and self.dtype_after is not None and self.dtype_before != self.dtype_after

    @property
    def values_changed(self) -> bool:
        return (self.changed_cells or 0) > 0


@dataclass
class DatasetDiff:
    """Full diff between one round's input and output for a dataset.

    ``rows_removed``/``rows_added`` are only meaningful when
    ``cell_comparison`` is "aligned" or "fingerprint"; in the other modes
    they stay 0.
    """

    dataset: str
    round: int
    input_path: Path
    output_path: Path
    rows_before: int = 0
    rows_after: int = 0
    rows_removed: int = 0
    rows_added: int = 0
    row_order_preserved: bool | None = None  # None = not verifiable
    index_key_used: bool = False
    cell_comparison: str = "aligned"  # aligned | identity | fingerprint | positional | skipped
    no_op: bool = False
    columns: list[ColumnDiff] = field(default_factory=list)

    @property
    def n_renamed(self) -> int:
        return sum(c.kind == "renamed" for c in self.columns)

    @property
    def n_added(self) -> int:
        return sum(c.kind == "added" for c in self.columns)

    @property
    def n_removed(self) -> int:
        return sum(c.kind == "removed" for c in self.columns)

    @property
    def n_value_changed(self) -> int:
        return sum(c.values_changed for c in self.columns)

    @property
    def total_changed_cells(self) -> int:
        return sum(c.changed_cells or 0 for c in self.columns)

    @property
    def changed_columns(self) -> list[ColumnDiff]:
        return [c for c in self.columns if c.kind != "unchanged" or c.values_changed]


def _schema_pair(
    data_in: pd.DataFrame,
    data_out: pd.DataFrame,
    dataset: str,
    rename_map: dict[tuple[str, str], str] | None,
) -> tuple[list[str], list[str], dict[int, int], set[int]]:
    """Pair output columns with input columns (exact name, then rename map)."""
    rmap = {k[1]: v for k, v in (rename_map or {}).items() if k[0] == dataset}
    before_cols = [str(c) for c in data_in.columns]
    after_cols = [str(c) for c in data_out.columns]
    before_pos: dict[str, int] = {}
    for i, name in enumerate(before_cols):
        before_pos.setdefault(name, i)

    pair: dict[int, int] = {}  # after position -> before position
    used_before: set[int] = set()
    for j, name_a in enumerate(after_cols):  # pass 1: exact name match
        i = before_pos.get(name_a)
        if i is not None and i not in used_before:
            pair[j] = i
            used_before.add(i)
    for j, name_a in enumerate(after_cols):  # pass 2: rename map
        if j in pair:
            continue
        for i, name_b in enumerate(before_cols):
            if i not in used_before and rmap.get(name_b) == name_a:
                pair[j] = i
                used_before.add(i)
                break
    return before_cols, after_cols, pair, used_before


def _is_reindexed_key(key_o: np.ndarray, n_out: int) -> bool:
    """True when the carried key values are a permutation of ``0..n_out-1``.

    That means the agent (re)assigned the position column a fresh sequential
    index, so it no longer identifies the input rows. Uniqueness of
    ``key_o`` must have been verified by the caller.
    """
    if len(key_o) != n_out:
        return False
    if n_out == 0:
        return True
    return int(key_o.min()) == 0 and int(key_o.max()) == n_out - 1


def _row_fingerprints(df: pd.DataFrame, cols: list[int]) -> np.ndarray:
    """Per-row fingerprint: the values of ``cols`` joined with a unit separator.

    Missing values are normalized to a NUL sentinel so that NA-vs-NA rows
    still match.
    """
    arrs: list[np.ndarray] = []
    for i in cols:
        s = df.iloc[:, i]
        if s.isna().any():
            s = s.fillna("\x00")
        arrs.append(s.astype(str).to_numpy())
    return np.array([_FP_SEP.join(row) for row in zip(*arrs, strict=True)])


def _fingerprint_align(
    data_in: pd.DataFrame,
    data_out: pd.DataFrame,
    pair: dict[int, int],
    widths: tuple[int, ...] = (4, 8, 16),
) -> np.ndarray | None:
    """Align rows assuming the output is an order-preserving subsequence of the input.

    Row fingerprints join the values of the first ``width`` columns that
    survived with the same name and dtype (leading columns first — in this
    pipeline those are the identifiers). Returns the input row position of
    each output row, or None when no window matches every output row. The
    index layers are not part of ``data_in``/``data_out``, so they never
    enter a fingerprint.
    """
    n_in = len(data_in)
    n_out = len(data_out)
    for width in widths:
        cols_j = []
        for j in sorted(pair):
            i = pair[j]
            if data_in.columns[i] == data_out.columns[j] and str(data_in.dtypes.iloc[i]) == str(
                data_out.dtypes.iloc[j]
            ):
                cols_j.append(j)
            if len(cols_j) == width:
                break
        if len(cols_j) < 2:
            continue
        cols_i = [pair[j] for j in cols_j]
        fp_in = _row_fingerprints(data_in, cols_i)
        fp_out = _row_fingerprints(data_out, cols_j)
        match = np.full(n_out, -1, dtype=np.int64)
        i = 0
        for j in range(n_out):
            target = fp_out[j]
            while i < n_in and fp_in[i] != target:
                i += 1
            if i < n_in:
                match[j] = i
                i += 1
            else:
                break  # remaining output rows cannot match
        if (match >= 0).all():
            return match
    return None


def compute_diff(
    dataset: str,
    round_: int,
    input_path: Path,
    output_path: Path,
    rename_map: dict[tuple[str, str], str] | None = None,
    max_samples: int = 10,
) -> DatasetDiff:
    """Diff one round's input vs. output for ``dataset``.

    ``rename_map`` is ``{(dataset, feature_name): norm_feature_name}`` from
    that round's instructions; it lets the diff report renames instead of an
    add+remove pair.
    """
    in_df = load_checkpoint(input_path)
    out_df = load_checkpoint(output_path)
    n_in, n_out = len(in_df), len(out_df)

    result = DatasetDiff(
        dataset=dataset,
        round=round_,
        input_path=input_path,
        output_path=output_path,
        rows_before=n_in,
        rows_after=n_out,
    )

    # Candidate data frames: drop the index layers that are write-time
    # byproducts (the input's fresh leading column is the row key; the
    # output's own fresh leading column is new, and its second column is
    # the carried copy of the input key).
    in_fresh = leading_is_fresh_index(in_df)
    out_fresh = leading_is_fresh_index(out_df)
    data_in = in_df.iloc[:, 1:] if in_fresh else in_df
    data_out = out_df.iloc[:, 2:] if (out_fresh and out_df.shape[1] >= 2) else out_df

    before_cols, after_cols, pair, used_before = _schema_pair(data_in, data_out, dataset, rename_map)

    # --- row alignment -----------------------------------------------------
    row_keys: np.ndarray | None = None  # input position of compared row k
    aligned_in, aligned_out = data_in, data_out

    key_o = out_df.iloc[:, 1].to_numpy() if (in_fresh and out_fresh and out_df.shape[1] >= 2) else None
    valid_key = False
    if key_o is not None:
        try:
            key_o = key_o.astype("int64")
            valid_key = len(np.unique(key_o)) == len(key_o) and bool(np.isin(key_o, np.arange(n_in)).all())
        except (ValueError, TypeError):
            valid_key = False

    if valid_key:
        # Strictly increasing in output order?
        monotonic = bool((np.diff(key_o) > 0).all()) if len(key_o) > 1 else True
        if n_in == n_out:
            if monotonic:
                # The carried key is a fresh sequential index: either the
                # agent reassigned it or no rows were removed — either way it
                # carries no information beyond physical position.
                result.cell_comparison = "identity"
                result.row_order_preserved = True
            else:
                # A non-monotonic permutation of the input positions: the
                # agent reordered the rows but carried the positions through.
                result.index_key_used = True
                result.cell_comparison = "aligned"
                result.row_order_preserved = False
                row_keys = key_o  # input position of each compared row
                aligned_in = data_in.iloc[key_o].reset_index(drop=True)
                aligned_out = data_out.reset_index(drop=True)
        elif monotonic and _is_reindexed_key(key_o, n_out):
            # The key was reassigned to a fresh index *and* rows are missing:
            # positions no longer identify the input rows. Recover the
            # alignment from row content (order-preserving subsequence).
            match = _fingerprint_align(data_in, data_out, pair)
            if match is not None:
                result.cell_comparison = "fingerprint"
                result.row_order_preserved = True
                row_keys = match
                result.rows_removed = int(n_in - n_out)
                aligned_in = data_in.iloc[match].reset_index(drop=True)
                aligned_out = data_out.reset_index(drop=True)
            else:
                result.cell_comparison = "skipped"
        else:
            # Intact key with gaps: values are the input row positions of the
            # surviving rows (possibly reordered).
            result.index_key_used = True
            result.cell_comparison = "aligned"
            common = set(int(v) for v in key_o)
            result.rows_removed = int(n_in - len(common))
            result.row_order_preserved = monotonic
            row_keys = np.fromiter(sorted(common), dtype=np.int64)
            out_pos = {int(v): p for p, v in enumerate(key_o)}
            out_rows = np.fromiter((out_pos[int(v)] for v in row_keys), dtype=np.int64)
            aligned_in = data_in.iloc[row_keys].reset_index(drop=True)
            aligned_out = data_out.iloc[out_rows].reset_index(drop=True)
    else:
        result.cell_comparison = "positional" if n_in == n_out else "skipped"

    comparable = result.cell_comparison in ("aligned", "identity", "fingerprint", "positional")

    # --- per-column diff ---------------------------------------------------
    for j, name_a in enumerate(after_cols):
        i = pair.get(j)
        if i is None:
            result.columns.append(
                ColumnDiff(
                    name_before=None,
                    name_after=name_a,
                    kind="added",
                    dtype_after=str(data_out.dtypes.iloc[j]),
                    nulls_after=int(data_out.iloc[:, j].isna().sum()),
                )
            )
            continue
        name_b = before_cols[i]
        bv = aligned_in.iloc[:, i] if comparable else data_in.iloc[:, i]
        av = aligned_out.iloc[:, j] if comparable else data_out.iloc[:, j]
        cd = ColumnDiff(
            name_before=name_b,
            name_after=name_a,
            kind="renamed" if name_b != name_a else "unchanged",
            dtype_before=str(bv.dtype),
            dtype_after=str(av.dtype),
            nulls_before=int(bv.isna().sum()),
            nulls_after=int(av.isna().sum()),
        )
        if comparable:
            b_na = bv.isna().to_numpy()
            a_na = av.isna().to_numpy()
            # ne() yields True for NaN vs NaN and NA for str-NA: normalize
            # so that only "one side missing or values differ" counts.
            changed = bv.ne(av).fillna(True).to_numpy()
            if b_na.any() and a_na.any():
                changed = changed & ~(b_na & a_na)
            cd.changed_cells = int(changed.sum())
            cd.n_compared = len(bv)
            if cd.changed_cells:
                for k in np.flatnonzero(changed)[:max_samples]:
                    cd.sample_changes.append(
                        {
                            "row_key": int(row_keys[k]) if row_keys is not None else int(k),
                            "before": _fmt(bv.iloc[k]),
                            "after": _fmt(av.iloc[k]),
                        }
                    )
        result.columns.append(cd)

    for i, name_b in enumerate(before_cols):
        if i not in used_before:
            result.columns.append(
                ColumnDiff(
                    name_before=name_b,
                    name_after=None,
                    kind="removed",
                    dtype_before=str(data_in.dtypes.iloc[i]),
                    nulls_before=int(data_in.iloc[:, i].isna().sum()),
                )
            )

    result.no_op = (
        result.cell_comparison != "skipped"
        and result.rows_removed == 0
        and result.rows_added == 0
        and all(c.kind == "unchanged" and not c.values_changed and not c.dtype_changed for c in result.columns)
    )
    return result


def lineage(diffs: list[DatasetDiff], base_columns: Sequence[str]) -> list[dict]:
    """Build the per-column round-over-round lineage table.

    ``diffs`` are the diffs for rounds 0..n-1 in order; ``base_columns`` are
    the input columns of round 0. Returns one row per column lineage:

    ``{'input_name': str | None, 'names': [name after round r, ...],
       'events': {round: 'unchanged'|'renamed'|'changed'|'added'|'removed'}}``

    with ``names[r]`` the column name in the output of round ``r`` (None once
    the lineage was removed).
    """
    n = len(diffs)
    rows: list[dict] = []
    active: dict[str, dict] = {}  # current column name -> lineage row

    for name in base_columns:
        row = {"input_name": str(name), "names": [None] * n, "events": {}}
        rows.append(row)
        active[str(name)] = row

    for rnd, diff in enumerate(diffs):
        for cd in diff.columns:
            if cd.kind == "removed":
                row = active.pop(cd.name_before, None)
                if row is not None:
                    row["events"][rnd] = "removed"
            elif cd.kind == "added":
                row = {
                    "input_name": None,
                    "names": [None] * rnd + [str(cd.name_after)] + [None] * (n - rnd - 1),
                    "events": {rnd: "added"},
                }
                rows.append(row)
                active[str(cd.name_after)] = row
            else:  # unchanged or renamed
                row = active.pop(cd.name_before, None)
                if row is None:  # defensive: input column not tracked
                    row = {"input_name": None, "names": [None] * n, "events": {}}
                    rows.append(row)
                if cd.kind == "renamed":
                    event = "renamed"
                elif cd.values_changed or cd.dtype_changed:
                    event = "changed"
                else:
                    event = "unchanged"
                row["events"][rnd] = event
                row["names"][rnd] = cd.name_after
                active[str(cd.name_after)] = row
    return rows
