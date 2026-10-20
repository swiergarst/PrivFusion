"""Loading of instruction tables and dataset checkpoints (lazy + cached).

Every file in this pipeline is written with ``DataFrame.to_csv`` *including*
the index. Each write therefore prepends a new leading column holding the
frame's row positions at write time, so checkpoints accumulate one such
"written-index layer" per round. A file's leading column is a freshly
written index layer when its values are exactly ``0..n-1`` in order; the
*second* column of such a file is (unless the agent renamed or dropped it)
a copy of the input file's leading column, which makes it a stable row key
for diffing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from viz.model import Instructions

_CHECKPOINT_CACHE: dict[tuple[str, float], pd.DataFrame] = {}


def load_instructions(path: Path) -> Instructions:
    return Instructions(path=path, df=pd.read_csv(path))


def load_checkpoint(path: Path) -> pd.DataFrame:
    """Load a (large) dataset checkpoint, cached by (path, mtime)."""
    key = (str(path), path.stat().st_mtime)
    df = _CHECKPOINT_CACHE.get(key)
    if df is None:
        df = pd.read_csv(path, engine="pyarrow")
        _CHECKPOINT_CACHE[key] = df
    return df


def read_columns(path: Path) -> list[str]:
    """Column names of a CSV without loading the data (fast)."""
    return list(pd.read_csv(path, nrows=0).columns)


def read_row_count(path: Path) -> int:
    """Row count of a CSV by reading only its first column."""
    return len(pd.read_csv(path, usecols=[0]))


def leading_is_fresh_index(df: pd.DataFrame) -> bool:
    """True when the first column holds exactly ``0..n-1`` in order.

    That value pattern identifies a freshly written RangeIndex (a
    ``to_csv`` layer). The column *name* is engine-dependent (``''`` under
    the pyarrow engine, ``Unnamed: N[.M]`` under the c engine), so the value
    pattern is what identifies the layer, not the name.
    """
    if df.shape[1] == 0 or len(df) == 0:
        return False
    col = df.iloc[:, 0]
    if not pd.api.types.is_integer_dtype(col):
        return False
    n = len(col)
    # Strictly increasing integers with n unique values ending at n-1
    # are exactly arange(n) (is_monotonic_increasing alone is non-strict).
    if not col.is_monotonic_increasing or col.nunique() != n:
        return False
    return int(col.iloc[-1]) == n - 1
