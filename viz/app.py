"""Streamlit visualization for PrivFusion federated-harmonization runs.

Read-only: for each (run, round) it shows the suggested modifications
(instructions), what each local agent did (steps, tool calls, replay
results), the federated LLM conversation, and — the main view — the
before/after diff of each dataset checkpoint.

Run with:
    streamlit run viz/app.py

The app is a thin consumer of the ``viz`` model: parsing and diffing happen
in the library; the app only selects, caches and renders.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the repo root importable regardless of how Streamlit was launched:
# ``streamlit run viz/app.py`` only puts viz/ itself on sys.path (a CWD
# launch such as ``python -m streamlit run`` would not need this).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from viz import dataset_io
from viz.diff import DatasetDiff, compute_diff, lineage
from viz.model import Run, Workspace
from viz.workspace import _input_path_for, load_workspace

st.set_page_config(page_title="PrivFusion viz", page_icon="🧬", layout="wide")

# ---------------------------------------------------------------------------
# cached data access (keyed by path + mtime so edits invalidate the cache)
# ---------------------------------------------------------------------------


def _default_root() -> Path:
    """Repo root = parent of viz/."""
    return Path(__file__).resolve().parent.parent


def _ws_key(root: str) -> tuple[str, tuple]:
    """Cache key for the workspace: root + mtimes of the artifact locations."""
    r = Path(root)
    mtimes = []
    for name in ("config.yaml", "dataset_modifications", "dataset_checkpoints", "logs", "datasets"):
        p = r / name
        mtimes.append(p.stat().st_mtime_ns if p.exists() else 0)
    return (str(r), tuple(mtimes))


def _file_key(path: Path) -> tuple[str, int]:
    return (str(path), path.stat().st_mtime_ns)


@st.cache_data(show_spinner="Loading workspace artifacts…")
def get_workspace(key: tuple[str, tuple]) -> Workspace:
    return load_workspace(key[0])


@st.cache_data(show_spinner="Computing dataset diff…")
def get_diff(in_key: tuple, out_key: tuple, dataset: str, round_: int, rename_items: tuple) -> DatasetDiff:
    return compute_diff(dataset, round_, Path(in_key[0]), Path(out_key[0]), rename_map=dict(rename_items))


@st.cache_data(show_spinner="Loading data preview…")
def get_head(path_key: tuple, n: int = 20) -> pd.DataFrame:
    return pd.read_csv(Path(path_key[0]), nrows=n)


@st.cache_data
def get_row_count(path_key: tuple) -> int:
    return dataset_io.read_row_count(Path(path_key[0]))


@st.cache_data
def get_lineage(entries: tuple, base_key: tuple) -> list[dict]:
    diffs = [get_diff(*entry) for entry in entries]
    base_path = Path(base_key[0]) if base_key[0] else None
    base_columns = dataset_io.read_columns(base_path)[1:] if base_path is not None and base_path.is_file() else []
    return lineage(diffs, base_columns)


# ---------------------------------------------------------------------------
# small rendering helpers
# ---------------------------------------------------------------------------


def _ts(dt: datetime | None) -> str:
    return dt.strftime("%H:%M:%S") if dt is not None else ""


def _clip(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n… (truncated, {len(text):,} chars total)"


def _order_label(order: bool | None) -> str:
    return {"true": "ok", "false": "REORDERED"}.get(str(order).lower(), "?")


def _column_table(diff: DatasetDiff) -> pd.DataFrame:
    rows = []
    for c in diff.columns:
        rows.append(
            {
                "column": c.label,
                "change": c.kind,
                "dtype": (
                    f"{c.dtype_before} → {c.dtype_after}"
                    if c.kind in ("unchanged", "renamed")
                    else (c.dtype_after or c.dtype_before or "")
                ),
                "nulls": f"{c.nulls_before} → {c.nulls_after}"
                if c.nulls_before is not None and c.nulls_after is not None
                else "",
                "changed cells": c.changed_cells,
                "compared": c.n_compared,
            }
        )
    return pd.DataFrame(rows)


def _lineage_table(rows: list[dict], labels: list[int]) -> pd.DataFrame:
    out = []
    for row in rows:
        entry = {"input column": row["input_name"]}
        for i, rnd in enumerate(labels):
            name = row["names"][i] if i < len(row["names"]) else None
            event = row["events"].get(i, "")
            entry[f"round {rnd}"] = (name or "·") + (f"  [{event}]" if event else "")
        out.append(entry)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# sidebar: selection
# ---------------------------------------------------------------------------

st.title("🧬 PrivFusion — harmonization run visualization")

root = st.sidebar.text_input("Workspace root", value=_default_root().as_posix(), key="root")
if st.sidebar.button("🔄 Reload workspace", use_container_width=True):
    st.cache_clear()
    st.rerun()

try:
    ws = get_workspace(_ws_key(root))
except Exception as exc:  # pragma: no cover - user error surface
    st.error(f"Could not load workspace: {exc}")
    st.stop()

if not ws.runs:
    st.warning("No runs found under the selected workspace root.")
    st.stop()

run_id = st.sidebar.selectbox("Run", ws.run_ids(), format_func=str)
run: Run = ws.runs[run_id]
rnd = st.sidebar.selectbox("Round", run.round_list, format_func=lambda r: f"round {r}")
ra = run.rounds[rnd]
datasets = sorted(ra.datasets)
if not datasets:
    st.warning("No datasets found for this round.")
    st.stop()
dataset = st.sidebar.selectbox("Dataset", datasets)

st.caption(
    f"Run **{run_id}** — round **{rnd}** — dataset **{dataset}** "
    f"({len(run.rounds)} rounds, {len(ws.base_datasets)} base datasets). Read-only view of the artifacts."
)

tab_overview, tab_changes, tab_instr, tab_local, tab_fed = st.tabs(
    ["🏠 Overview", "📊 Dataset changes", "📋 Instructions", "🤖 Local agent", "🛰️ Federated log"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

with tab_overview:
    st.subheader(f"Run {run_id}")
    rows = []
    for r in run.round_list:
        ra_r = run.rounds[r]
        row = {"round": r}
        if ra_r.instructions is not None:
            row["instructions"] = (
                f"{len(ra_r.instructions.normalized)} normalized / {len(ra_r.instructions.dropped)} dropped"
            )
        if ra_r.fed_log is not None:
            row["fed LLM calls"] = len(ra_r.fed_log.calls)
        if ra_r.local_log is not None:
            sess = ", ".join(s.dataset or "?" for s in ra_r.local_log.sessions)
            row["local sessions"] = f"{len(ra_r.local_log.sessions)} ({sess})"
        for name, lr in sorted(ra_r.local_results.items()):
            row[f"results: {name}"] = f"{'ok' if lr.success else 'FAIL'}, {len(lr.steps)} steps, replay " + (
                "PASS" if lr.replay_ok else "FAIL" if lr.replay_ok is False else "?"
            )
        for name, path in sorted(ra_r.checkpoints.items()):
            row[f"checkpoint: {name}"] = f"{get_row_count(_file_key(path)):,} rows"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.markdown("**Base datasets**")
    for name, path in sorted(ws.base_datasets.items()):
        try:
            n = get_row_count(_file_key(path))
            st.caption(f"- `{name}`: {path} — {n:,} rows")
        except Exception:
            st.caption(f"- `{name}`: {path} (row count unavailable)")

    if ws.skipped:
        with st.expander(f"{len(ws.skipped)} file(s) skipped by discovery"):
            st.dataframe(pd.DataFrame([{"file": s.path.name, "reason": s.reason} for s in ws.skipped]), hide_index=True)

# ---------------------------------------------------------------------------
# Dataset changes (before/after diff)
# ---------------------------------------------------------------------------

with tab_changes:
    in_path = _input_path_for(ws, run, rnd, dataset)
    out_path = ra.checkpoints.get(dataset)
    if in_path is None or out_path is None or not in_path.is_file() or not out_path.is_file():
        st.warning(f"Cannot diff `{dataset}` for round {rnd}: input or output file not found.")
    else:
        ins = ra.instructions
        d = get_diff(
            _file_key(in_path),
            _file_key(out_path),
            dataset,
            rnd,
            tuple(sorted((ins.rename_map if ins is not None else {}).items())),
        )
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(
            "Rows",
            f"{d.rows_before:,} → {d.rows_after:,}",
            delta=f"{d.rows_after - d.rows_before:+,}",
            delta_color="off",
        )
        m2.metric("Rows removed", f"{d.rows_removed:,}")
        m3.metric("Rows added", f"{d.rows_added:,}")
        m4.metric("Columns", f"{d.n_renamed} ren · {d.n_added} add · {d.n_removed} rem")
        m5.metric("Changed cells", f"{d.total_changed_cells:,}")
        m6.metric("Comparison", d.cell_comparison)
        st.caption(
            f"input: `{in_path.name}` → output: `{out_path.name}` — row order: **{_order_label(d.row_order_preserved)}**"
        )
        if d.no_op:
            st.success("No changes detected in this round.")

        st.markdown("**Per-column changes**")
        st.dataframe(_column_table(d), hide_index=True)

        for c in d.columns:
            if c.sample_changes:
                with st.expander(
                    f"💡 {c.label}: {c.changed_cells:,} changed cells (first {len(c.sample_changes)} shown)"
                ):
                    st.dataframe(pd.DataFrame(c.sample_changes), hide_index=True)

        with st.expander("Before / after data preview (first 20 rows)"):
            left, right = st.columns(2)
            left.markdown(f"**Before** — `{in_path.name}`")
            left.dataframe(get_head(_file_key(in_path)), hide_index=True)
            right.markdown(f"**After** — `{out_path.name}`")
            right.dataframe(get_head(_file_key(out_path)), hide_index=True)

        with st.expander("Column lineage across the run"):
            entries = []
            labels: list[int] = []
            for r in run.round_list:
                o_p = run.rounds[r].checkpoints.get(dataset)
                i_p = _input_path_for(ws, run, r, dataset)
                if o_p is None or i_p is None or not i_p.is_file() or not o_p.is_file():
                    continue
                ins_r = run.rounds[r].instructions
                entries.append(
                    (
                        _file_key(i_p),
                        _file_key(o_p),
                        dataset,
                        r,
                        tuple(sorted((ins_r.rename_map if ins_r is not None else {}).items())),
                    )
                )
                labels.append(r)
            if entries:
                base = ws.base_datasets.get(dataset)
                base_key = _file_key(base) if base is not None and base.is_file() else ("", 0)
                st.dataframe(_lineage_table(get_lineage(tuple(entries), base_key), labels), hide_index=True)
            else:
                st.info("Not enough checkpoints to build a lineage.")

# ---------------------------------------------------------------------------
# Instructions (suggested modifications)
# ---------------------------------------------------------------------------

with tab_instr:
    ins = ra.instructions
    if ins is None:
        st.info("No instructions file for this round.")
    else:
        st.caption(f"{len(ins.normalized)} normalized / {len(ins.dropped)} dropped (cluster_id == 9999)")
        scope = st.radio("Show", [dataset, "all datasets"], horizontal=True, key=f"instr_scope_{dataset}")
        tab_norm, tab_drop = st.tabs(["Normalized", "Dropped"])
        with tab_norm:
            df = ins.normalized
            if scope != "all datasets":
                df = df[df["dataset"] == scope]
            st.dataframe(df, hide_index=True)
        with tab_drop:
            df = ins.dropped
            if scope != "all datasets":
                df = df[df["dataset"] == scope]
            st.dataframe(df, hide_index=True)

# ---------------------------------------------------------------------------
# Local agent (per-dataset results + session from the local log)
# ---------------------------------------------------------------------------

with tab_local:
    lr = ra.local_results.get(dataset)
    if lr is None:
        st.info(f"No local_results for `{dataset}` in round {rnd}.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Success", "yes" if lr.success else "no")
        m2.metric("Replay", "PASS" if lr.replay_ok else "FAIL" if lr.replay_ok is False else "?")
        m3.metric("Steps", len(lr.steps))
        if lr.summary:
            st.markdown(f"**Summary:** {lr.summary}")
        if lr.replay_message:
            st.caption(f"replay: {lr.replay_message}")
        if lr.error:
            st.error(lr.error)
        for i, step in enumerate(lr.steps):
            with st.expander(f"Step {i}: {step.get('description', '')}"):
                st.code(step.get("code", ""), language="python")

    session = next(
        (s for s in (ra.local_log.sessions if ra.local_log is not None else []) if s.dataset == dataset), None
    )
    if session is not None:
        st.markdown("**Agent session (from the local log)**")
        counts = session.tool_call_counts
        cap1, cap2, cap3 = st.columns(3)
        cap1.metric("Tool calls", ", ".join(f"{k}×{v}" for k, v in sorted(counts.items())) or "none")
        cap2.metric("Request size", f"{session.request_chars.get('total', '?')} chars")
        cap3.metric("Attribution", "exact" if session.attribution_exact else "heuristic")
        with st.expander("Tool calls"):
            if session.tool_events:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "tool": e.name,
                                "call": _ts(e.call_ts),
                                "args": e.args_preview,
                                "result": e.result_preview,
                            }
                            for e in session.tool_events
                        ]
                    ),
                    hide_index=True,
                )
            else:
                st.info("No tool calls recorded.")
        with st.expander("User prompt"):
            st.text(session.user_prompt or "(empty)")
        with st.expander("Final response"):
            st.text(_clip(session.response_text, 12000) or "(empty)")
    elif ra.local_log is not None:
        st.info("No local-log session attributed to this dataset in this round.")

# ---------------------------------------------------------------------------
# Federated harmonization log
# ---------------------------------------------------------------------------

with tab_fed:
    fl = ra.fed_log
    if fl is None or not fl.calls:
        st.info("No federated LLM calls recorded for this round.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        models = sorted({c.model for c in fl.calls if c.model})
        m1.metric("LLM calls", len(fl.calls))
        m2.metric("Model(s)", ", ".join(models) or "?")
        m3.metric("Total tokens", f"{sum(c.prompt_tokens + c.completion_tokens for c in fl.calls):,}")
        m4.metric("Total compute", f"{sum(c.compute_time_s for c in fl.calls):.1f} s")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "#": i,
                        "time": _ts(c.response_ts or c.input_ts),
                        "model": c.model,
                        "in tokens": c.prompt_tokens,
                        "out tokens": c.completion_tokens,
                        "compute s": round(c.compute_time_s, 2),
                        "parsed": "✓" if c.parse_ok else "✗",
                    }
                    for i, c in enumerate(fl.calls)
                ]
            ),
            hide_index=True,
        )
        sel = st.selectbox("Inspect call", range(len(fl.calls)), format_func=lambda i: f"call {i}")
        call = fl.calls[sel]
        st.markdown(
            f"**call {sel}** — model `{call.model}` — {call.prompt_tokens:,} in / {call.completion_tokens:,} out tokens — {call.compute_time_s:.2f} s"
        )
        if not call.parse_ok:
            st.warning("Payload could not be parsed; raw text is shown below.")
            st.code(call.input_raw[:8000], language=None)
            st.code(call.response_raw[:8000], language=None)
        else:
            for msg in call.input_messages or []:
                st.markdown(f"**{msg.get('role', '?')}**")
                st.text(_clip(str(msg.get("content", ""))))
            st.markdown("**response**")
            st.text(_clip(call.response_text) or "(empty)")
            if call.reasoning:
                with st.expander("Reasoning trace"):
                    st.text(_clip(call.reasoning))

st.caption(
    "Visualization only — nothing is re-run or modified. Data: `viz` package (discovery, log parsers, diff engine)."
)
