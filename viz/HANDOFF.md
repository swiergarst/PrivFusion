# PrivFusion viz — handoff / status / plan

> Written for a fresh session to pick up the work. Everything below is verified
> as of 2026-10-20. Nothing is committed yet (deliberate — user wants to test
> manually first). All files listed are **new/untracked**.

## 1. Objective

Visualization-only tooling for the PrivFusion federated-harmonization workflow:
`local datasets → suggested modifications → modified dataset`, per round, keyed
by `round` + `run_id`. Built from the artifacts `main.py` writes:
harmonization-instruction CSVs, dataset checkpoints, per-round LLM logs, and
local-agent result JSONs.

Constraints that were agreed:
- **Streamlit** UI (Step 2) consumes a model/parsing layer (Step 1) with a CLI.
- **Visualization only** — no re-running rounds, no parameter modification.
- **New naming scheme only** (as written by current `main.py`); legacy formats
  are collected into a skip list, not parsed.
- **Priority feature: dataset changes** (before/after diff per round).
- Implementation lives in a **top-level `viz/` folder**.
- `reproduce_scripts/` is ignored.
- Test/debugging run: **`1792398151.674631`** — the only complete run in the
  repo (3 rounds, all artifact types, both datasets).

## 2. Current status: both steps are functionally complete

### Step 1 — model/parsing layer + CLI (done, validated)

| File | Role |
|---|---|
| `viz/__init__.py` | Package docstring. |
| `viz/model.py` | Typed dataclasses: `Workspace`, `Run`, `RoundArtifacts`, `Instructions`, `FedLog`/`FedCall`, `LocalLog`/`AgentSession`/`ToolEvent`, `LocalResults`, `SkippedFile`. |
| `viz/discovery.py` | 5 filename regexes (`instructions`, `checkpoint`, `fed_log`, `local_log`, `local_results`); `discover(root, ...)` → `(dict[run_id, dict[round, RoundArtifacts]], skipped)`; skip reasons (legacy hints, office lock files, empty, unrecognized). |
| `viz/log_parsers.py` | `RECORD_RE` splits records on `%(asctime)s %(name)s %(levelname)s %(message)s`; continuation lines attach to the previous record. `parse_fed_log`: pairs i-th `input messages: <repr>` with i-th `response: <repr>` via `ast.literal_eval` (raw kept + `parse_ok=False` on failure); extracts `model`, usage tokens, `surf.compute_time_s`. `parse_local_log`: sessions on `>>> LLM REQUEST <<<`/`>>> LLM RESPONSE <<<`; `[initial request size]` line (before the request) → `request_chars`; `tool call:`/`tool result:` attached to the open session; session→dataset attribution by matching the prompt's `Columns/dtypes:` block (exact set match, else best Jaccard). |
| `viz/dataset_io.py` | `load_checkpoint` (pyarrow, cached by `(path, mtime)`), `load_instructions`, `read_columns` (nrows=0, **c engine**), `read_row_count` (`usecols=[0]`, **c engine** — pyarrow rejects integer `usecols`; `nrows` unsupported with pyarrow in pandas 3.0), `leading_is_fresh_index(df)` — value-based index-layer detection (see §3). |
| `viz/diff.py` | `ColumnDiff`, `DatasetDiff`, `compute_diff(dataset, round_, in_path, out_path, rename_map, max_samples)`, `lineage(diffs, base_columns)`. See §3 for the alignment model. |
| `viz/workspace.py` | `load_workspace(root)` (eager parse of small artifacts; checkpoints lazy) + CLI `python -m viz.workspace [--root R] [--diff]`. |
| `conftest.py` (repo root) | Puts repo root on `sys.path` so `import viz` works under `uv run pytest tests/`. |

### Step 2 — Streamlit UI (done, smoke-tested)

- `viz/app.py` — run with **`streamlit run viz/app.py`** (repo root).
- Sidebar: workspace root (defaults to repo root = parent of `viz/`), Reload button, Run / Round / Dataset selects.
- Tabs:
  - **Overview**: per-round artifact table (instructions normalized/dropped, fed calls, local sessions, local_results status, checkpoint row counts), base datasets, skipped files.
  - **Dataset changes** (the priority view): metrics (rows before→after, removed/added, column Δ, changed cells, comparison mode), row-order flag, per-column table (name/label, kind, dtype, nulls, changed cells, compared), per-column sample-change expanders, before/after 20-row preview, cross-round **column lineage** table.
  - **Instructions**: normalized vs dropped tables, filter by selected dataset or all.
  - **Local agent**: success/replay/steps metrics, summary, per-step code expanders, plus the attributed local-log session (tool-call counts/table, request size, attribution mode, user prompt, final response).
  - **Federated log**: call metrics (count, models, tokens, compute), call table, "Inspect call" selectbox → messages/response/reasoning (truncated display).
- Caching: everything expensive is `st.cache_data` keyed by **path + mtime** (`_file_key`, `_ws_key`); Reload button calls `st.cache_clear()`.
- Read-only: no widget writes to disk.

### Tests (21 passing, all in `tests/test_viz_*.py`)

- `test_viz_discovery.py` (4): grouping by run/round, distinct runs, empty matched file, skip reasons.
- `test_viz_log_parsers.py` (6): fed pairing, malformed payload → raw kept, unrelated records, local session/tools/attribution (exact), heuristic attribution (Jaccard), empty file.
- `test_viz_diff.py` (9): aligned with gaps + row removal + schema changes; no-op (identity); **NaN-vs-NaN is not a change**; positional fallback; skipped (unaligned, size mismatch); dtype-only change; row reordering detection; broken key → positional; **fingerprint alignment when the agent reindexes**.
  - Fixture convention: `_write_base` (base-style file, `to_csv(index=True)` on a frame *without* a position column) and `_round_out` (mimics CSVAgent: `pd.read_csv` the input — the agent sees every column, including the leading index column — apply a mutation, `to_csv(index=True)`). The output frame must carry *every* input column to be realistic.
- `test_viz_app.py` (2): `AppTest` smoke on a synthetic workspace built in `tmp_path` (asserts run discovered, diff metrics, rename from instructions shows as `renamed`); app with no artifacts shows the "No runs found" warning.
  - AppTest gotcha: selectbox `.options` are the **formatted display strings**; `.select()` maps back to original values via testing data. Re-fetch elements from the tree after every `.run()` (old references raise KeyError).

### Validation on the real run (`1792398151.674631`)

CLI `python -m viz.workspace --diff` (≈3 s) now matches the agents' self-reported
steps exactly:

| round | dataset | diff result (matches ground truth) |
|---|---|---|
| 0 | covid19-global | 166326 → 156370 rows, **9956 removed** (non-country aggregates filtered), 0 changed cells, mode=`fingerprint` (agent reindexed `Unnamed: 0`) |
| 0 | covid19-indonesia | 31822 rows, 2 renames, 3 value-changed cols: Date M/D/YYYY→ISO (31822), Location ISO Code → IDN (30893), Location → 'Indonesia' (30893), mode=`identity` |
| 1 | covid19-global | `new_deaths` 20826 cells null→0 (nulls 20826→0), mode=`identity` |
| 1 | covid19-indonesia | 2 renames, 0 value changes (3 steps = renames/casts), mode=`identity` |
| 2 | covid19-global | `total_deaths_per_million` rounded to 2 dp (122054 cells, NaNs 20662 preserved), `Unnamed: 0.1`→`row_number` rename, mode=`identity` |
| 2 | covid19-indonesia | 5 snake_case renames, 0 value changes, mode=`identity` |

Other validated facts: fed call counts 133/132/139; local sessions 2/round with
exact attribution; local_results all success + replay PASS; checkpoint row
counts 156370 / 31822; base 166326 (global) / 31822 (indonesia).

### Test/lint status

- Full suite (CI-parity: `uv run pytest tests/ -v --tb=short --maxfail=5` from
  repo root): **97 passed, 2 skipped, 1 failed**.
- The 1 failure is **pre-existing and unrelated**:
  `tests/test_dataset_extraction.py::test_store_dataset_information` calls
  `DatasetInformation.write(...)` — the dataclass in
  `src/privfusion/data_models.py` has no `write`/`read` methods (stale test;
  those files are unmodified by this work).
- `ruff check` + `ruff format --check` clean on `viz/`, `tests/test_viz_*.py`,
  `conftest.py` (repo ruff config: line-length 120, double quotes, py311).
- Streamlit 1.65 headless boot verified (HTTP 200, clean server log); `AppTest`
  exercised all rounds × datasets + fed inspector on the real workspace.

## 3. Critical design decisions (do not regress)

1. **Index-layer model.** Every file is written with `to_csv(index=True)`, so
   each round stacks a new *leading* written-index layer:
   `F_r` = [write-index `0..n-1`, copy of previous file's leading column, …, data].
   Column *names* are engine-dependent (`''` under pyarrow, `Unnamed: N[.M]`
   under the c engine) — detection must be **value-based**:
   `leading_is_fresh_index(df)` = integer dtype + monotonic + `nunique()==n` +
   last value `== n-1` (≡ exactly `arange(n)`; cheap, no full array compare).
2. **Diff alignment modes** (`DatasetDiff.cell_comparison`):
   - input key: when the input's leading column is fresh, rows are identified by
     physical position (`key_i = arange(n_in)`);
   - output key: when the output's leading column is fresh, `key_o = out_df`'s
     **second** column (the carried copy of the input's leading column);
   - `aligned`: `key_o` integer, unique, ⊆ `key_i`, **not** a fresh
     `arange(n_out)` sequence → values are true input positions (gaps = removed
     rows; non-monotonic = reordered rows, `row_order_preserved=False`).
     `data_in = in_df[:, 1:]`, `data_out = out_df[:, 2:]`, rows re-ordered so
     compared row k shares the same input position.
   - `identity`: `key_o` is a fresh sequential index (agent reassigned it, or
     nothing removed) and `n_in == n_out` → positional pairing, order assumed.
   - `fingerprint`: `key_o` reindexed **and** `n_in != n_out` (real round-0
     case: agent did `df['Unnamed: 0'] = np.arange(len(df))` after filtering) →
     two-pointer order-preserving subsequence match on row fingerprints
     (`_fingerprint_align`): joined string values of the first 4/8/16
     same-name+same-dtype paired columns (leading columns are the identifiers in
     this pipeline); NUL sentinel for NA; accepted only when **all** output rows
     match, else `skipped`.
   - `positional` / `skipped`: fallbacks (equal sizes / sizes differ).
   - Validity: a valid key with `n_out > n_in` is impossible (unique subset).
3. **NaN-safe cell comparison.** `Series.ne` yields True for NaN-vs-NaN and NA
   for str-NA. Used: `changed = bv.ne(av).fillna(True)` then mask out
   `bv.isna() & av.isna()`. Null counts and changed cells are over the
   compared rows (aligned/identity/fingerprint/positional frames).
4. **Schema alignment**: exact name match first, then the instructions' rename
   map `{(dataset, feature_name) → norm_feature_name}` (from `cluster_id != 9999`
   rows) so renames are reported as `renamed`, not add+remove.
5. **Log parsing**: fed records are single-line `input messages: <repr>` /
   `response: <repr>`; local logs have multi-line request blocks (continuation
   lines = lines without the record prefix).
6. **Session attribution**: prompt `Columns/dtypes:` block parsed with
   `rsplit(None, 1)` per line; exact set match against candidate column lists
   (round 0 → base datasets; round r → round r-1 checkpoints via
   `read_columns`), else best Jaccard (`attribution_exact` flag).

## 4. Commands

```bash
# from repo root, Python 3.12.13, pandas 3.0.6, pyarrow 24.0.0, streamlit 1.65.0
.venv/bin/python -m viz.workspace --diff        # CLI incl. diffs (~3 s)
.venv/bin/python -m streamlit run viz/app.py    # the UI
uv run pytest tests/ -v --tb=short --maxfail=5  # full suite (≈1–2.5 min; 2 tests are network-flaky)
uv run pytest tests/test_viz_*.py -q            # viz-only (≈15 s)
.venv/bin/python -m ruff check viz/ tests/test_viz_*.py conftest.py
.venv/bin/python -m ruff format --check viz/ tests/test_viz_*.py conftest.py
```

## 5. Plan / remaining work

Nothing is blocking. Suggested order:

1. **Manual user testing** (user's next step): `streamlit run viz/app.py`,
   walk all tabs on run `1792398151.674631`; fix UX nits as they come up.
2. **Polish candidates** (pick as needed):
   - UI: per-column before/after *value histograms* or null-share bars for the
     changed columns; download-a-row-range expander; timeline of the 3 rounds'
     row counts on the Overview tab; persist sidebar selections via
     `st.session_state` keys.
   - Diff: `rows_removed`/`rows_added` in `identity` mode are assumed 0 (a
     same-count replace would be invisible) — could add an optional
     fingerprint cross-check when `identity` diffs report an abnormal number of
     changed cells.
   - `lineage` display when a mid-run checkpoint for a dataset is missing
     (currently lineage uses the available contiguous rounds; labels map
     correctly, but round-0 input columns must come from the base dataset).
3. **Commit** when the user is happy (suggested split):
   - `viz` package + `conftest.py` + `tests/test_viz_*.py` as one feature commit;
     or split core (discovery/parsers/diff/workspace) from `app.py`+its test.
   - `uv.lock` currently shows as modified (M) — inspect (`git diff uv.lock`)
     before committing; it changed while adding streamlit-era deps. If it's not
     part of this feature, restore it.
4. **Known caveats to remember** (document in commit message/PR if relevant):
   - `test_dataset_extraction.py::test_store_dataset_information` fails
     pre-existing (stale API) — not caused by this work.
   - Discovery skip-reason logic for *legacy* files is covered by synthetic
     fixtures only (user cleaned the repo; only the new-format run remains).
   - `st.cache_data` workspace key uses directory mtimes — editing a file's
     *contents* (not adding/removing files) needs the Reload button.
   - `fingerprint` mode assumes output rows form an order-preserving subsequence
     of the input (true for the observed agent behavior) and that some leading
     columns were not value-modified in the same round; otherwise it degrades to
     `skipped` (schema diff + row counts still shown).
   - The carried position column is deliberately **excluded** from the data
     frames (it is the row key), so its own value changes are only visible
     when the key is invalid (positional fallback).
   - Fed-log `compute_time_s`/token fields are 0 when `parse_ok` is False.

## 6. Key facts / paths

- Run id: `1792398151.674631` (the only run in the repo now).
- Config: `config.yaml` → `rounds: 3`; datasets `covid19-global`
  (`datasets/covid_global.csv`), `covid19-indonesia` (`datasets/covid_indonesia.csv`).
- Artifacts: `dataset_modifications/harmonization_instructions_round_{0,1,2}_<id>.csv`;
  `dataset_checkpoints/<ds>_modified_round{0,1,2}_<id>.csv` (46–48 MB global,
  ~8 MB indonesia); `logs/fed_harmonization_logs_round{r}_id_<id>.log`;
  `logs/local_modif_logs_round{r}_id_<id>.log`;
  `logs/local_results_<ds>_round_{r}_id_<id>.json`.
- Observed checkpoint shape (global): base 68 file cols `[pos, data]`;
  r0 69 `[L0, P, data]`; r1 70; r2 71 (r2 agent renamed `Unnamed: 0.1` →
  `row_number`). From round 0 on, every index layer is contiguous `0..n-1`
  because the agent reindexed the position column in round 0.
- Environment: Linux, Python 3.12.13 (`.venv/bin/python`), pandas 3.0.6
  (string columns default to `str` dtype — `astype(str)` does NOT stringify
  NA; fillna a sentinel first, as `_row_fingerprints` does), pyarrow 24.0.0,
  streamlit 1.65.0, ruff per repo `pyproject.toml` (line-length 120, double
  quotes, target py311).
- No git commits for this work yet; untracked: `viz/`, `conftest.py`,
  `tests/test_viz_app.py`, `tests/test_viz_diff.py`, `tests/test_viz_discovery.py`,
  `tests/test_viz_log_parsers.py`; modified: `uv.lock`.
