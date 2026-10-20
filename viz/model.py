"""Typed in-memory model for the viz artifact ingestion layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class SkippedFile:
    """A file inside a watched directory that did not match any known pattern."""

    path: Path
    reason: str


@dataclass
class FedCall:
    """One LLM request/response pair from a federated-harmonization log."""

    input_raw: str
    response_raw: str = ""
    input_ts: datetime | None = None
    response_ts: datetime | None = None
    input_messages: list[dict] | None = None  # parsed via ast.literal_eval
    response: dict | None = None
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    compute_time_s: float = 0.0
    parse_ok: bool = True

    @property
    def response_text(self) -> str:
        """Assistant content of the response, '' when missing/unparseable."""
        if not isinstance(self.response, dict):
            return ""
        try:
            return self.response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return ""

    @property
    def reasoning(self) -> str:
        """Optional model reasoning trace, '' when not present."""
        if not isinstance(self.response, dict):
            return ""
        try:
            return self.response["choices"][0]["message"].get("reasoning", "") or ""
        except (KeyError, IndexError, TypeError):
            return ""


@dataclass
class ToolEvent:
    """One tool call (and its result, once matched) from a local agent session."""

    name: str
    args_preview: str = ""
    call_ts: datetime | None = None
    result_preview: str = ""
    result_ts: datetime | None = None
    has_result: bool = False


@dataclass
class AgentSession:
    """One CSVAgent run (one dataset) inside a per-round local log file."""

    index: int
    dataset: str = ""
    attribution_exact: bool = False
    user_prompt: str = ""
    request_ts: datetime | None = None
    response_ts: datetime | None = None
    response_text: str = ""
    request_chars: dict[str, int] = field(default_factory=dict)
    tool_events: list[ToolEvent] = field(default_factory=list)

    @property
    def tool_call_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ev in self.tool_events:
            counts[ev.name] = counts.get(ev.name, 0) + 1
        return counts


@dataclass
class FedLog:
    """Parsed federated-harmonization log: one file per (run, round)."""

    path: Path
    calls: list[FedCall] = field(default_factory=list)
    other_records: list[str] = field(default_factory=list)  # records that are neither input nor response


@dataclass
class LocalLog:
    """Parsed local-modification log: one file per (run, round), N agent sessions."""

    path: Path
    sessions: list[AgentSession] = field(default_factory=list)


@dataclass
class LocalResults:
    """Parsed local_results_<dataset>_round_<r>_id_<id>.json (CSVAgent output)."""

    path: Path
    success: bool
    summary: str
    steps: list[dict]  # [{'description': str, 'code': str}, ...]
    replay_ok: bool | None
    replay_message: str
    error: str | None


@dataclass
class Instructions:
    """Parsed harmonization-instructions CSV for one (run, round)."""

    path: Path
    df: pd.DataFrame

    @property
    def normalized(self) -> pd.DataFrame:
        """Rows that were clustered and normalized (cluster_id != 9999)."""
        return self.df[self.df["cluster_id"] != 9999].reset_index(drop=True)

    @property
    def dropped(self) -> pd.DataFrame:
        """Rows left out of normalization (cluster_id == 9999)."""
        return self.df[self.df["cluster_id"] == 9999].reset_index(drop=True)

    @property
    def rename_map(self) -> dict[tuple[str, str], str]:
        """(dataset, feature_name) -> norm_feature_name for normalized rows."""
        out: dict[tuple[str, str], str] = {}
        for row in self.normalized.itertuples(index=False):
            if isinstance(row.norm_feature_name, str) and row.norm_feature_name:
                out[(row.dataset, row.feature_name)] = row.norm_feature_name
        return out

    def for_dataset(self, dataset: str) -> pd.DataFrame:
        return self.df[self.df["dataset"] == dataset].reset_index(drop=True)


@dataclass
class RoundArtifacts:
    """All artifacts for one (run, round).

    Discovery fills the ``*_path`` fields; the workspace then parses them
    into the typed objects (checkpoints stay as paths and are loaded lazily).
    """

    round: int
    instructions_path: Path | None = None
    fed_log_path: Path | None = None
    local_log_path: Path | None = None
    checkpoints: dict[str, Path] = field(default_factory=dict)  # dataset -> path (lazy)
    local_results_paths: dict[str, Path] = field(default_factory=dict)
    instructions: Instructions | None = None
    fed_log: FedLog | None = None
    local_log: LocalLog | None = None
    local_results: dict[str, LocalResults] = field(default_factory=dict)

    @property
    def datasets(self) -> list[str]:
        names = set(self.checkpoints) | set(self.local_results)
        if self.instructions is not None:
            names |= set(self.instructions.df["dataset"].unique())
        return sorted(names)


@dataclass
class Run:
    """All rounds of one run (files sharing a run_id)."""

    run_id: float
    rounds: dict[int, RoundArtifacts] = field(default_factory=dict)

    @property
    def round_list(self) -> list[int]:
        return sorted(self.rounds)


@dataclass
class Workspace:
    """Everything discovered under a workspace root."""

    root: Path
    config: dict
    runs: dict[float, Run] = field(default_factory=dict)
    base_datasets: dict[str, Path] = field(default_factory=dict)  # config name -> path
    skipped: list[SkippedFile] = field(default_factory=list)

    def run_ids(self) -> list[float]:
        return sorted(self.runs)
