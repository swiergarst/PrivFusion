"""Parsers for the per-round ``.log`` files written by ``main.py``.

Both loggers use the same format (``setup_round_logging`` in ``main.py``):
``%(asctime)s %(name)s %(levelname)s %(message)s``. A record starts at each
line carrying that prefix; every following line without a prefix is a
continuation of the previous record's message (multi-line messages only
occur in the local logs, e.g. the ``>>> LLM REQUEST <<<`` block).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from viz.model import AgentSession, FedCall, FedLog, LocalLog, ToolEvent

RECORD_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) (?P<name>\S+) (?P<level>[A-Z]+) (?P<msg>.*)$"
)
_TS_FORMAT = "%Y-%m-%d %H:%M:%S,%f"

REQUEST_SIZE_RE = re.compile(
    r"\[(?P<label>[\w ]+) request size\] system_prompt=(?P<system_prompt>\d+) chars, "
    r"tools=(?P<tools>\d+) \((?P<tools_chars>\d+) chars\), "
    r"user_msg=(?P<user_msg>\d+) chars, TOTAL ≈ (?P<total>\d+) chars"
)
TOOL_CALL_RE = re.compile(r"^tool call: (?P<name>\w+)\((?P<args>.*)\)\s*$", re.DOTALL)
TOOL_RESULT_RE = re.compile(r"^tool result: (?P<name>\w+) -> (?P<result>.*)\s*$", re.DOTALL)

INPUT_PREFIX = "input messages: "
RESPONSE_PREFIX = "response: "
REQUEST_MARKER = ">>> LLM REQUEST <<<"
RESPONSE_MARKER = ">>> LLM RESPONSE <<<"


@dataclass
class LogRecord:
    """One log record: a prefixed line plus its continuation lines."""

    ts: datetime | None
    level: str | None
    message: str


def split_records(text: str) -> list[LogRecord]:
    """Split raw log text into records (see module docstring)."""
    records: list[LogRecord] = []
    for line in text.splitlines():
        m = RECORD_RE.match(line)
        if m:
            records.append(LogRecord(ts=datetime.strptime(m["ts"], _TS_FORMAT), level=m["level"], message=m["msg"]))
        elif records:
            records[-1].message += "\n" + line
        else:
            # Defensive: leading line without a timestamp prefix.
            records.append(LogRecord(ts=None, level=None, message=line))
    return records


def _literal_eval(raw: str):
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def _decode_fed_call(call: FedCall) -> None:
    call.input_messages = _literal_eval(call.input_raw)
    call.response = _literal_eval(call.response_raw)
    call.parse_ok = call.input_messages is not None and call.response is not None
    if isinstance(call.response, dict):
        call.model = str(call.response.get("model", ""))
        usage = call.response.get("usage") or {}
        call.prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        call.completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        surf = call.response.get("surf") or {}
        call.compute_time_s = float(surf.get("compute_time_s", 0.0) or 0.0)


def parse_fed_log(path: Path) -> FedLog:
    """Parse a federated-harmonization log into input/response call pairs.

    An ``input messages`` record opens a call and the next ``response``
    record closes it; orphans on either side are kept as best-effort calls
    or listed in ``other_records``.
    """
    log = FedLog(path=path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return log

    pending: FedCall | None = None
    for rec in split_records(text):
        msg = rec.message
        if msg.startswith(INPUT_PREFIX):
            if pending is not None:
                log.calls.append(pending)  # input that never got a response
            pending = FedCall(input_raw=msg[len(INPUT_PREFIX) :], input_ts=rec.ts)
        elif msg.startswith(RESPONSE_PREFIX):
            if pending is None:
                log.other_records.append(msg)
                continue
            pending.response_raw = msg[len(RESPONSE_PREFIX) :]
            pending.response_ts = rec.ts
            _decode_fed_call(pending)
            log.calls.append(pending)
            pending = None
        else:
            log.other_records.append(msg)
    if pending is not None:
        log.calls.append(pending)
    return log


def _prompt_columns(user_prompt: str) -> list[str] | None:
    """Column names from the ``Columns/dtypes:`` block of an agent user prompt."""
    lines = user_prompt.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "Columns/dtypes:"), None)
    if start is None:
        return None
    cols: list[str] = []
    for line in lines[start + 1 :]:
        if not line.strip() or line.strip() == "Sample rows:":
            break
        if " " not in line and "\t" not in line:
            break
        name, _dtype = line.rsplit(None, 1)
        cols.append(name)
    return cols or None


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def parse_local_log(path: Path, candidates: dict[str, list[str]] | None = None) -> LocalLog:
    """Parse a per-round local-modification log into agent sessions.

    One session = one ``>>> LLM REQUEST <<<`` ... ``>>> LLM RESPONSE <<<``
    block plus the tool events in between. If ``candidates`` (dataset name ->
    column list of that dataset's *input* file for the round) is given, each
    session is attributed to a dataset by matching its ``Columns/dtypes``
    block; ``AgentSession.attribution_exact`` is False when the match was
    approximate.
    """
    log = LocalLog(path=path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return log

    session: AgentSession | None = None
    pending_size: dict[str, int] | None = None
    open_calls: list[ToolEvent] = []
    index = 0

    for rec in split_records(text):
        msg = rec.message
        if msg.startswith(REQUEST_MARKER):
            session = AgentSession(
                index=index,
                user_prompt=msg[len(REQUEST_MARKER) :].lstrip("\n"),
                request_ts=rec.ts,
                request_chars=pending_size or {},
            )
            pending_size = None
            open_calls = []
            index += 1
            log.sessions.append(session)
        elif msg.startswith(RESPONSE_MARKER):
            if session is not None:
                session.response_text = msg[len(RESPONSE_MARKER) :].lstrip("\n")
                session.response_ts = rec.ts
            session = None
        elif size := REQUEST_SIZE_RE.search(msg):
            pending_size = {k: int(v) for k, v in size.groupdict().items() if k != "label"}
        elif call := TOOL_CALL_RE.match(msg):
            if session is not None:
                ev = ToolEvent(name=call["name"], args_preview=call["args"], call_ts=rec.ts)
                session.tool_events.append(ev)
                open_calls.append(ev)
        elif result := TOOL_RESULT_RE.match(msg):
            for ev in reversed(open_calls):
                if ev.name == result["name"] and not ev.has_result:
                    ev.result_preview = result["result"]
                    ev.result_ts = rec.ts
                    ev.has_result = True
                    break
        # separators, debug lines and stray text are intentionally ignored

    if candidates:
        for s in log.sessions:
            cols = _prompt_columns(s.user_prompt)
            if cols is None:
                continue
            exact = next((name for name, cand in candidates.items() if set(cand) == set(cols)), None)
            if exact is not None:
                s.dataset = exact
                s.attribution_exact = True
            else:
                best, best_score = "", 0.0
                for name, cand in candidates.items():
                    if (score := _jaccard(set(cols), set(cand))) > best_score:
                        best, best_score = name, score
                s.dataset = best
    return log
