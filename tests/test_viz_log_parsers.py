"""Tests for the per-round .log parsers (new format only)."""

from pathlib import Path

from viz import log_parsers


def _rec(ts: str, msg: str, level: str = "INFO") -> str:
    return f"{ts} testLogger {level} {msg}"


# --- federated-harmonization logs ------------------------------------------


def test_parse_fed_log_pairs_inputs_and_responses(tmp_path: Path) -> None:
    p = tmp_path / "fed.log"
    p.write_text(
        _rec("2026-10-19 10:00:00,000", "input messages: [{'role': 'system', 'content': 'hi'}]")
        + "\n"
        + _rec(
            "2026-10-19 10:00:02,000",
            "response: {'model': 'm', 'choices': [{'message': {'content': 'done', 'reasoning': 'think'}}], "
            "'usage': {'prompt_tokens': 10, 'completion_tokens': 20}, 'surf': {'compute_time_s': 1.5}}",
        )
        + "\n",
        encoding="utf-8",
    )

    log = log_parsers.parse_fed_log(p)

    assert len(log.calls) == 1
    call = log.calls[0]
    assert call.parse_ok
    assert call.input_messages == [{"role": "system", "content": "hi"}]
    assert call.model == "m"
    assert call.response_text == "done"
    assert call.reasoning == "think"
    assert (call.prompt_tokens, call.completion_tokens) == (10, 20)
    assert call.compute_time_s == 1.5
    assert call.input_ts is not None
    assert call.response_ts is not None
    assert log.other_records == []


def test_parse_fed_log_malformed_payload_keeps_raw(tmp_path: Path) -> None:
    p = tmp_path / "fed.log"
    p.write_text(
        _rec("2026-10-19 10:00:00,000", "input messages: [{'role': 'user', 'content': 'unterminated")
        + "\n"
        + _rec("2026-10-19 10:00:02,000", "response: not a dict")
        + "\n",
        encoding="utf-8",
    )

    log = log_parsers.parse_fed_log(p)

    assert len(log.calls) == 1
    assert not log.calls[0].parse_ok
    assert log.calls[0].input_raw.startswith("[{'role'")
    assert log.calls[0].response_raw == "not a dict"
    assert log.calls[0].response_text == ""


def test_parse_fed_log_ignores_unrelated_records(tmp_path: Path) -> None:
    p = tmp_path / "fed.log"
    p.write_text(
        _rec("2026-10-19 10:00:00,000", "some setup message")
        + "\n"
        + _rec("2026-10-19 10:00:01,000", "input messages: [{'role': 'user', 'content': 'x'}]")
        + "\n"
        + _rec("2026-10-19 10:00:02,000", "response: {'model': 'm'}")
        + "\n",
        encoding="utf-8",
    )

    log = log_parsers.parse_fed_log(p)

    assert len(log.calls) == 1
    assert log.other_records == ["some setup message"]


# --- local-modification logs -------------------------------------------------


def test_parse_local_log_sessions_tools_and_attribution(tmp_path: Path) -> None:
    p = tmp_path / "local.log"
    lines = [
        _rec(
            "2026-10-19 11:00:00,000",
            "[initial request size] system_prompt=100 chars, tools=5 (200 chars), user_msg=300 chars, TOTAL ≈ 600 chars",
        ),
        _rec("2026-10-19 11:00:01,000", ">>> LLM REQUEST <<<"),
        "Columns/dtypes:",
        "a    int64",
        "b    str",
        "",
        "Sample rows:",
        "a,b",
        "1,x",
        _rec("2026-10-19 11:00:05,000", "tool call: explore(code='print(df)')"),
        _rec("2026-10-19 11:00:05,500", "tool result: explore -> ok"),
        _rec(
            "2026-10-19 11:00:06,000",
            "tool call: apply_transformation(description='rename', code='df = df.rename(columns=dict(a=aa))')",
        ),
        _rec("2026-10-19 11:00:06,500", "tool result: apply_transformation -> OK. Committed step 1."),
        _rec("2026-10-19 11:00:07,000", ">>> LLM RESPONSE <<<"),
        "final answer text",
        _rec("2026-10-19 11:00:08,000", "--------------------------------------------------------------"),
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log = log_parsers.parse_local_log(p, candidates={"ds-a": ["a", "b"], "ds-b": ["c", "d"]})

    assert len(log.sessions) == 1
    s = log.sessions[0]
    assert s.dataset == "ds-a"
    assert s.attribution_exact
    assert s.request_chars == {"system_prompt": 100, "tools": 5, "tools_chars": 200, "user_msg": 300, "total": 600}
    assert [e.name for e in s.tool_events] == ["explore", "apply_transformation"]
    assert s.tool_events[0].has_result
    assert s.tool_events[0].result_preview == "ok"
    assert "Committed" in s.tool_events[1].result_preview
    assert s.response_text == "final answer text"
    assert s.tool_call_counts == {"explore": 1, "apply_transformation": 1}


def test_parse_local_log_heuristic_attribution(tmp_path: Path) -> None:
    p = tmp_path / "local.log"
    lines = [
        _rec("2026-10-19 11:00:01,000", ">>> LLM REQUEST <<<"),
        "Columns/dtypes:",
        "a    int64",
        "c    str",
        "",
        _rec("2026-10-19 11:00:07,000", ">>> LLM RESPONSE <<<"),
        "resp 1",
        _rec("2026-10-19 11:01:01,000", ">>> LLM REQUEST <<<"),
        "Columns/dtypes:",
        "a    int64",
        "z    str",
        "w    str",
        "",
        _rec("2026-10-19 11:01:07,000", ">>> LLM RESPONSE <<<"),
        "resp 2",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log = log_parsers.parse_local_log(p, candidates={"ds-x": ["a", "c"], "ds-y": ["a", "z"]})

    assert [s.dataset for s in log.sessions] == ["ds-x", "ds-y"]
    assert log.sessions[0].attribution_exact  # exact set match
    assert not log.sessions[1].attribution_exact  # jaccard fallback


def test_parse_local_log_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "local.log"
    p.write_text("", encoding="utf-8")

    log = log_parsers.parse_local_log(p)

    assert log.sessions == []
