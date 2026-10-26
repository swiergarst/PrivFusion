"""Tests for the WillmaLLM client (retries, failure handling, payload)."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests as requests_lib
from langchain_core.messages import HumanMessage, SystemMessage

from privfusion.agents.llms import WillmaLLM


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("WILLMA_API_KEY", "test-key")


def _llm(**kwargs) -> WillmaLLM:
    return WillmaLLM(model_name="test-model", **kwargs)


def _post(status_code: int = 200, body: dict | None = None) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    if body is None:
        body = {"choices": [{"message": {"content": "hello"}}]}
    mock.json.return_value = body
    return mock


class TestWillmaLLM:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("WILLMA_API_KEY", raising=False)
        with pytest.raises(OSError):
            WillmaLLM(model_name="test-model")

    def test_happy_path(self, api_key):
        llm = _llm()
        with patch("privfusion.agents.llms.requests.post", return_value=_post()) as post:
            result = llm.chat([SystemMessage("sys"), HumanMessage("hi")])

        assert result["response"] == "hello"
        assert result["reason"] == ""
        assert post.call_count == 1

    def test_retries_on_timeout_then_succeeds(self, api_key):
        llm = _llm()
        with (
            patch("privfusion.agents.llms.requests.post", side_effect=[requests_lib.Timeout("t"), _post()]) as post,
            patch("privfusion.agents.llms.time.sleep"),
        ):
            result = llm.chat([HumanMessage("hi")])

        assert result["response"] == "hello"
        assert post.call_count == 2

    def test_retries_on_server_error_then_succeeds(self, api_key):
        llm = _llm()
        with (
            patch(
                "privfusion.agents.llms.requests.post",
                side_effect=[_post(status_code=503), _post()],
            ) as post,
            patch("privfusion.agents.llms.time.sleep"),
        ):
            result = llm.chat([HumanMessage("hi")])

        assert result["response"] == "hello"
        assert post.call_count == 2

    def test_raises_after_max_retries(self, api_key):
        llm = _llm(max_retries=3)
        with (
            patch("privfusion.agents.llms.requests.post", side_effect=requests_lib.ConnectionError("down")) as post,
            patch("privfusion.agents.llms.time.sleep"),
            pytest.raises(RuntimeError, match="failed after 3 attempts"),
        ):
            llm.chat([HumanMessage("hi")])

        assert post.call_count == 3

    def test_error_body_retries_then_raises(self, api_key):
        llm = _llm(max_retries=2)
        with (
            patch(
                "privfusion.agents.llms.requests.post",
                return_value=_post(body={"error": "rate limited"}),
            ) as post,
            patch("privfusion.agents.llms.time.sleep"),
            pytest.raises(ValueError, match="error from willma"),
        ):
            llm.chat([HumanMessage("hi")])

        assert post.call_count == 2

    def test_no_content_retries_then_raises(self, api_key):
        llm = _llm(max_retries=2)
        with (
            patch("privfusion.agents.llms.requests.post", return_value=_post(body={"choices": []})) as post,
            patch("privfusion.agents.llms.time.sleep"),
            pytest.raises(ValueError, match="no content"),
        ):
            llm.chat([HumanMessage("hi")])

        assert post.call_count == 2

    def test_role_mapping_and_payload(self, api_key):
        llm = _llm()
        with patch("privfusion.agents.llms.requests.post", return_value=_post()) as post:
            llm.chat([SystemMessage("sys"), HumanMessage("hi")])

        body = json.loads(post.call_args.kwargs["data"])
        assert body["model"] == "test-model"
        assert body["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]

    def test_response_format_passthrough(self, api_key):
        llm = _llm()
        with patch("privfusion.agents.llms.requests.post", return_value=_post()) as post:
            llm.chat([HumanMessage("hi")], response_format={"type": "json_object"})

        body = json.loads(post.call_args.kwargs["data"])
        assert body["response_format"] == {"type": "json_object"}
