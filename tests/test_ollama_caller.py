"""Tests for ``auxima_ai.intake.ollama.OllamaLLMCaller``.

Uses ``httpx.MockTransport`` so the test suite has zero network
dependence — the same caller code path runs as in production.

Coverage:
  - Happy path: success response -> normalised LLMResponse with the
    structured payload, token counts, ms-converted latency.
  - "ollama/qwen2.5:32b" model_id has the "ollama/" prefix stripped
    before being sent to the Ollama API.
  - prompt_eval_count + eval_count map to prompt/completion tokens.
  - total_duration in ns is converted to ms.
  - Missing model field falls back to the request model_id.
  - 4xx / 5xx responses raise OllamaBadStatusError with status_code.
  - Timeout raises OllamaTimeoutError.
  - Connection error raises OllamaConnectionError.
  - Non-JSON response body raises OllamaMalformedResponseError.
  - Inner "response" not parseable as JSON raises OllamaMalformedResponseError.
  - Inner payload not an object raises.
  - Negative / non-int token counts rejected.
  - Construction validation rejects empty base_url / non-positive timeout.
  - Caller is usable as a context manager.
"""
from __future__ import annotations

import json

import httpx
import pytest

from auxima_ai.intake.ollama import (
    OllamaBadStatusError,
    OllamaConnectionError,
    OllamaError,
    OllamaLLMCaller,
    OllamaMalformedResponseError,
    OllamaTimeoutError,
)


def _ok_payload(
    *,
    inner: dict | None = None,
    model: str = "qwen2.5:32b",
    prompt_eval_count: int = 100,
    eval_count: int = 50,
    total_duration: int = 42_000_000,  # 42 ms in ns
) -> dict:
    return {
        "model": model,
        "response": json.dumps(inner if inner is not None else {"ok": True, "value": 1}),
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "total_duration": total_duration,
        "done": True,
    }


def _mock_caller(handler) -> OllamaLLMCaller:
    return OllamaLLMCaller(
        base_url="http://localhost:11434",
        transport=httpx.MockTransport(handler),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_call_returns_normalised_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        body = json.loads(request.content)
        assert body["model"] == "qwen2.5:32b"  # prefix stripped
        assert body["prompt"] == "hello"
        assert body["format"] == "json"
        assert body["stream"] is False
        return httpx.Response(200, json=_ok_payload())

    with _mock_caller(handler) as caller:
        r = caller.call(model_id="ollama/qwen2.5:32b", prompt="hello")
    assert r.payload == {"ok": True, "value": 1}
    assert r.prompt_tokens == 100
    assert r.completion_tokens == 50
    assert r.latency_ms == 42
    assert r.model_version == "qwen2.5:32b"


def test_model_id_without_prefix_passes_through() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "llama3.1:8b"
        return httpx.Response(200, json=_ok_payload(model="llama3.1:8b"))

    with _mock_caller(handler) as caller:
        r = caller.call(model_id="llama3.1:8b", prompt="x")
    assert r.model_version == "llama3.1:8b"


def test_total_duration_ns_converted_to_ms() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_payload(total_duration=1_500_000_000))  # 1.5s

    with _mock_caller(handler) as caller:
        r = caller.call(model_id="ollama/x:7b", prompt="x")
    assert r.latency_ms == 1500


def test_missing_model_field_falls_back_to_request_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _ok_payload()
        del payload["model"]
        return httpx.Response(200, json=payload)

    with _mock_caller(handler) as caller:
        r = caller.call(model_id="ollama/qwen2.5:32b", prompt="x")
    assert r.model_version == "ollama/qwen2.5:32b"


def test_missing_token_counts_default_to_zero() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _ok_payload()
        del payload["prompt_eval_count"]
        del payload["eval_count"]
        del payload["total_duration"]
        return httpx.Response(200, json=payload)

    with _mock_caller(handler) as caller:
        r = caller.call(model_id="ollama/x:7b", prompt="x")
    assert r.prompt_tokens == 0
    assert r.completion_tokens == 0
    assert r.latency_ms == 0


# ---------------------------------------------------------------------------
# HTTP errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [400, 404, 422, 500, 502, 503])
def test_non_2xx_status_raises_bad_status_error(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=f"bad: {status_code}")

    with _mock_caller(handler) as caller:
        with pytest.raises(OllamaBadStatusError) as exc_info:
            caller.call(model_id="ollama/x:7b", prompt="x")
    assert exc_info.value.status_code == status_code


def test_timeout_raises_ollama_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    with _mock_caller(handler) as caller:
        with pytest.raises(OllamaTimeoutError):
            caller.call(model_id="ollama/x:7b", prompt="x")


def test_connection_error_raises_ollama_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with _mock_caller(handler) as caller:
        with pytest.raises(OllamaConnectionError):
            caller.call(model_id="ollama/x:7b", prompt="x")


# ---------------------------------------------------------------------------
# Malformed response
# ---------------------------------------------------------------------------


def test_non_json_response_body_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    with _mock_caller(handler) as caller:
        with pytest.raises(OllamaMalformedResponseError):
            caller.call(model_id="ollama/x:7b", prompt="x")


def test_inner_response_not_parseable_as_json_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _ok_payload()
        payload["response"] = "this isn't JSON"
        return httpx.Response(200, json=payload)

    with _mock_caller(handler) as caller:
        with pytest.raises(OllamaMalformedResponseError):
            caller.call(model_id="ollama/x:7b", prompt="x")


def test_inner_response_not_an_object_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _ok_payload()
        payload["response"] = json.dumps([1, 2, 3])  # array, not object
        return httpx.Response(200, json=payload)

    with _mock_caller(handler) as caller:
        with pytest.raises(OllamaMalformedResponseError, match="object"):
            caller.call(model_id="ollama/x:7b", prompt="x")


def test_missing_response_field_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "x", "done": True})

    with _mock_caller(handler) as caller:
        with pytest.raises(OllamaMalformedResponseError):
            caller.call(model_id="ollama/x:7b", prompt="x")


@pytest.mark.parametrize("bad", [-1, "5", 1.5, True])
def test_negative_or_non_int_token_count_raises(bad: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _ok_payload()
        payload["prompt_eval_count"] = bad
        return httpx.Response(200, json=payload)

    with _mock_caller(handler) as caller:
        with pytest.raises(OllamaMalformedResponseError):
            caller.call(model_id="ollama/x:7b", prompt="x")


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   "])
def test_construction_rejects_empty_base_url(bad: str) -> None:
    with pytest.raises(OllamaError, match="base_url"):
        OllamaLLMCaller(base_url=bad)


@pytest.mark.parametrize("bad", [0, -1, -0.001])
def test_construction_rejects_non_positive_timeout(bad: float) -> None:
    with pytest.raises(OllamaError, match="timeout"):
        OllamaLLMCaller(base_url="http://localhost:11434", timeout_seconds=bad)


def test_call_rejects_non_string_prompt() -> None:
    with _mock_caller(lambda r: httpx.Response(200, json=_ok_payload())) as caller:
        with pytest.raises(OllamaError, match="prompt"):
            caller.call(model_id="ollama/x:7b", prompt=42)  # type: ignore[arg-type]


def test_call_rejects_empty_model_id() -> None:
    with _mock_caller(lambda r: httpx.Response(200, json=_ok_payload())) as caller:
        with pytest.raises(OllamaError, match="model_id"):
            caller.call(model_id="", prompt="x")
