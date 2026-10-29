import json

import httpx
import pytest

from llm_hiring_audit.backends import GeminiBackend, MistralBackend, MockBackend, OllamaBackend


def _client(handler: httpx.MockTransport, base_url: str) -> httpx.Client:
    return httpx.Client(transport=handler, base_url=base_url)


def test_ollama_request_and_response() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "72"}})

    backend = OllamaBackend("mistral", client=_client(httpx.MockTransport(handler), "http://x"))
    assert backend.complete("bonjour", temperature=0.5, seed=3) == "72"
    assert backend.name == "ollama/mistral"
    body = seen["body"]
    assert isinstance(body, dict)
    assert seen["path"] == "/api/chat"
    assert body["options"] == {"temperature": 0.5, "seed": 3}
    assert body["messages"][-1] == {"role": "user", "content": "bonjour"}


def test_mistral_request_and_response() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "55"}}]})

    client = _client(httpx.MockTransport(handler), "https://api.mistral.ai")
    backend = MistralBackend("mistral-small-latest", api_key="test", client=client)
    assert backend.complete("bonjour", temperature=0.2, seed=1) == "55"
    body = seen["body"]
    assert isinstance(body, dict)
    assert seen["path"] == "/v1/chat/completions"
    assert body["random_seed"] == 1


def test_mistral_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        MistralBackend("mistral-small-latest")


def test_gemini_request_and_response() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        parts = [{"text": "thinking...", "thought": True}, {"text": "64"}]
        return httpx.Response(200, json={"candidates": [{"content": {"parts": parts}}]})

    client = _client(httpx.MockTransport(handler), "https://generativelanguage.googleapis.com")
    backend = GeminiBackend("gemini-2.5-flash", api_key="test", client=client)
    # The thought part is ignored: only the visible answer is returned.
    assert backend.complete("bonjour", temperature=0.7, seed=2) == "64"
    assert backend.name == "gemini/gemini-2.5-flash"
    body = seen["body"]
    assert isinstance(body, dict)
    assert seen["path"] == "/v1beta/models/gemini-2.5-flash:generateContent"
    assert body["generationConfig"] == {"temperature": 0.7, "seed": 2}
    assert body["contents"][0]["parts"][0]["text"] == "bonjour"


def test_gemini_blocked_prompt_returns_empty_answer() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"promptFeedback": {"blockReason": "OTHER"}})
    )
    client = _client(transport, "https://generativelanguage.googleapis.com")
    backend = GeminiBackend("gemini-2.5-flash", api_key="test", client=client)
    assert backend.complete("bonjour", temperature=0.7, seed=0) == ""


def test_gemini_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiBackend("gemini-2.5-flash")


def test_http_errors_are_raised() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(429))
    backend = OllamaBackend("m", client=_client(transport, "http://x"))
    with pytest.raises(httpx.HTTPStatusError):
        backend.complete("x", temperature=0, seed=0)


def test_mock_is_reproducible() -> None:
    mock = MockBackend()
    assert mock.complete("abc", temperature=0, seed=1) == mock.complete(
        "abc", temperature=0, seed=1
    )


def test_mock_applies_planted_penalty() -> None:
    mock = MockBackend(penalties={"Sevran": -20}, noise_sd=0)
    assert mock.complete("CV, Sevran", temperature=0, seed=0) == "50"
    assert mock.complete("CV, Paris", temperature=0, seed=0) == "70"
