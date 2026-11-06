"""Free-model rotation: fallbacks on removed/rate-limited models, no retry on auth."""
from __future__ import annotations

import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.generate import FREE_POOL, OpenAILLM, rotation_list  # noqa: E402


class _FailThenWin:
    """Stub chat.completions: raise `exc` `fails` times, then return text."""

    def __init__(self, exc: BaseException, fails: int, text: str = "ok [1]") -> None:
        self.exc = exc
        self.fails = fails
        self.text = text
        self.calls: list[str] = []

    def create(self, **kw):  # noqa: ANN001, ANN202
        self.calls.append(kw["model"])
        if len(self.calls) <= self.fails:
            raise self.exc
        msg = types.SimpleNamespace(content=self.text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _llm(exc: BaseException, fails: int, models: list[str]) -> tuple[OpenAILLM, _FailThenWin]:
    llm = OpenAILLM(api_key="dummy", model=models[0], models=models)
    stub = _FailThenWin(exc, fails)
    llm._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=stub))
    return llm, stub


def _status_exc(code: int, msg: str = "") -> BaseException:
    exc = RuntimeError(f"http {code} {msg}".strip())
    exc.status_code = code  # type: ignore[attr-defined]
    return exc


def test_rotation_list_primary_first_deduped() -> None:
    assert rotation_list("beam-free", "")[0] == "beam-free"
    assert len(rotation_list("beam-free", "")) == len(FREE_POOL)
    assert rotation_list("x", "a,b,a") == ["x", "a", "b"]
    assert "beam-free" in FREE_POOL


def test_complete_falls_back_on_404() -> None:
    llm, stub = _llm(_status_exc(404), 1, ["gone-model", "beam-free"])
    assert llm.complete("s", "u") == "ok [1]"
    assert stub.calls == ["gone-model", "beam-free"]
    assert llm.name == "beam-free"


def test_complete_falls_back_on_429_and_500() -> None:
    llm, stub = _llm(_status_exc(429), 1, ["a", "b"])
    assert llm.complete("s", "u") == "ok [1]"
    assert stub.calls == ["a", "b"]
    llm2, stub2 = _llm(_status_exc(500), 2, ["a", "b", "c"])
    assert llm2.complete("s", "u") == "ok [1]"
    assert stub2.calls == ["a", "b", "c"]


def test_complete_retries_400_model_gone_not_param_error() -> None:
    llm, stub = _llm(_status_exc(400, "Model is unavailable"), 1, ["gone", "beam-free"])
    assert llm.complete("s", "u") == "ok [1]"
    assert stub.calls == ["gone", "beam-free"]
    llm2, stub2 = _llm(_status_exc(400, "invalid temperature"), 99, ["a", "b"])
    try:
        llm2.complete("s", "u")
    except RuntimeError:
        pass
    else:
        raise AssertionError("should raise")
    assert stub2.calls == ["a"]


def test_complete_no_retry_on_401() -> None:
    llm, stub = _llm(_status_exc(401), 99, ["a", "b"])
    try:
        llm.complete("s", "u")
    except RuntimeError:
        pass
    else:
        raise AssertionError("should raise")
    assert stub.calls == ["a"]


def test_complete_raises_last_when_all_fail() -> None:
    llm, stub = _llm(_status_exc(404), 99, ["a", "b"])
    try:
        llm.complete("s", "u")
    except RuntimeError:
        pass
    else:
        raise AssertionError("should raise")
    assert stub.calls == ["a", "b"]
