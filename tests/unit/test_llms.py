"""Unit tests for LangChainLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from dspy_lc_bridge.llms import LangChainLM, _extract_text, _to_lc_messages

# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


def test_to_lc_messages_converts_user() -> None:
    from langchain_core.messages import HumanMessage

    msgs = _to_lc_messages([{"role": "user", "content": "Hello"}])
    assert len(msgs) == 1
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "Hello"


def test_to_lc_messages_converts_system() -> None:
    from langchain_core.messages import SystemMessage

    msgs = _to_lc_messages([{"role": "system", "content": "You are helpful."}])
    assert isinstance(msgs[0], SystemMessage)


def test_to_lc_messages_converts_assistant() -> None:
    from langchain_core.messages import AIMessage

    msgs = _to_lc_messages([{"role": "assistant", "content": "I help!"}])
    assert isinstance(msgs[0], AIMessage)


def test_extract_text_from_aimessage() -> None:
    msg = AIMessage(content="hello")
    assert _extract_text(msg) == "hello"


def test_extract_text_from_string() -> None:
    assert _extract_text("raw") == "raw"


def test_extract_text_from_text_attr() -> None:
    obj = MagicMock()
    obj.content = None
    del obj.content
    obj.text = "via text"
    assert _extract_text(obj) == "via text"


# ---------------------------------------------------------------------------
# LangChainLM with chat model
# ---------------------------------------------------------------------------


def test_call_with_prompt(mock_chat_llm) -> None:
    lm = LangChainLM(mock_chat_llm)
    result = lm(prompt="What is 2+2?")
    assert isinstance(result, list)
    assert len(result) == 1
    assert "42" in result[0]


def test_call_with_messages(mock_chat_llm) -> None:
    lm = LangChainLM(mock_chat_llm)
    result = lm(messages=[{"role": "user", "content": "Hi"}])
    assert isinstance(result, list)
    assert len(result) == 1


def test_call_raises_without_prompt_or_messages(mock_chat_llm) -> None:
    lm = LangChainLM(mock_chat_llm)
    with pytest.raises(ValueError, match="Either"):
        lm()


def test_history_is_recorded(mock_chat_llm) -> None:
    lm = LangChainLM(mock_chat_llm)
    lm(prompt="Q1")
    lm(prompt="Q2")
    assert len(lm.history) == 2
    assert lm.history[0]["messages"] == [{"role": "user", "content": "Q1"}]


def test_inspect_history_returns_last_n(mock_chat_llm) -> None:
    lm = LangChainLM(mock_chat_llm)
    for i in range(5):
        lm(prompt=f"Q{i}")
    last_2 = lm.inspect_history(n=2)
    assert len(last_2) == 2


def test_usage_metadata(mock_chat_llm) -> None:
    lm = LangChainLM(mock_chat_llm)
    lm(prompt="test")
    meta = lm.get_usage_metadata()
    assert meta["total_calls"] == 1


def test_repr_chat_model(mock_chat_llm) -> None:
    with patch.object(LangChainLM, "_detect_chat_model", return_value=True):
        lm = LangChainLM(mock_chat_llm)
    assert "LangChainLM" in repr(lm)
    assert "is_chat=True" in repr(lm)


# ---------------------------------------------------------------------------
# LangChainLM with completion model (non-chat)
# ---------------------------------------------------------------------------


def test_call_completion_model(mock_completion_llm) -> None:
    # Force non-chat detection
    with patch.object(LangChainLM, "_detect_chat_model", return_value=False):
        lm = LangChainLM(mock_completion_llm)
        result = lm(prompt="What is 2+2?")
    assert isinstance(result, list)
    assert len(result) == 1


def test_temperature_and_max_tokens_forwarded(mock_chat_llm) -> None:
    lm = LangChainLM(mock_chat_llm, temperature=0.0, max_tokens=100)
    lm(prompt="test")
    call_kwargs = mock_chat_llm.invoke.call_args[1]
    assert call_kwargs.get("temperature") == 0.0
    assert call_kwargs.get("max_tokens") == 100
