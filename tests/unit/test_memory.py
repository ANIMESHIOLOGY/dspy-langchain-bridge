"""Unit tests for LangChainMemoryAdapter."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from dspy_lc_bridge.memory import LangChainMemoryAdapter

# ---------------------------------------------------------------------------
# get_history
# ---------------------------------------------------------------------------


def test_get_history_returns_list(mock_lc_memory) -> None:
    adapter = LangChainMemoryAdapter(mock_lc_memory)
    history = adapter.get_history()
    assert isinstance(history, list)


def test_get_history_correct_roles(mock_lc_memory) -> None:
    adapter = LangChainMemoryAdapter(mock_lc_memory)
    history = adapter.get_history()
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_get_history_correct_content(mock_lc_memory) -> None:
    adapter = LangChainMemoryAdapter(mock_lc_memory)
    history = adapter.get_history()
    assert "France" in history[0]["content"]
    assert "Paris" in history[1]["content"]


def test_get_history_custom_roles(mock_lc_memory) -> None:
    adapter = LangChainMemoryAdapter(mock_lc_memory, human_role="human", ai_role="ai")
    history = adapter.get_history()
    assert history[0]["role"] == "human"
    assert history[1]["role"] == "ai"


# ---------------------------------------------------------------------------
# system messages
# ---------------------------------------------------------------------------


def test_get_history_system_message() -> None:
    memory = MagicMock()
    memory.chat_memory = MagicMock()
    memory.chat_memory.messages = [SystemMessage(content="You are helpful.")]
    adapter = LangChainMemoryAdapter(memory)
    history = adapter.get_history()
    assert history[0]["role"] == "system"
    assert "helpful" in history[0]["content"]


# ---------------------------------------------------------------------------
# save_turn
# ---------------------------------------------------------------------------


def test_save_turn_calls_save_context(mock_lc_memory) -> None:
    adapter = LangChainMemoryAdapter(mock_lc_memory)
    adapter.save_turn("What is 2+2?", "4")
    mock_lc_memory.save_context.assert_called_once_with({"input": "What is 2+2?"}, {"output": "4"})


def test_save_turn_no_method_does_not_raise() -> None:
    memory = MagicMock(spec=[])  # No save_context method
    adapter = LangChainMemoryAdapter(memory)
    adapter.save_turn("Q", "A")  # Should not raise


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_calls_memory_clear(mock_lc_memory) -> None:
    adapter = LangChainMemoryAdapter(mock_lc_memory)
    adapter.clear()
    mock_lc_memory.clear.assert_called_once()


# ---------------------------------------------------------------------------
# Fallback: load_memory_variables path
# ---------------------------------------------------------------------------


def test_get_history_from_load_memory_variables() -> None:
    memory = MagicMock(spec=["load_memory_variables"])
    memory.load_memory_variables = MagicMock(
        return_value={"history": [HumanMessage(content="Hi"), AIMessage(content="Hello!")]}
    )
    adapter = LangChainMemoryAdapter(memory)
    history = adapter.get_history()
    assert len(history) == 2


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_repr(mock_lc_memory) -> None:
    adapter = LangChainMemoryAdapter(mock_lc_memory)
    assert "LangChainMemoryAdapter" in repr(adapter)
