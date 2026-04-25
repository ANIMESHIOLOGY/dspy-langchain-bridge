"""Unit tests for DSPyRunnable."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from dspy_lc_bridge.runnables import DSPyRunnable, _prediction_to_dict


# ---------------------------------------------------------------------------
# _prediction_to_dict helper
# ---------------------------------------------------------------------------


def test_prediction_to_dict_from_dict():
    assert _prediction_to_dict({"answer": "Paris"}) == {"answer": "Paris"}


def test_prediction_to_dict_from_items():
    pred = MagicMock()
    pred.items = lambda: {"answer": "Paris", "reasoning": "Because."}.items()
    del pred._store  # ensure items() path is taken
    result = _prediction_to_dict(pred)
    assert result["answer"] == "Paris"


def test_prediction_to_dict_from_store():
    pred = MagicMock()
    pred._store = {"answer": "Paris"}
    result = _prediction_to_dict(pred)
    assert result == {"answer": "Paris"}


# ---------------------------------------------------------------------------
# DSPyRunnable.invoke
# ---------------------------------------------------------------------------


def test_invoke_returns_dict(simple_dspy_module):
    runnable = DSPyRunnable(simple_dspy_module)
    result = runnable.invoke({"question": "What is the capital of France?"})
    assert isinstance(result, dict)
    assert "answer" in result
    assert result["answer"] == "Paris"


def test_invoke_calls_module_with_kwargs(simple_dspy_module):
    runnable = DSPyRunnable(simple_dspy_module)
    runnable.invoke({"question": "Hello?"})
    simple_dspy_module.__call__.assert_called_once_with(question="Hello?")


def test_invoke_accepts_aimessage_like_input(simple_dspy_module):
    """If a LangChain AIMessage is passed, content should be extracted."""
    msg = MagicMock()
    msg.content = "extracted text"
    # When passed directly (e.g. from a raw ChatModel output), should not crash
    runnable = DSPyRunnable(simple_dspy_module)
    # Should call module with {"text": "extracted text"}
    result = runnable.invoke(msg)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# DSPyRunnable.ainvoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ainvoke_returns_dict(simple_dspy_module):
    runnable = DSPyRunnable(simple_dspy_module)
    result = await runnable.ainvoke({"question": "async question"})
    assert isinstance(result, dict)
    assert "answer" in result


# ---------------------------------------------------------------------------
# DSPyRunnable.batch
# ---------------------------------------------------------------------------


def test_batch_processes_all_inputs(simple_dspy_module):
    runnable = DSPyRunnable(simple_dspy_module)
    inputs = [{"question": "Q1"}, {"question": "Q2"}, {"question": "Q3"}]
    results = runnable.batch(inputs)
    assert len(results) == 3
    assert all(isinstance(r, dict) for r in results)


# ---------------------------------------------------------------------------
# DSPyRunnable.abatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abatch_processes_all_inputs(simple_dspy_module):
    runnable = DSPyRunnable(simple_dspy_module)
    inputs = [{"question": "Q1"}, {"question": "Q2"}]
    results = await runnable.abatch(inputs)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_repr(simple_dspy_module):
    runnable = DSPyRunnable(simple_dspy_module)
    assert "DSPyRunnable" in repr(runnable)
    assert "MockChainOfThought" in repr(runnable)
