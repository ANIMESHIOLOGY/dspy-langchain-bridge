"""Unit tests for the optimizer bridge."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from dspy_lc_bridge.optimizers import (
    _build_optimized_prompt,
    _extract_optimized_artefacts,
    _normalize_trainset,
    _PromptModule,
    optimize_langchain_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_sig_mock(input_names, output_names):
    sig = MagicMock()
    sig.input_fields = {n: MagicMock() for n in input_names}
    sig.output_fields = {n: MagicMock() for n in output_names}
    sig.__doc__ = "Answer the question."
    return sig


def make_compiled_module(demos=None, instructions=""):
    """Build a mock compiled DSPy module with demos attached to its predictor."""
    compiled = MagicMock()
    predictor = MagicMock()
    predictor.demos = demos or []

    sig = MagicMock()
    sig.__doc__ = instructions
    predictor.signature = sig

    compiled.predictors = MagicMock(return_value=[predictor])
    return compiled


# ---------------------------------------------------------------------------
# _PromptModule
# ---------------------------------------------------------------------------


def test_prompt_module_is_callable():
    sig = make_sig_mock(["question"], ["answer"])
    module = _PromptModule(sig)
    assert callable(module)


def test_prompt_module_named_predictors():
    sig = make_sig_mock(["question"], ["answer"])
    module = _PromptModule(sig)
    named = module.named_predictors()
    assert len(named) == 1
    assert named[0][0] == "predict"


# ---------------------------------------------------------------------------
# _normalize_trainset
# ---------------------------------------------------------------------------


def test_normalize_trainset_dicts():
    sig = make_sig_mock(["question"], ["answer"])
    examples = [{"question": "Q?", "answer": "A."}]
    result = _normalize_trainset(examples, sig)
    assert len(result) == 1


def test_normalize_trainset_preserves_dspy_examples():
    import dspy

    sig = make_sig_mock(["q"], ["a"])
    ex = MagicMock(spec=dspy.Example)
    result = _normalize_trainset([ex], sig)
    assert result[0] is ex


# ---------------------------------------------------------------------------
# _extract_optimized_artefacts
# ---------------------------------------------------------------------------


def test_extract_artefacts_demos():
    demo = MagicMock()
    demo._store = {"question": "Q?", "answer": "A."}
    compiled = make_compiled_module(demos=[demo])
    demos, instructions = _extract_optimized_artefacts(compiled)
    assert len(demos) == 1
    assert demos[0]["question"] == "Q?"


def test_extract_artefacts_instructions():
    compiled = make_compiled_module(instructions="Think step by step.")
    _, instructions = _extract_optimized_artefacts(compiled)
    assert "step" in instructions


def test_extract_artefacts_empty():
    compiled = make_compiled_module(demos=[], instructions="")
    demos, instructions = _extract_optimized_artefacts(compiled)
    assert demos == []
    assert instructions == ""


# ---------------------------------------------------------------------------
# _build_optimized_prompt
# ---------------------------------------------------------------------------


def test_build_optimized_prompt_returns_chat_template():
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

    sig = make_sig_mock(["question"], ["answer"])
    original = PromptTemplate.from_template("{question}")
    result = _build_optimized_prompt(original, demos=[], instructions="", sig=sig)
    assert isinstance(result, ChatPromptTemplate)


def test_build_optimized_prompt_injects_demos():
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

    sig = make_sig_mock(["question"], ["answer"])
    original = PromptTemplate.from_template("{question}")
    demos = [{"question": "What is 2+2?", "answer": "4"}]
    result = _build_optimized_prompt(original, demos=demos, instructions="", sig=sig)
    # System + human demo + assistant demo + live input = 4 messages
    assert len(result.messages) == 4


def test_build_optimized_prompt_injects_instructions():
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

    sig = make_sig_mock(["question"], ["answer"])
    original = PromptTemplate.from_template("{question}")
    result = _build_optimized_prompt(
        original, demos=[], instructions="Think carefully.", sig=sig
    )
    system_msg = result.messages[0].prompt.template
    assert "carefully" in system_msg


# ---------------------------------------------------------------------------
# optimize_langchain_prompt (integration-style with full mocks)
# ---------------------------------------------------------------------------


def test_optimize_langchain_prompt_returns_chat_template():
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

    import dspy

    original = PromptTemplate.from_template("{question}")
    optimizer = MagicMock()
    demo = MagicMock()
    demo._store = {"question": "Q?", "answer": "A."}

    predictor = MagicMock()
    predictor.demos = [demo]
    predictor.signature = MagicMock()
    predictor.signature.__doc__ = ""

    compiled = MagicMock()
    compiled.predictors = MagicMock(return_value=[predictor])
    optimizer.compile = MagicMock(return_value=compiled)

    trainset = [{"question": "What is 2+2?", "answer": "4"}]

    with patch("dspy_lc_bridge.optimizers.dspy.Predict"):
        result = optimize_langchain_prompt(
            prompt=original,
            optimizer=optimizer,
            trainset=trainset,
        )

    assert isinstance(result, ChatPromptTemplate)
    optimizer.compile.assert_called_once()
