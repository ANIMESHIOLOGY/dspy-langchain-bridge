"""Unit tests for signature ↔ PromptTemplate conversion."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dspy_lc_bridge.signatures import (
    _get_field_desc,
    prompt_to_signature,
    signature_to_chat_prompt,
    signature_to_prompt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sig(input_names, output_names, doc="Answer the question."):
    """Build a minimal mock DSPy Signature."""
    sig = MagicMock()
    sig.__doc__ = doc
    sig.input_fields = {
        name: MagicMock(json_schema_extra={"desc": f"The {name}"}) for name in input_names
    }
    sig.output_fields = {
        name: MagicMock(json_schema_extra={"desc": f"The {name}"}) for name in output_names
    }
    return sig


# ---------------------------------------------------------------------------
# _get_field_desc
# ---------------------------------------------------------------------------


def test_get_field_desc_from_json_schema_extra() -> None:
    field = MagicMock()
    field.json_schema_extra = {"desc": "My description"}
    assert _get_field_desc(field) == "My description"


def test_get_field_desc_from_dict() -> None:
    assert _get_field_desc({"desc": "dict desc"}) == "dict desc"


def test_get_field_desc_empty() -> None:
    field = MagicMock()
    field.json_schema_extra = {}
    assert _get_field_desc(field) == ""


# ---------------------------------------------------------------------------
# signature_to_prompt
# ---------------------------------------------------------------------------


def test_signature_to_prompt_single_input() -> None:
    sig = make_sig(["question"], ["answer"])
    prompt = signature_to_prompt(sig)
    assert "question" in prompt.input_variables
    assert "{question}" in prompt.template


def test_signature_to_prompt_multiple_inputs() -> None:
    sig = make_sig(["context", "question"], ["answer"])
    prompt = signature_to_prompt(sig)
    assert set(prompt.input_variables) == {"context", "question"}


def test_signature_to_prompt_includes_descriptions() -> None:
    sig = make_sig(["question"], ["answer"])
    prompt = signature_to_prompt(sig)
    # Description comment should appear above the variable placeholder
    assert "# The question" in prompt.template


def test_signature_to_prompt_output_hint() -> None:
    sig = make_sig(["question"], ["answer"])
    prompt = signature_to_prompt(sig, include_output_hint=True)
    assert "answer" in prompt.template


def test_signature_to_prompt_raises_on_invalid() -> None:
    with pytest.raises(TypeError):
        signature_to_prompt("not a signature")


# ---------------------------------------------------------------------------
# signature_to_chat_prompt
# ---------------------------------------------------------------------------


def test_signature_to_chat_prompt_returns_chat_template() -> None:
    from langchain_core.prompts import ChatPromptTemplate

    sig = make_sig(["question"], ["answer"])
    chat_prompt = signature_to_chat_prompt(sig)
    assert isinstance(chat_prompt, ChatPromptTemplate)


def test_signature_to_chat_prompt_system_uses_docstring() -> None:
    sig = make_sig(["question"], ["answer"], doc="You are an expert.")
    chat_prompt = signature_to_chat_prompt(sig)
    # The system message should contain the docstring
    system_template = chat_prompt.messages[0].prompt.template
    assert "expert" in system_template


def test_signature_to_chat_prompt_has_human_message() -> None:
    sig = make_sig(["question"], ["answer"])
    chat_prompt = signature_to_chat_prompt(sig)
    assert len(chat_prompt.messages) == 2
    assert "{question}" in chat_prompt.messages[1].prompt.template


# ---------------------------------------------------------------------------
# prompt_to_signature
# ---------------------------------------------------------------------------


def test_prompt_to_signature_basic() -> None:
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate.from_template("Answer: {question}")
    sig = prompt_to_signature(prompt)
    assert "question" in sig.input_fields


def test_prompt_to_signature_multiple_vars() -> None:
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate.from_template("{context}\n{question}")
    sig = prompt_to_signature(prompt, output_fields=["answer"])
    assert "context" in sig.input_fields
    assert "question" in sig.input_fields
    assert "answer" in sig.output_fields


def test_prompt_to_signature_default_output() -> None:
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate.from_template("{question}")
    sig = prompt_to_signature(prompt)
    assert "answer" in sig.output_fields


def test_prompt_to_signature_custom_output_fields() -> None:
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate.from_template("{document}")
    sig = prompt_to_signature(prompt, output_fields=["summary", "title"])
    assert "summary" in sig.output_fields
    assert "title" in sig.output_fields


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------


def test_round_trip_signature_prompt_signature() -> None:
    sig = make_sig(["context", "question"], ["answer"])
    prompt = signature_to_prompt(sig)
    sig2 = prompt_to_signature(prompt, output_fields=["answer"])
    assert set(sig2.input_fields.keys()) == {"context", "question"}
    assert "answer" in sig2.output_fields
