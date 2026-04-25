"""Unit tests for DSPyTool, LangChainTool, and DSPyNode."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from dspy_lc_bridge.tools import DSPyNode, DSPyTool, LangChainTool, _snake_case


# ---------------------------------------------------------------------------
# _snake_case helper
# ---------------------------------------------------------------------------


def test_snake_case_camel():
    assert _snake_case("ChainOfThought") == "chain_of_thought"


def test_snake_case_already_lower():
    assert _snake_case("predict") == "predict"


def test_snake_case_single_word():
    assert _snake_case("Predict") == "predict"


# ---------------------------------------------------------------------------
# DSPyTool
# ---------------------------------------------------------------------------


def test_dspy_tool_name_from_module(simple_dspy_module):
    tool = DSPyTool(simple_dspy_module, name="my_tool")
    assert tool.name == "my_tool"


def test_dspy_tool_name_inferred(simple_dspy_module):
    tool = DSPyTool(simple_dspy_module)
    assert tool.name == "mock_chain_of_thought"


def test_dspy_tool_run_string_input(simple_dspy_module, mock_dspy_prediction):
    tool = DSPyTool(simple_dspy_module, input_key="question", output_key="answer")
    result = tool._run("What is the capital of France?")
    assert result == "Paris"


def test_dspy_tool_run_json_input(simple_dspy_module, mock_dspy_prediction):
    tool = DSPyTool(simple_dspy_module, input_key="question", output_key="answer")
    result = tool._run(json.dumps({"question": "What is the capital?"}))
    # The module was called with the parsed dict
    simple_dspy_module.__call__.assert_called_with(question="What is the capital?")


def test_dspy_tool_run_returns_string(simple_dspy_module):
    tool = DSPyTool(simple_dspy_module, output_key="answer")
    result = tool.run("test input")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_dspy_tool_arun(simple_dspy_module):
    tool = DSPyTool(simple_dspy_module, output_key="answer")
    result = await tool._arun("async input")
    assert isinstance(result, str)


def test_dspy_tool_repr(simple_dspy_module):
    tool = DSPyTool(simple_dspy_module, name="my_tool")
    assert "DSPyTool" in repr(tool)
    assert "my_tool" in repr(tool)


# ---------------------------------------------------------------------------
# LangChainTool
# ---------------------------------------------------------------------------


def test_langchain_tool_name_and_desc(mock_lc_tool):
    tool = LangChainTool(mock_lc_tool)
    assert tool.name == "search"
    assert "Search" in tool.desc


def test_langchain_tool_call_via_run(mock_lc_tool):
    tool = LangChainTool(mock_lc_tool)
    result = tool("Paris weather")
    assert "Paris" in result


def test_langchain_tool_call_via_invoke():
    lc_tool = MagicMock()
    lc_tool.name = "wiki"
    lc_tool.description = "Wikipedia search"
    del lc_tool.run  # Force invoke path
    lc_tool.invoke = MagicMock(return_value="Wikipedia result")
    tool = LangChainTool(lc_tool)
    result = tool("query")
    assert result == "Wikipedia result"


@pytest.mark.asyncio
async def test_langchain_tool_acall(mock_lc_tool):
    mock_lc_tool.arun = MagicMock(return_value="async result")
    # Make arun awaitable
    import asyncio as _asyncio

    mock_lc_tool.arun = MagicMock(
        return_value=_asyncio.coroutine(lambda *a, **k: "async result")()
    )
    tool = LangChainTool(mock_lc_tool)
    result = await tool.acall("query")
    assert isinstance(result, str)


def test_langchain_tool_repr(mock_lc_tool):
    tool = LangChainTool(mock_lc_tool)
    assert "LangChainTool" in repr(tool)
    assert "search" in repr(tool)


# ---------------------------------------------------------------------------
# DSPyNode
# ---------------------------------------------------------------------------


def test_dspy_node_returns_callable(simple_dspy_module):
    node_fn = DSPyNode(simple_dspy_module)
    assert callable(node_fn)


def test_dspy_node_passes_state_to_module(simple_dspy_module, mock_dspy_prediction):
    node_fn = DSPyNode(simple_dspy_module)
    state = {"question": "What is Paris?"}
    result = node_fn(state)
    assert isinstance(result, dict)


def test_dspy_node_with_input_map(simple_dspy_module, mock_dspy_prediction):
    node_fn = DSPyNode(
        simple_dspy_module,
        input_map={"user_query": "question"},
    )
    state = {"user_query": "Tell me about Paris"}
    result = node_fn(state)
    simple_dspy_module.__call__.assert_called_with(question="Tell me about Paris")


def test_dspy_node_with_output_map(simple_dspy_module, mock_dspy_prediction):
    node_fn = DSPyNode(
        simple_dspy_module,
        output_map={"answer": "final_answer"},
    )
    state = {"question": "Capital of France?"}
    result = node_fn(state)
    assert "final_answer" in result
    assert result["final_answer"] == "Paris"


def test_dspy_node_function_name(simple_dspy_module):
    node_fn = DSPyNode(simple_dspy_module)
    assert node_fn.__name__ == "mock_chain_of_thought"
