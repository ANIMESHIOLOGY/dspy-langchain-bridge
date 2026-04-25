"""Integration tests for LangGraph + DSPy agent interop.

Requires real LLM API calls.
Run with: pytest tests/integration/ -m slow
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
@pytest.mark.integration
def test_langgraph_agent_with_dspy_tool() -> None:
    """LangGraph agent that uses a DSPy ChainOfThought module as a tool."""
    pytest.importorskip("langgraph")

    import dspy
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, StateGraph
    from typing_extensions import TypedDict

    from dspy_lc_bridge import DSPyNode, LangChainLM

    llm = ChatOpenAI(model="gpt-4o-mini")
    dspy.settings.configure(lm=LangChainLM(llm))

    class AgentState(TypedDict):
        question: str
        answer: str

    cot = dspy.ChainOfThought("question -> answer")
    reason_node = DSPyNode(
        cot,
        input_map={"question": "question"},
        output_map={"answer": "answer"},
    )

    graph = StateGraph(AgentState)
    graph.add_node("reason", reason_node)
    graph.set_entry_point("reason")
    graph.add_edge("reason", END)

    app = graph.compile()
    result = app.invoke({"question": "What is the capital of France?", "answer": ""})
    assert "Paris" in result.get("answer", "")


@pytest.mark.slow
@pytest.mark.integration
def test_dspy_module_as_langgraph_tool() -> None:
    """DSPy module used as a LangChain tool inside a LangGraph ReAct agent."""
    pytest.importorskip("langgraph")

    import dspy
    from langchain_openai import ChatOpenAI

    from dspy_lc_bridge import DSPyTool, LangChainLM

    llm = ChatOpenAI(model="gpt-4o-mini")
    dspy.settings.configure(lm=LangChainLM(llm))

    summarizer = dspy.ChainOfThought("document -> summary")
    tool = DSPyTool(
        summarizer,
        name="summarize",
        description="Summarize a document.",
        input_key="document",
        output_key="summary",
    )

    result = tool.run("France is a country in Western Europe. Paris is its capital.")
    assert isinstance(result, str)
    assert len(result) > 0
