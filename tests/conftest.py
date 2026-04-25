"""Shared pytest fixtures for dspy-langchain-bridge tests.

All LLM calls are mocked — no real API keys are needed for unit tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# DSPy fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_dspy_prediction():
    """A minimal mock of a dspy.Prediction object."""
    pred = MagicMock()
    pred.answer = "Paris"
    pred.reasoning = "France's capital is Paris."
    pred._store = {"answer": "Paris", "reasoning": "France's capital is Paris."}
    pred.items = lambda: pred._store.items()
    return pred


@pytest.fixture()
def simple_dspy_module(mock_dspy_prediction):
    """A DSPy module that always returns a fixed Prediction."""
    module = MagicMock()
    module.__class__.__name__ = "MockChainOfThought"
    module.return_value = mock_dspy_prediction
    module.__call__ = MagicMock(return_value=mock_dspy_prediction)

    # Minimal signature
    sig = MagicMock()
    sig.input_fields = {"question": MagicMock(json_schema_extra={"desc": "The question"})}
    sig.output_fields = {"answer": MagicMock(json_schema_extra={"desc": "The answer"})}
    sig.__doc__ = "Answer a question."
    module.signature = sig

    return module


@pytest.fixture()
def mock_dspy_signature():
    """A minimal dspy.Signature mock with one input and one output field."""
    sig = MagicMock()
    sig.input_fields = {
        "question": MagicMock(json_schema_extra={"desc": "The question to answer"})
    }
    sig.output_fields = {
        "answer": MagicMock(json_schema_extra={"desc": "The answer"})
    }
    sig.__doc__ = "Answer the given question."
    return sig


# ---------------------------------------------------------------------------
# LangChain LLM fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_chat_llm():
    """A mock LangChain ChatModel that returns a fixed AIMessage."""
    from langchain_core.messages import AIMessage

    llm = MagicMock()
    llm.__class__.__name__ = "MockChatOpenAI"
    response = AIMessage(content="The answer is 42.")
    llm.invoke = MagicMock(return_value=response)
    llm.ainvoke = MagicMock(return_value=response)
    return llm


@pytest.fixture()
def mock_completion_llm():
    """A mock LangChain completion LLM."""
    llm = MagicMock()
    llm.__class__.__name__ = "MockLLM"
    llm.invoke = MagicMock(return_value="The answer is 42.")
    return llm


# ---------------------------------------------------------------------------
# LangChain retriever fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_lc_retriever():
    """A mock LangChain BaseRetriever."""
    from langchain_core.documents import Document

    retriever = MagicMock()
    docs = [
        Document(page_content="Paris is the capital of France.", metadata={}),
        Document(page_content="France is in Western Europe.", metadata={}),
        Document(page_content="The Eiffel Tower is in Paris.", metadata={}),
    ]
    retriever.invoke = MagicMock(return_value=docs)
    retriever.get_relevant_documents = MagicMock(return_value=docs)
    return retriever


@pytest.fixture()
def mock_dspy_retrieve(mock_dspy_prediction):
    """A mock DSPy Retrieve module."""

    retrieve = MagicMock()
    retrieve.__class__.__name__ = "MockRetrieve"
    retrieve.k = 3
    passages_pred = MagicMock()
    passages_pred.passages = [
        "Paris is the capital of France.",
        "The Eiffel Tower is in Paris.",
    ]
    retrieve.return_value = passages_pred
    retrieve.__call__ = MagicMock(return_value=passages_pred)
    return retrieve


# ---------------------------------------------------------------------------
# LangChain memory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_lc_memory():
    """A mock LangChain ConversationBufferMemory."""
    from langchain_core.messages import AIMessage, HumanMessage

    memory = MagicMock()
    memory.__class__.__name__ = "ConversationBufferMemory"
    chat_memory = MagicMock()
    chat_memory.messages = [
        HumanMessage(content="What is the capital of France?"),
        AIMessage(content="The capital of France is Paris."),
    ]
    memory.chat_memory = chat_memory
    memory.save_context = MagicMock()
    memory.clear = MagicMock()
    return memory


# ---------------------------------------------------------------------------
# LangChain tool fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_lc_tool():
    """A mock LangChain BaseTool."""
    tool = MagicMock()
    tool.name = "search"
    tool.description = "Search the web for information."
    tool.run = MagicMock(return_value="Search result: Paris is the capital of France.")
    tool.invoke = MagicMock(return_value="Search result: Paris is the capital of France.")
    return tool
