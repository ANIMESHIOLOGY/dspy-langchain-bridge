"""Unit tests for LangChainRetriever and DSPyRetriever."""

from __future__ import annotations

import pytest

from dspy_lc_bridge.retrievers import DSPyRetriever, LangChainRetriever


# ---------------------------------------------------------------------------
# LangChainRetriever
# ---------------------------------------------------------------------------


def test_langchain_retriever_returns_prediction(mock_lc_retriever):
    retriever = LangChainRetriever(mock_lc_retriever, k=3)
    pred = retriever("What is the capital of France?")
    assert hasattr(pred, "passages")
    assert len(pred.passages) > 0


def test_langchain_retriever_respects_k(mock_lc_retriever):
    retriever = LangChainRetriever(mock_lc_retriever, k=2)
    pred = retriever("query", k=2)
    # At most k passages from each query
    assert len(pred.passages) <= 2


def test_langchain_retriever_multiple_queries(mock_lc_retriever):
    retriever = LangChainRetriever(mock_lc_retriever, k=2)
    pred = retriever(["query1", "query2"])
    # Each query returns up to k passages
    assert len(pred.passages) <= 4


def test_langchain_retriever_passages_are_strings(mock_lc_retriever):
    retriever = LangChainRetriever(mock_lc_retriever, k=3)
    pred = retriever("test query")
    assert all(isinstance(p, str) for p in pred.passages)
    assert "Paris" in pred.passages[0]


def test_langchain_retriever_repr(mock_lc_retriever):
    retriever = LangChainRetriever(mock_lc_retriever)
    assert "LangChainRetriever" in repr(retriever)


@pytest.mark.asyncio
async def test_langchain_retriever_acall(mock_lc_retriever):
    from unittest.mock import AsyncMock

    mock_lc_retriever.ainvoke = AsyncMock(return_value=mock_lc_retriever.invoke.return_value)
    retriever = LangChainRetriever(mock_lc_retriever, k=2)
    pred = await retriever.acall("async query", k=2)
    assert hasattr(pred, "passages")


# ---------------------------------------------------------------------------
# DSPyRetriever
# ---------------------------------------------------------------------------


def test_dspy_retriever_returns_documents(mock_dspy_retrieve):
    retriever = DSPyRetriever(mock_dspy_retrieve)
    docs = retriever.invoke("What is the capital of France?")
    assert len(docs) > 0


def test_dspy_retriever_returns_document_objects(mock_dspy_retrieve):
    from langchain_core.documents import Document

    retriever = DSPyRetriever(mock_dspy_retrieve)
    docs = retriever.invoke("query")
    assert all(isinstance(d, Document) for d in docs)


def test_dspy_retriever_page_content_populated(mock_dspy_retrieve):
    retriever = DSPyRetriever(mock_dspy_retrieve)
    docs = retriever.invoke("France capital")
    assert all(len(d.page_content) > 0 for d in docs)
    assert "Paris" in docs[0].page_content


def test_dspy_retriever_legacy_interface(mock_dspy_retrieve):
    """get_relevant_documents should also work (LangChain legacy)."""
    retriever = DSPyRetriever(mock_dspy_retrieve)
    docs = retriever.get_relevant_documents("query")
    assert len(docs) > 0


def test_dspy_retriever_repr(mock_dspy_retrieve):
    retriever = DSPyRetriever(mock_dspy_retrieve)
    assert "DSPyRetriever" in repr(retriever)


@pytest.mark.asyncio
async def test_dspy_retriever_ainvoke(mock_dspy_retrieve):
    retriever = DSPyRetriever(mock_dspy_retrieve)
    docs = await retriever.ainvoke("async query")
    assert len(docs) > 0
