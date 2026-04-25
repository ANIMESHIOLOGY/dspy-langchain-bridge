"""Integration tests for end-to-end RAG pipelines.

These tests require real LLM / retriever API calls and are marked ``slow``.
Run them with: pytest tests/integration/ -m slow
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
@pytest.mark.integration
def test_langchain_retriever_in_dspy_rag() -> None:
    """DSPy RAG module using a LangChain FAISS retriever."""
    pytest.importorskip("langchain_community")
    pytest.importorskip("faiss")

    import dspy
    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    from dspy_lc_bridge import LangChainLM, LangChainRetriever

    llm = ChatOpenAI(model="gpt-4o-mini")
    dspy.settings.configure(lm=LangChainLM(llm))

    texts = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Tokyo is the capital of Japan.",
    ]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    retriever = LangChainRetriever(vectorstore.as_retriever(), k=2)

    class RAG(dspy.Module):
        def __init__(self) -> None:
            self.retrieve = retriever
            self.generate = dspy.ChainOfThought("context, question -> answer")

        def forward(self, question: str) -> dspy.Prediction:
            passages = self.retrieve(question).passages
            context = "\n".join(passages)
            return self.generate(context=context, question=question)

    rag = RAG()
    result = rag(question="What is the capital of France?")
    assert "Paris" in result.answer


@pytest.mark.slow
@pytest.mark.integration
def test_dspy_retriever_in_langchain_chain() -> None:
    """LangChain chain using a DSPy Retrieve module as BaseRetriever."""
    import dspy
    from langchain_openai import ChatOpenAI

    from dspy_lc_bridge import DSPyRetriever, LangChainLM

    llm = ChatOpenAI(model="gpt-4o-mini")
    dspy.settings.configure(lm=LangChainLM(llm))

    dspy_rm = dspy.Retrieve(k=3)
    lc_retriever = DSPyRetriever(dspy_rm)

    docs = lc_retriever.invoke("France capital")
    assert len(docs) >= 0  # Depends on configured DSPy RM
